"""
Cascade de calcul du brief (Bloc A, cadrage §3) :

    PV (Prix de Vente) × PA (Panier Article) -> PM (Panier Moyen)
    (PM + inflation) × nb transactions -> CA net

reconstruite AU-DESSUS des prévisions SKU × jour agrégées (recommandation
wiki §0.2 : prévoir le volume, reconstituer le CA) :

  - volume(s,j)       = Σ_SKU quantité prévue
  - CA_modèle(s,j)    = Σ_SKU quantité × prix -> PV(s,j) = CA / volume
  - PA(s,j) projeté   = profil historique lissé par magasin × jour de semaine
                        (moyenne des N dernières semaines, 🟡 pas de 2e modèle)
  - transactions(s,j) = volume / PA
  - PM(s,j)           = PV × PA
  - CA net(s,j)       = PM × (1 + Δinflation) × transactions × coefficients

Hypothèses CDG (saisie à la maille MOIS, brief 1.1) : coefficients
multiplicatifs sur PV / PA / transactions + Δ d'inflation (%). ⚠️ Les prix
prévus embarquent déjà la trajectoire CPI projetée : le paramètre inflation
de la cascade est un DELTA d'hypothèse centrale (défaut 0), pas l'inflation
totale — sinon double compte.

Par construction, coefficients = 1 et Δinflation = 0 => CA net == CA modèle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

HYPOTHESIS_COLS = ["coef_pv", "coef_pa", "coef_transactions", "inflation_delta_pct"]


def default_hypotheses(months) -> pd.DataFrame:
    """Hypothèses neutres (pré-remplissage 'proposé par l'outil')."""
    return pd.DataFrame({"month": list(months), "coef_pv": 1.0, "coef_pa": 1.0,
                         "coef_transactions": 1.0, "inflation_delta_pct": 0.0})


def project_pa(sales_hist: pd.DataFrame, traffic_hist: pd.DataFrame,
               n_weeks: int = config.PA_SMOOTHING_WEEKS) -> pd.DataFrame:
    """
    PA (articles/ticket) projeté par magasin × jour de semaine : moyenne des
    n_weeks dernières semaines observées. 🟡 Hypothèse V1 (cadrage) : pas de
    modèle dédié aux transactions, le PA est structurellement stable.
    """
    qty = (sales_hist.groupby(["store_id", "date"], observed=True)["quantity"]
           .sum().reset_index())
    m = qty.merge(traffic_hist, on=["store_id", "date"], how="inner")
    m = m[m.nb_tickets > 0]
    cutoff = m["date"].max() - pd.Timedelta(weeks=n_weeks)
    recent = m[m["date"] > cutoff].copy()
    recent["day_of_week"] = recent["date"].dt.dayofweek
    recent["pa"] = recent["quantity"] / recent["nb_tickets"]
    pa = (recent.groupby(["store_id", "day_of_week"], observed=True)["pa"]
          .mean().rename("pa_proj").reset_index())
    # repli : PA moyen du magasin (jours de semaine sans observation récente)
    store_mean = recent.groupby("store_id", observed=True)["pa"].mean().rename("pa_store")
    return pa, store_mean


def cascade(forecast_daily: pd.DataFrame, sales_hist: pd.DataFrame,
            traffic_hist: pd.DataFrame,
            hypotheses: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Applique la cascade aux prévisions journalières SKU et renvoie le détail
    magasin × jour : volume, pv, pa, pm, transactions, ca_net (+ ca_modele).

    Args:
        forecast_daily : sortie de src.forecast (quantity_pred, ca_pred).
        hypotheses     : DataFrame month × HYPOTHESIS_COLS (défaut : neutre).
    """
    agg = (forecast_daily.groupby(["store_id", "date"], observed=True)
           .agg(volume=("quantity_pred", "sum"), ca_modele=("ca_pred", "sum"))
           .reset_index())
    agg["pv"] = np.where(agg.volume > 0, agg.ca_modele / agg.volume, 0.0)
    agg["day_of_week"] = agg["date"].dt.dayofweek

    pa, store_mean = project_pa(sales_hist, traffic_hist)
    agg = agg.merge(pa, on=["store_id", "day_of_week"], how="left")
    agg = agg.merge(store_mean, on="store_id", how="left")
    agg["pa_proj"] = agg["pa_proj"].fillna(agg["pa_store"])
    agg = agg.drop(columns=["pa_store"])

    agg["month"] = agg["date"].dt.to_period("M").astype(str)
    hyp = hypotheses if hypotheses is not None else default_hypotheses(agg.month.unique())
    agg = agg.merge(hyp, on="month", how="left")
    agg[HYPOTHESIS_COLS] = agg[HYPOTHESIS_COLS].fillna(
        {"coef_pv": 1.0, "coef_pa": 1.0, "coef_transactions": 1.0, "inflation_delta_pct": 0.0})

    # --- cascade du brief ---
    agg["pv_adj"] = agg.pv * agg.coef_pv
    agg["pa_adj"] = agg.pa_proj * agg.coef_pa
    agg["pm"] = agg.pv_adj * agg.pa_adj * (1.0 + agg.inflation_delta_pct / 100.0)
    agg["transactions"] = np.where(agg.pa_adj > 0,
                                   agg.volume / agg.pa_adj, 0.0) * agg.coef_transactions
    agg["ca_net"] = agg.pm * agg.transactions
    return agg


def cascade_monthly(cascade_daily: pd.DataFrame) -> pd.DataFrame:
    """Vue mensuelle par magasin (l'horizon de pilotage du CDG)."""
    m = (cascade_daily.groupby(["store_id", "month"], observed=True)
         .agg(volume=("volume", "sum"), ca_modele=("ca_modele", "sum"),
              ca_net=("ca_net", "sum"), transactions=("transactions", "sum"))
         .reset_index())
    m["pm"] = np.where(m.transactions > 0, m.ca_net / m.transactions, 0.0)
    m["pa"] = np.where(m.transactions > 0, m.volume / m.transactions, 0.0)
    m["pv"] = np.where(m.volume > 0, m.ca_net / m.volume, 0.0)
    return m
