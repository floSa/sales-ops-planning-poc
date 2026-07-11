"""
Analyse d'écarts CA réel vs prévision (Bloc D — brief use case 2).

Décomposition par magasin × mois (Online = entité agrégée, cadrage §6) en
effets prix / volume / mix / promo / calendaire, méthode séquentielle
documentée (l'ordre des effets est une convention, figée ici) :

  écart total = CA_réel − CA_prévu
    1. EFFET PRIX    = Σ_SKU (p_réel − p_prévu) × q_réel
       (valorisation de l'écart de prix aux quantités réelles)
    2. EFFET QUANTITÉ = Σ_SKU p_prévu × (q_réel − q_prévu), scindé en :
       a. EFFET VOLUME = PV_prévu_moyen × (Q_réel − Q_prévu)   [Q = Σ quantités]
       b. EFFET MIX    = effet quantité − effet volume
          (déformation de la composition du panier à prix prévus constants)
  L'effet quantité est ensuite REVENTILÉ par driver (lecture métier) :
       - dont PROMO      : lignes sous promotion (toutes typologies),
       - dont CALENDAIRE : lignes hors promo sur jours à événement calendaire
         (férié, veille de férié, pont),
       - dont AUTRE      : le reste (tendance, météo, aléa).

⚠️ Dans ce POC, prix réels == prix prévus (le calendrier promo et le CPI sont
connus à l'avance dans le synthétique) : l'effet prix réel/prévision est ~0.
Il devient non trivial dans la comparaison scénario ajusté vs baseline
(coefficient PV de la cascade) — même fonction, autres entrées.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

EFFECT_COLS = ["effet_prix", "effet_volume", "effet_mix",
               "dont_promo", "dont_calendaire", "dont_autre"]


def decompose(df: pd.DataFrame, group_cols=("store_id", "month")) -> pd.DataFrame:
    """
    Décompose l'écart de CA entre deux jeux de quantités/prix par SKU × jour.

    Args:
        df : une ligne par (store_id, sku_id, date) avec q_reel, p_reel,
             q_prev, p_prev, is_promo (0/1), is_calendar (0/1).
    Returns:
        une ligne par groupe avec CA, écart total et les 6 effets
        (identité : écart == prix + volume + mix ; volume + mix == promo +
        calendaire + autre).
    """
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["ca_reel"] = d.q_reel * d.p_reel
    d["ca_prev"] = d.q_prev * d.p_prev
    d["effet_prix_l"] = (d.p_reel - d.p_prev) * d.q_reel
    d["effet_qte_l"] = d.p_prev * (d.q_reel - d.q_prev)

    g = list(group_cols)
    agg = d.groupby(g, observed=True).agg(
        ca_reel=("ca_reel", "sum"), ca_prev=("ca_prev", "sum"),
        effet_prix=("effet_prix_l", "sum"), effet_qte=("effet_qte_l", "sum"),
        q_reel=("q_reel", "sum"), q_prev=("q_prev", "sum"),
    ).reset_index()

    # PV prévu moyen du groupe -> effet volume pur, le mix est le résidu
    pv_prev = np.where(agg.q_prev > 0, agg.ca_prev / agg.q_prev, 0.0)
    agg["effet_volume"] = pv_prev * (agg.q_reel - agg.q_prev)
    agg["effet_mix"] = agg.effet_qte - agg.effet_volume

    # reventilation de l'effet quantité par driver
    drivers = (d.assign(
        promo_l=lambda x: x.effet_qte_l * x.is_promo,
        cal_l=lambda x: x.effet_qte_l * ((1 - x.is_promo) * x.is_calendar))
        .groupby(g, observed=True)[["promo_l", "cal_l"]].sum().reset_index())
    agg = agg.merge(drivers, on=g, how="left")
    agg["dont_promo"] = agg.pop("promo_l")
    agg["dont_calendaire"] = agg.pop("cal_l")
    agg["dont_autre"] = agg.effet_qte - agg.dont_promo - agg.dont_calendaire

    agg["ecart_total"] = agg.ca_reel - agg.ca_prev
    check = (agg.ecart_total - (agg.effet_prix + agg.effet_volume + agg.effet_mix)).abs()
    assert check.max() < 0.01, "décomposition non exacte"
    return agg.drop(columns=["effet_qte", "q_reel", "q_prev"])


def build_backtest_ecarts(scenario: int = 1) -> pd.DataFrame:
    """
    Construit la décomposition réel vs prévision sur la fenêtre de backtest
    (les 4 plis couvrent les ~4 derniers mois d'historique).
    """
    bt = pd.read_parquet(config.RESULTS_DIR / "backtest" / f"scenario{scenario}"
                         / "results.parquet")
    sales = pd.read_parquet(config.DATA_TRX / "sales_daily.parquet")
    sales["date"] = pd.to_datetime(sales["date"])
    bt["date"] = pd.to_datetime(bt["date"])

    d = bt.merge(sales[["store_id", "sku_id", "date", "unit_price", "promo_id"]],
                 on=["store_id", "sku_id", "date"], how="left")
    # calendrier : férié / veille / pont
    from src.dataset import build_calendar
    cal = build_calendar(pd.DatetimeIndex(sorted(d.date.unique())))
    cal["is_calendar"] = ((cal.is_holiday == 1) | (cal.is_day_before_holiday == 1)
                          | (cal.is_pont == 1)).astype(int)
    d = d.merge(cal[["date", "is_calendar"]], on="date", how="left")

    d = d.rename(columns={"y_true": "q_reel", "pred_hyb": "q_prev"})
    d["p_reel"] = d["unit_price"].astype(float)
    d["p_prev"] = d["p_reel"]  # prix connus à l'avance dans ce POC (cf. docstring)
    d["is_promo"] = d["promo_id"].notna().astype(int)
    out = decompose(d)
    out_dir = config.RESULTS_DIR / "ecarts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"ecarts_scenario{scenario}.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    print(build_backtest_ecarts(a.scenario).head(20).to_string(index=False))
