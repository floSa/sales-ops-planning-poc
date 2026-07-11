"""
Moteur de simulation du dashboard (Bloc B) — saisie d'une campagne promo.

Le brief (1.2) demande : « l'outil propose une variation prix/quantité basée
sur l'historique, modifiable par l'utilisateur ». Ici :

  1. `typology_uplift_reference` mesure dans l'HISTORIQUE l'uplift médian par
     SKU de chaque typologie (méthode baseline vs promo, wiki §9.1 — c'est une
     approximation POC : baseline = ventes hors promo du même SKU, pas un vrai
     contrefactuel modélisé).
  2. `propose_uplift` en déduit la proposition pour une campagne saisie
     (l'élasticité des promos 'produits' est rapportée à la profondeur de
     remise saisie).
  3. `apply_promo_to_forecast` applique l'uplift (modifié ou non par le CDG)
     aux lignes de prévision ciblées : quantités × uplift, prix × (1 − remise)
     pour les typologies remisées, recalcul du CA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def typology_uplift_reference(sales: pd.DataFrame, promos: pd.DataFrame,
                              scope: pd.DataFrame) -> pd.DataFrame:
    """Uplift médian par SKU et remise moyenne observés par typologie."""
    s = sales.merge(promos[["promo_id", "promo_type", "discount_rate"]],
                    on="promo_id", how="left")
    rows = []
    for ptype, g in s.dropna(subset=["promo_type"]).groupby("promo_type"):
        per_sku_promo = g.groupby("sku_id", observed=True)["quantity"].mean()
        base = (sales[sales.promo_id.isna() & sales.sku_id.isin(per_sku_promo.index)]
                .groupby("sku_id", observed=True)["quantity"].mean())
        ratio = (per_sku_promo / base.reindex(per_sku_promo.index)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(promo_type=ptype,
                         uplift_median=float(ratio.median()),
                         discount_mean=float(g.discount_rate.mean()),
                         n_sku=len(ratio)))
    return pd.DataFrame(rows)


def propose_uplift(ref: pd.DataFrame, promo_type: str, discount_rate: float) -> float:
    """
    Proposition d'uplift pour une campagne saisie. Pour 'produits', l'uplift
    historique est rapporté à la remise saisie via une élasticité implicite
    (uplift = 1 + e × remise) ; pour les autres typologies, uplift médian
    historique tel quel.
    """
    r = ref[ref.promo_type == promo_type]
    if r.empty:
        return 1.0
    up, disc = float(r.uplift_median.iloc[0]), float(r.discount_mean.iloc[0])
    if promo_type == "produits" and disc > 0 and discount_rate > 0:
        elasticity = (up - 1.0) / disc
        return round(1.0 + elasticity * discount_rate, 2)
    return round(up, 2)


def apply_promo_to_forecast(forecast: pd.DataFrame, *, promo_type: str,
                            date_start, date_end, perimeter: str,
                            sku_ids: list[str] | None, uplift: float,
                            discount_rate: float = 0.0,
                            stores: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Retourne une COPIE de la prévision avec la campagne simulée appliquée.

    - typologies 'seuils'/'cadeau_seuil'/'ouverture_magasin' : tous les SKU
      du périmètre ; les autres : SKU ciblés uniquement.
    - 'influence' : courbe décalée (pic vers J+10, extinction J+28), comme la
      signature observée dans l'historique.
    """
    fc = forecast.copy()
    dates = pd.to_datetime(fc["date"])
    d0, d1 = pd.Timestamp(date_start), pd.Timestamp(date_end)

    mask = pd.Series(True, index=fc.index)
    if perimeter == "magasin":
        mask &= fc.store_id != config.ONLINE_STORE_ID
    elif perimeter == "online":
        mask &= fc.store_id == config.ONLINE_STORE_ID
    if sku_ids and promo_type not in ("seuils", "cadeau_seuil", "ouverture_magasin"):
        mask &= fc.sku_id.isin(sku_ids)

    if promo_type == "influence":
        t = (dates - d0).dt.days
        curve = 1.0 + (uplift - 1.0) * np.exp(-((t - 10) / 8.0) ** 2)
        factor = np.where((t >= 0) & (t <= 28) & mask, curve, 1.0)
    else:
        in_window = (dates >= d0) & (dates <= d1)
        factor = np.where(in_window & mask, uplift, 1.0)

    fc["quantity_pred"] = fc["quantity_pred"] * factor
    if discount_rate > 0 and promo_type == "produits":
        in_window = (dates >= d0) & (dates <= d1)
        fc.loc[in_window & mask, "unit_price"] *= (1.0 - discount_rate)
    fc["ca_pred"] = fc["quantity_pred"] * fc["unit_price"]
    return fc
