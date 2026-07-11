"""
Features DYNAMIQUES (dépendantes de l'historique de la cible) — adaptation de
Retail/Sales_Forecasting/src/features_dynamic_engineering.py.

Recalculées à chaque pli de backtest et à chaque pas d'inférence récursive :
elles ne doivent JAMAIS voir le futur (shift(1) systématique sur les rollings).

Différence assumée vs Sales_Forecasting : la cible reste en ÉCHELLE RÉELLE
(pas de log1p) — la loss Tweedie gère nativement les zéros et l'asymétrie,
et on évite le biais de ré-exponentiation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset import KEYS

LAG_DAYS = [1, 7, 14, 28, 364]
ROLLING_WINDOWS = [7, 28]
EWMA_SPANS = [7]

DYNAMIC_FEATURES = ([f"lag_{l}" for l in LAG_DAYS]
                    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
                    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
                    + [f"ewma_{s}" for s in EWMA_SPANS])


def add_dynamic_features(df: pd.DataFrame, target_col: str = "quantity") -> pd.DataFrame:
    """df doit être trié par (store_id, sku_id, date) et à grille journalière
    continue par série (garanti par dataset.build_panel)."""
    df = df.copy()
    g = df.groupby(KEYS, observed=True)[target_col]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        roll = shifted.groupby([df[k] for k in KEYS], observed=True).rolling(w, min_periods=max(2, w // 4))
        # .values : le rolling groupby renvoie un MultiIndex (clés, index d'origine) ;
        # df étant trié par clés+date, l'ordre concaténé des groupes == l'ordre du df,
        # l'assignation positionnelle est correcte même si l'index n'est pas 0..n-1
        df[f"roll_mean_{w}"] = roll.mean().values
        df[f"roll_std_{w}"] = roll.std().values

    for span in EWMA_SPANS:
        df[f"ewma_{span}"] = (shifted.groupby([df[k] for k in KEYS], observed=True)
                              .transform(lambda x: x.ewm(span=span, min_periods=2).mean()))
    return df
