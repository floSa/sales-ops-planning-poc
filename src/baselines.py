"""
Baseline à battre : naïve saisonnière hebdomadaire (lag-7) — cadrage §9.

Version vectorisée de Retail/Sales_Forecasting/src/baselines.py (l'original
itère ligne à ligne, intenable sur ~22 000 lignes × pli).

Pour un horizon h > 7 jours, la référence est la valeur du même jour de
semaine de la DERNIÈRE semaine connue avant le cutoff (soit d - 7·⌈h/7⌉),
avec repli sur la moyenne des 7 derniers jours connus de la série.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset import KEYS


def seasonal_naive(df_hist: pd.DataFrame, df_eval: pd.DataFrame,
                   cutoff: pd.Timestamp, m: int = 7,
                   target: str = "quantity") -> np.ndarray:
    """
    Args:
        df_hist : historique réel (KEYS + date + target), dates <= cutoff.
        df_eval : lignes à prédire (KEYS + date), dates > cutoff.
    Returns:
        prédictions alignées sur df_eval (échelle réelle).
    """
    ev = df_eval[KEYS + ["date"]].copy()
    h = (ev["date"] - cutoff).dt.days
    ev["ref_date"] = ev["date"] - pd.to_timedelta(m * np.ceil(h / m).astype(int), unit="D")

    lookup = df_hist.set_index(KEYS + ["date"])[target]
    pred = lookup.reindex(pd.MultiIndex.from_frame(ev[KEYS + ["ref_date"]])).values.astype(float)

    # repli : moyenne des m derniers jours connus de la série
    fallback = (df_hist.sort_values("date").groupby(KEYS, observed=True)[target]
                .apply(lambda s: s.tail(m).mean()))
    fb = fallback.reindex(pd.MultiIndex.from_frame(ev[KEYS])).values
    return np.where(np.isfinite(pred), pred, fb)
