"""
Inférence — adaptation de Retail/Sales_Forecasting/src/inferencing.py.

Deux stratégies, combinées en hybride dans le backtest (architecture reprise) :
  - DIRECTE : features dynamiques calculées avec la cible masquée après le
    cutoff (les lags courts sont NaN loin du cutoff, gérés par les GBM).
  - RÉCURSIVE : jour par jour, chaque prédiction réinjectée dans les lags.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.dataset import KEYS
from src.features_dynamic import add_dynamic_features
from src.modeling import predict_models


def direct_predict(models: Dict[str, Any], panel_masked: pd.DataFrame,
                   eval_index: pd.Index, features: List[str]) -> np.ndarray:
    """panel_masked = historique + fenêtre à prédire avec quantity=NaN,
    trié par (KEYS, date). Retourne les prédictions alignées sur eval_index."""
    dyn = add_dynamic_features(panel_masked)
    X = dyn.loc[eval_index, features]
    return predict_models(models, X)


def recursive_predict(models: Dict[str, Any], df_eval: pd.DataFrame,
                      df_hist_window: pd.DataFrame, features: List[str],
                      target_col: str = "quantity") -> pd.Series:
    """
    Prévision récursive J+1 -> J+h : prédiction, réinjection dans la cible,
    recalcul des features dynamiques, jour suivant.

    Args:
        df_eval : lignes à prédire (features statiques prêtes, cible NaN).
        df_hist_window : historique récent (>= 364 j pour lag_364).
    Returns:
        prédictions indexées comme df_eval.
    """
    df_eval = df_eval.sort_values(KEYS + ["date"]).copy()
    df_eval[target_col] = np.nan
    hist_cols = list(df_eval.columns)
    hist = df_hist_window[hist_cols].copy()

    for d in sorted(df_eval["date"].unique()):
        past_preds = df_eval[df_eval["date"] < d]
        today = df_eval[df_eval["date"] == d]
        combined = pd.concat([hist, past_preds, today]).sort_values(KEYS + ["date"])
        combined = add_dynamic_features(combined, target_col=target_col)
        X_today = combined.loc[today.index, features]
        df_eval.loc[today.index, target_col] = predict_models(models, X_today)

    return df_eval[target_col]
