"""
Modélisation GBM — adaptation de Retail/Sales_Forecasting/src/modeling.py.

Architecture reprise : GBM avec loss Tweedie + sample weights sur ruptures.
Deux adaptations assumées (documentées dans le README) :
  1. Cible en ÉCHELLE RÉELLE (pas de log1p) : Tweedie gère nativement zéros
     et asymétrie, on évite le biais de ré-exponentiation.
  2. MODEL_MODE="fast" (LightGBM seul) par défaut : on passe de 9 séries à
     ~1 560 séries × 1,4 M lignes, l'ensemble 3-GBM × 4 plis serait long pour
     un POC. L'ensemble XGB+LGBM+CatBoost reste disponible (MODEL_MODE="ensemble").

Les encodages catégoriels sont des codes entiers (convention Sales_Forecasting),
traités en numérique par les trois librairies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.nansum(np.abs(y_true))
    return float(np.nansum(np.abs(y_true - y_pred)) / denom) if denom else float("nan")


def bias(y_true, y_pred) -> float:
    """Biais moyen en unités (> 0 = sur-prévision systématique)."""
    return float(np.nanmean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def bias_pct(y_true, y_pred) -> float:
    """Biais relatif (Σ prévu / Σ réel − 1) — lecture directe pour un CDG."""
    denom = np.nansum(np.asarray(y_true, float))
    return float(np.nansum(np.asarray(y_pred, float)) / denom - 1.0) if denom else float("nan")


DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    # dimensionné pour ~1,4 M lignes × 4 plis × 2 scénarios sur un poste de
    # travail : 300 arbres × 63 feuilles suffisent largement sur le synthétique
    "lgbm": dict(objective="tweedie", tweedie_variance_power=config.TWEEDIE_VARIANCE_POWER,
                 n_estimators=300, learning_rate=0.10, num_leaves=63,
                 min_child_samples=60, subsample=0.9, subsample_freq=1,
                 colsample_bytree=0.9, reg_lambda=1.0, n_jobs=-1, verbose=-1,
                 random_state=config.SEED),
    "xgb": dict(objective="reg:tweedie", tweedie_variance_power=config.TWEEDIE_VARIANCE_POWER,
                n_estimators=600, learning_rate=0.06, max_depth=8,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                n_jobs=-1, random_state=config.SEED, verbosity=0),
    "cat": dict(loss_function=f"Tweedie:variance_power={config.TWEEDIE_VARIANCE_POWER}",
                iterations=600, learning_rate=0.06, depth=8,
                random_seed=config.SEED, verbose=0, allow_writing_files=False),
}


def train_models(X: pd.DataFrame, y: pd.Series,
                 weights: Optional[pd.Series] = None,
                 mode: str | None = None,
                 params_overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Entraîne le(s) GBM selon le mode. Retourne {nom: modèle}."""
    mode = mode or config.MODEL_MODE
    overrides = params_overrides or {}
    models: Dict[str, Any] = {}

    import lightgbm as lgb
    p = {**DEFAULT_PARAMS["lgbm"], **overrides.get("lgbm", {})}
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y, sample_weight=weights)
    models["lgbm"] = m

    if mode == "ensemble":
        import xgboost as xgb
        from catboost import CatBoostRegressor
        p = {**DEFAULT_PARAMS["xgb"], **overrides.get("xgb", {})}
        m = xgb.XGBRegressor(**p)
        m.fit(X, y, sample_weight=weights)
        models["xgb"] = m
        p = {**DEFAULT_PARAMS["cat"], **overrides.get("cat", {})}
        m = CatBoostRegressor(**p)
        m.fit(X, y, sample_weight=weights)
        models["cat"] = m
    return models


def predict_models(models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Prédiction (moyenne d'ensemble si plusieurs modèles), clip à zéro
    (post-traitement métier standard, wiki §6.5)."""
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return np.maximum(preds.mean(axis=1), 0.0)
