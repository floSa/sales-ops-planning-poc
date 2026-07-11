"""
Backtest rolling-origin multi-plis (Bloc A) — extension de
Retail/Sales_Forecasting/src/backtest.py, conformément au cadrage §9 :
WAPE + biais (jamais le MAPE), déclinés par magasin / catégorie / SKU /
horizon, contre la baseline naïve saisonnière hebdomadaire (lag-7).

Pour chaque pli : réentraînement sur le passé, évaluation de l'horizon
suivant en directe / récursive / hybride (α calibré par pli, médiane retenue).

Usage :
    python -m src.backtest --scenario 1 [--mode fast|ensemble] [--folds 4] [--horizon 28]

Scénarios (brief 1.1) :
    1 = baseline : calendaire + promos, sans signaux externes.
    2 = baseline + IA : ajout des features météo (anomalie de température, pluie).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src import config
from src.baselines import seasonal_naive
from src.dataset import KEYS, build_panel
from src.features_dynamic import add_dynamic_features
from src.inferencing import direct_predict, recursive_predict
from src.modeling import bias, bias_pct, train_models, wape


def _metrics_row(name: str, y, p) -> dict:
    return dict(modele=name, WAPE=wape(y, p), biais=bias(y, p), biais_pct=bias_pct(y, p))


def run_backtest(scenario: int = 1, mode: str | None = None,
                 n_folds: int = config.BACKTEST_N_FOLDS,
                 horizon: int = config.BACKTEST_HORIZON_DAYS,
                 out_dir=None) -> Dict[str, Any]:
    t0 = time.time()
    include_weather = scenario == 2
    print(f"=== Backtest scénario {scenario} "
          f"({'avec' if include_weather else 'sans'} météo), "
          f"mode {mode or config.MODEL_MODE}, {n_folds} plis × {horizon} j ===")

    panel, features, _ = build_panel(config.HIST_START, config.HIST_END,
                                     include_weather=include_weather)
    assert not panel.quantity.isna().any(), "trous dans le panel historique"
    print(f"Panel : {len(panel):,} lignes, {len(features)} features")

    # features dynamiques des lignes de TRAIN : calculées une fois sur tout
    # l'historique (lags strictement rétrospectifs -> pas de fuite pour le train)
    panel_dyn = add_dynamic_features(panel)
    last = panel["date"].max()

    all_results, fold_metrics, alphas_folds = [], [], []
    alph_grid = np.round(np.linspace(0.0, 1.0, 11), 2)

    for i in range(n_folds, 0, -1):
        cut = last - pd.Timedelta(days=horizon * i)
        fold_no = n_folds - i + 1
        val_mask = (panel["date"] > cut) & (panel["date"] <= cut + pd.Timedelta(days=horizon))
        val = panel[val_mask]
        train = panel_dyn[panel_dyn["date"] <= cut]
        y_true = val["quantity"].values.astype(float)
        print(f"\n[Pli {fold_no}] train <= {cut.date()} ({len(train):,} l.) | "
              f"val {val['date'].min().date()} -> {val['date'].max().date()} ({len(val):,} l.)")

        feats = features + [c for c in add_dynamic_cols() if c in train.columns]
        models = train_models(train[feats], train["quantity"],
                              weights=train["sample_weight"], mode=mode)

        # --- directe : cible masquée après le cutoff ---
        masked = panel[panel["date"] <= cut + pd.Timedelta(days=horizon)].copy()
        masked.loc[masked["date"] > cut, "quantity"] = np.nan
        pred_dir = direct_predict(models, masked, val.index, feats)

        # --- récursive ---
        hist_window = panel[(panel["date"] > cut - pd.Timedelta(days=config.HIST_WINDOW_DAYS))
                            & (panel["date"] <= cut)]
        pred_rec = recursive_predict(models, val, hist_window, feats).values

        # --- hybride : α calibré sur le pli (comme Sales_Forecasting) ---
        best_alpha, best_w = 0.5, np.inf
        for a in alph_grid:
            wp = wape(y_true, a * pred_rec + (1 - a) * pred_dir)
            if np.isfinite(wp) and wp < best_w:
                best_w, best_alpha = wp, a
        pred_hyb = best_alpha * pred_rec + (1 - best_alpha) * pred_dir
        alphas_folds.append(best_alpha)

        # --- baseline naïve saisonnière lag-7 ---
        hist_real = panel.loc[panel["date"] <= cut, KEYS + ["date", "quantity"]]
        pred_naive = seasonal_naive(hist_real, val, cutoff=cut, m=config.SEASONAL_NAIVE_LAG)

        res = val[KEYS + ["date", "commodity_group"]].copy()
        res["y_true"] = y_true
        res["pred_dir"], res["pred_rec"] = pred_dir, pred_rec
        res["pred_hyb"], res["pred_naive"] = pred_hyb, pred_naive
        res["horizon"] = (res["date"] - cut).dt.days
        res["fold"] = fold_no
        all_results.append(res)

        met = pd.DataFrame([_metrics_row("directe", y_true, pred_dir),
                            _metrics_row("recursive", y_true, pred_rec),
                            _metrics_row("hybride", y_true, pred_hyb),
                            _metrics_row("naive_saiso", y_true, pred_naive)]).assign(fold=fold_no)
        fold_metrics.append(met)
        print(met.to_string(index=False), f"\n  alpha pli : {best_alpha}")

    results = pd.concat(all_results, ignore_index=True)
    best_alpha = float(np.median(alphas_folds))

    summary = pd.DataFrame([
        _metrics_row("directe", results.y_true, results.pred_dir),
        _metrics_row("recursive", results.y_true, results.pred_rec),
        _metrics_row("hybride", results.y_true, results.pred_hyb),
        _metrics_row("naive_saiso", results.y_true, results.pred_naive),
    ])

    def _seg(col):
        cols = ["y_true", "pred_hyb", "pred_naive"]
        return (results.groupby(col, observed=True)[cols]
                .apply(lambda d: pd.Series({
                    "WAPE_hybride": wape(d.y_true, d.pred_hyb),
                    "WAPE_naive": wape(d.y_true, d.pred_naive),
                    "biais_pct_hybride": bias_pct(d.y_true, d.pred_hyb),
                    "volume_reel": d.y_true.sum()}))
                .reset_index())

    by_store, by_cat = _seg("store_id"), _seg("commodity_group")
    by_sku, by_horizon = _seg("sku_id"), _seg("horizon")

    w_h = summary.loc[summary.modele == "hybride", "WAPE"].iloc[0]
    w_n = summary.loc[summary.modele == "naive_saiso", "WAPE"].iloc[0]
    gain = (1 - w_h / w_n) * 100

    out = out_dir or (config.RESULTS_DIR / "backtest" / f"scenario{scenario}")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / "results.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(fold_metrics, ignore_index=True).to_csv(out / "fold_metrics.csv", index=False)
    by_store.to_csv(out / "by_store.csv", index=False)
    by_cat.to_csv(out / "by_category.csv", index=False)
    by_sku.to_csv(out / "by_sku.csv", index=False)
    by_horizon.to_csv(out / "by_horizon.csv", index=False)
    (out / "alpha.txt").write_text(str(best_alpha))  # lu par src/forecast.py

    print("\n--- SYNTHÈSE (tous plis) ---")
    print(summary.to_string(index=False))
    print(f"\nGain hybride vs naïve saisonnière : {gain:.1f}% "
          f"(WAPE {w_h:.4f} vs {w_n:.4f}) | alpha médian : {best_alpha}")
    print(f"Résultats -> {out} | durée {time.time() - t0:.0f}s")
    return dict(summary=summary, results=results, best_alpha=best_alpha, gain_vs_naive=gain)


def add_dynamic_cols():
    from src.features_dynamic import DYNAMIC_FEATURES
    return DYNAMIC_FEATURES


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    ap.add_argument("--mode", default=None, choices=[None, "fast", "ensemble"])
    ap.add_argument("--folds", type=int, default=config.BACKTEST_N_FOLDS)
    ap.add_argument("--horizon", type=int, default=config.BACKTEST_HORIZON_DAYS)
    a = ap.parse_args()
    run_backtest(scenario=a.scenario, mode=a.mode, n_folds=a.folds, horizon=a.horizon)
    sys.exit(0)
