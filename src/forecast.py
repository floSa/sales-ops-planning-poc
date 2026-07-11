"""
Prévision opérationnelle (Bloc A) : entraînement sur tout l'historique puis
prévision magasin × SKU × jour du 2026-07-01 au 2026-12-31 (atterrissage fin
d'année), par scénario.

Stratégie : récursive pure (quasi non biaisée au backtest, contrairement à
l'hybride WAPE-optimale dont la branche directe biaise bas — cf. commentaire
dans run_forecast). Pour un consommateur budget/finance, le biais prime sur
quelques points de WAPE (wiki §0.4).

Le CA prévisionnel par ligne = quantité prévue × prix théorique futur
(base_price × CPI projeté × (1 − remise des promos planifiées)) — le
calendrier promo 2026 H2 est déjà dans data/referentiel/promo_calendar.csv.

Usage :
    python -m src.forecast --scenario 1 [--mode fast|ensemble]
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src import config
from src.dataset import KEYS, build_panel
from src.features_dynamic import DYNAMIC_FEATURES, add_dynamic_features
from src.inferencing import recursive_predict
from src.modeling import train_models


def run_forecast(scenario: int = 1, mode: str | None = None) -> pd.DataFrame:
    t0 = time.time()
    include_weather = scenario == 2
    print(f"=== Prévision scénario {scenario} : {config.HIST_END} -> {config.FORECAST_END} ===")

    panel, features, _ = build_panel(config.HIST_START, config.FORECAST_END,
                                     include_weather=include_weather)
    feats = features + DYNAMIC_FEATURES
    hist_end = pd.Timestamp(config.HIST_END)
    hist = panel[panel["date"] <= hist_end]
    future = panel[panel["date"] > hist_end]
    assert not hist.quantity.isna().any() and future.quantity.isna().all()

    print(f"Entraînement sur {len(hist):,} lignes…")
    hist_dyn = add_dynamic_features(hist)
    models = train_models(hist_dyn[feats], hist_dyn["quantity"],
                          weights=hist_dyn["sample_weight"], mode=mode)

    # Prévision opérationnelle = RÉCURSIVE pure. Le backtest (summary.csv)
    # montre : hybride WAPE-optimale = biais -7,3 % (la branche directe biaise
    # bas, et de plus en plus avec l'horizon : -2 % en semaine 1 -> -15 % en
    # semaine 4) ; récursive = WAPE quasi égal (+1,9 % relatif) mais biais
    # +2,7 %. Pour un usage budget/finance c'est le biais qui prime (wiki
    # §0.4) — l'hybride reste l'affichage du backtest court terme.
    print(f"Prévision récursive ({future['date'].nunique()} jours)…")
    hist_window = hist[hist["date"] > hist_end - pd.Timedelta(days=config.HIST_WINDOW_DAYS)]
    pred = recursive_predict(models, future, hist_window, feats).values

    out = future[KEYS + ["date", "commodity_group", "unit_price", "is_open"]].copy()
    out["quantity_pred"] = pred
    out["ca_pred"] = pred * out["unit_price"]
    out["scenario"] = scenario

    out_dir = config.RESULTS_DIR / "forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"forecast_daily_scenario{scenario}.parquet", index=False)
    print(f"-> {out_dir / f'forecast_daily_scenario{scenario}.parquet'} "
          f"({len(out):,} lignes, {time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    ap.add_argument("--mode", default=None, choices=[None, "fast", "ensemble"])
    a = ap.parse_args()
    run_forecast(scenario=a.scenario, mode=a.mode)
    sys.exit(0)
