"""
Rolling forecast mensuel (Bloc D) : atterrissage fin d'année.

    Atterrissage 2026 = CA réel YTD (janv -> juin, dernier réalisé)
                      + CA prévisionnel (juil -> déc, sortie du Bloc A)

Décliné par magasin et par mois. Le suivi de crédibilité de la prévision
(réel vs prévu mois par mois sur la fenêtre de backtest, dans l'esprit de
Pilotage_StoreItem/src/drift_monitor.py) est produit en complément : si le
WAPE mensuel dérive, l'atterrissage est à recalibrer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.modeling import bias_pct, wape


def atterrissage(scenario: int = 1) -> pd.DataFrame:
    """CA mensuel par magasin : réel (janv-juin 2026) + prévision (juil-déc)."""
    sales = pd.read_parquet(config.DATA_TRX / "sales_daily.parquet")
    sales["date"] = pd.to_datetime(sales["date"])
    year = pd.Timestamp(config.HIST_END).year

    reel = sales[sales.date.dt.year == year].copy()
    reel["month"] = reel.date.dt.to_period("M").astype(str)
    reel_m = (reel.groupby(["store_id", "month"], observed=True)["revenue"]
              .sum().rename("ca").reset_index())
    reel_m["type"] = "réel"

    fc = pd.read_parquet(config.RESULTS_DIR / "forecast"
                         / f"forecast_daily_scenario{scenario}.parquet")
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc[fc.date.dt.year == year].copy()
    fc["month"] = fc.date.dt.to_period("M").astype(str)
    prev_m = (fc.groupby(["store_id", "month"], observed=True)["ca_pred"]
              .sum().rename("ca").reset_index())
    prev_m["type"] = "prévision"

    out = pd.concat([reel_m, prev_m], ignore_index=True).sort_values(
        ["store_id", "month"]).reset_index(drop=True)
    out_dir = config.RESULTS_DIR / "rolling_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"atterrissage_{year}_scenario{scenario}.csv", index=False)
    return out


def suivi_mensuel(scenario: int = 1) -> pd.DataFrame:
    """Crédibilité de la prévision : réel vs prévu (hybride) agrégé par mois
    sur la fenêtre de backtest, en CA (quantités valorisées au prix réel)."""
    bt = pd.read_parquet(config.RESULTS_DIR / "backtest" / f"scenario{scenario}"
                         / "results.parquet")
    sales = pd.read_parquet(config.DATA_TRX / "sales_daily.parquet")
    sales["date"] = pd.to_datetime(sales["date"])
    bt["date"] = pd.to_datetime(bt["date"])
    d = bt.merge(sales[["store_id", "sku_id", "date", "unit_price"]],
                 on=["store_id", "sku_id", "date"], how="left")
    d["month"] = d.date.dt.to_period("M").astype(str)
    d["ca_reel"] = d.y_true * d.unit_price
    d["ca_prev"] = d.pred_hyb * d.unit_price

    rows = []
    for m, g in d.groupby("month"):
        rows.append(dict(month=m,
                         ca_reel=g.ca_reel.sum(), ca_prev=g.ca_prev.sum(),
                         wape_qte=wape(g.y_true, g.pred_hyb),
                         biais_ca_pct=bias_pct(g.ca_reel, g.ca_prev),
                         n_jours=g.date.nunique()))
    out = pd.DataFrame(rows).sort_values("month")
    out_dir = config.RESULTS_DIR / "rolling_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"suivi_mensuel_scenario{scenario}.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    print(atterrissage(a.scenario).groupby("type")["ca"].sum())
    print(suivi_mensuel(a.scenario).to_string(index=False))
