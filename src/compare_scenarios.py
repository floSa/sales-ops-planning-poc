"""
Comparaison scénario 1 (baseline) vs scénario 2 (baseline + IA météo) sur les
backtests — l'argumentaire chiffré du « Scénario 2 » du brief (1.1).

Deux lentilles :
  - grain natif SKU × magasin × jour : l'apport météo est marginal au global
    (le bruit des séries fines domine) mais réel sur les jours d'anomalie ;
  - grain de PILOTAGE magasin × jour (celui du CDG) : l'agrégation annule le
    bruit SKU et l'apport devient net, surtout sur les épisodes météo.

Sortie : results/backtest/comparaison_scenarios.csv + affichage console.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.modeling import wape


def compare() -> pd.DataFrame:
    b1 = pd.read_parquet(config.RESULTS_DIR / "backtest" / "scenario1" / "results.parquet")
    b2 = pd.read_parquet(config.RESULTS_DIR / "backtest" / "scenario2" / "results.parquet")
    wx = pd.read_csv(config.DATA_REF / "weather.csv", parse_dates=["date"])
    for b in (b1, b2):
        b["date"] = pd.to_datetime(b["date"])

    key = ["store_id", "sku_id", "date"]
    sku = b1[key + ["y_true", "pred_hyb"]].merge(
        b2[key + ["y_true", "pred_hyb"]], on=key, suffixes=("_s1", "_s2"))
    sku = sku.merge(wx[["store_id", "date", "temp_anomaly", "rain_mm"]],
                    on=["store_id", "date"])

    def agg_store(b):
        return (b.groupby(["store_id", "date"], observed=True)[["y_true", "pred_hyb"]]
                .sum().reset_index())
    st = agg_store(b1).merge(agg_store(b2), on=["store_id", "date"],
                             suffixes=("_s1", "_s2"))
    st = st.merge(wx[["store_id", "date", "temp_anomaly", "rain_mm"]],
                  on=["store_id", "date"])

    segments = [("tous jours", lambda d: np.ones(len(d), bool)),
                ("anomalie température > 3°C", lambda d: d.temp_anomaly.abs() > 3),
                ("anomalie température > 5°C", lambda d: d.temp_anomaly.abs() > 5),
                ("pluie > 6 mm", lambda d: d.rain_mm > 6)]
    rows = []
    for grain, df in [("SKU × magasin × jour", sku), ("magasin × jour (pilotage)", st)]:
        for seg, fn in segments:
            d = df[fn(df)]
            w1 = wape(d.y_true_s1, d.pred_hyb_s1)
            w2 = wape(d.y_true_s2, d.pred_hyb_s2)
            rows.append(dict(grain=grain, segment=seg, n=len(d),
                             WAPE_scenario1=round(w1, 4), WAPE_scenario2=round(w2, 4),
                             gain_scenario2_pct=round((w1 - w2) / w1 * 100, 1)))
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "backtest" / "comparaison_scenarios.csv", index=False)
    return out


if __name__ == "__main__":
    print(compare().to_string(index=False))
