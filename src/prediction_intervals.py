"""
Prévision probabiliste (feuille de route P3) — fourchettes P10 / P90.

Le moteur produit une prévision *ponctuelle* (une valeur par magasin × SKU ×
jour). Pour un usage réassort / stock de sécurité, on veut aussi une
**fourchette** : entre quelles bornes la vente réelle a-t-elle de bonnes chances
de tomber ? On la calibre **empiriquement sur le backtest**, sans réentraîner :
on connaît, sur les plis déjà joués, l'écart entre le prévu et le réel — on en
déduit de combien la réalité s'écarte typiquement de la prévision.

Méthode (assumée pour un POC, cf. README « Limites ») :
  1. Grain de calibration = magasin × jour (le grain de pilotage, WAPE ~0,16 —
     au grain SKU les ~46 % de zéros rendent la notion de fourchette peu
     parlante). On somme les SKU par magasin × jour.
  2. On mesure le ratio r = réel / prévu, et on prend ses quantiles empiriques
     P10 / P90 — d'où des facteurs multiplicatifs f10, f90 tels que
     [prévu × f10, prévu × f90] couvre ~80 % des cas.
  3. L'incertitude croît avec l'horizon : on calibre par semaine d'horizon
     (1→4) sur les 28 jours du backtest, puis au-delà on élargit la fourchette
     en √(horizon) (accumulation d'aléa de type marche aléatoire — extrapolation
     explicite, à revalider sur données réelles).
  4. Agrégation (mois, atterrissage) : les aléas journaliers se compensent
     largement (quasi-indépendants d'un jour à l'autre — vérifié : la fourchette
     journalière ~±23 % tombe à ~±5 % au cumul 28 j). On agrège donc les écarts
     **en quadrature** (racine de la somme des carrés), ce qui reproduit la
     fourchette cumulée observée.

Usage :
    python -m src.prediction_intervals --scenario 1
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config

# quantiles cibles de la fourchette
P_LOW, P_HIGH = 0.10, 0.90
HORIZON_WEEKS = 4                       # semaines calibrées (backtest = 28 j)
MIN_PRED = 1.0                          # magasin×jour ignoré si prévu <= 1 (ratio instable)
# Corrélation moyenne des aléas journaliers pour l'agrégation (cf. aggregate_band).
# Calibrée sur le backtest : ρ=0,05 amène la couverture de la fourchette agrégée
# au cumul 28 j × magasin à ~80-85 % (ρ=0 -> ~60 %, trop serré ; les erreurs ne
# sont pas parfaitement indépendantes, une composante de biais reste corrélée).
AGG_RHO = 0.05


def _week_bucket(horizon_days: pd.Series) -> pd.Series:
    """Semaine d'horizon 1..HORIZON_WEEKS (au-delà de 28 j, plafonnée)."""
    return (((horizon_days - 1) // 7) + 1).clip(lower=1, upper=HORIZON_WEEKS)


def calibrate(scenario: int) -> pd.DataFrame:
    """Calibre les facteurs f10/f90 par semaine d'horizon depuis le backtest.

    Retourne un DataFrame [wk, f10, f90, n, couverture] et l'écrit dans
    results/forecast/pi_factors_scenario{scenario}.csv."""
    bt = config.RESULTS_DIR / "backtest" / f"scenario{scenario}" / "results.parquet"
    if not bt.exists():
        raise FileNotFoundError(
            f"Backtest manquant ({bt}). Lancer d'abord : "
            f"python -m src.backtest --scenario {scenario}")
    r = pd.read_parquet(bt)

    # SKU -> magasin × jour (grain pilotage), sur la prévision hybride du backtest
    sd = (r.groupby(["fold", "store_id", "date", "horizon"], as_index=False)
          .agg(real=("y_true", "sum"), pred=("pred_hyb", "sum")))
    sd = sd[sd.pred > MIN_PRED].copy()
    sd["ratio"] = sd.real / sd.pred
    sd["wk"] = _week_bucket(sd.horizon)

    fac = (sd.groupby("wk").ratio
           .quantile([P_LOW, P_HIGH]).unstack()
           .rename(columns={P_LOW: "f10", P_HIGH: "f90"}))
    fac["n"] = sd.groupby("wk").size()

    # couverture réelle de l'intervalle par semaine (contrôle de calibration)
    chk = sd.merge(fac[["f10", "f90"]], left_on="wk", right_index=True)
    inside = (chk.ratio >= chk.f10) & (chk.ratio <= chk.f90)
    fac["couverture"] = chk.assign(inside=inside).groupby("wk").inside.mean()

    fac = fac.reset_index()
    out = config.RESULTS_DIR / "forecast" / f"pi_factors_scenario{scenario}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fac.to_csv(out, index=False)
    return fac


def load_factors(scenario: int) -> pd.DataFrame | None:
    p = config.RESULTS_DIR / "forecast" / f"pi_factors_scenario{scenario}.csv"
    return pd.read_csv(p) if p.exists() else None


def add_line_intervals(forecast_df: pd.DataFrame, factors: pd.DataFrame,
                       hist_end=None) -> pd.DataFrame:
    """Ajoute quantity_p10/p90 et ca_p10/p90 à la prévision quotidienne.

    Le facteur magasin×jour est appliqué uniformément aux SKU de ce jour (la
    calibration est au grain magasin×jour) : la somme des bornes par SKU
    redonne bien la borne au grain magasin×jour. Au-delà de 28 j, l'écart à 1
    est élargi en √(horizon/28) (extrapolation explicite)."""
    hist_end = pd.Timestamp(hist_end or config.HIST_END)
    df = forecast_df.copy()
    horizon = (pd.to_datetime(df["date"]) - hist_end).dt.days.clip(lower=1)
    wk = _week_bucket(horizon)
    fmap = factors.set_index("wk")
    f10 = wk.map(fmap.f10)
    f90 = wk.map(fmap.f90)

    # élargissement au-delà de la fenêtre calibrée (28 j) : ~√(horizon)
    grow = np.sqrt((horizon / (HORIZON_WEEKS * 7)).clip(lower=1.0))
    f10 = 1.0 - (1.0 - f10) * grow
    f90 = 1.0 + (f90 - 1.0) * grow

    df["quantity_p10"] = (df["quantity_pred"] * f10).clip(lower=0)
    df["quantity_p90"] = df["quantity_pred"] * f90
    df["ca_p10"] = (df["ca_pred"] * f10).clip(lower=0)
    df["ca_p90"] = df["ca_pred"] * f90
    return df


def store_day_bounds(df_lines: pd.DataFrame, factors: pd.DataFrame,
                     value="ca_net", date_col="date", hist_end=None) -> pd.DataFrame:
    """Réduit des lignes (…×SKU×jour) au grain magasin×jour et pose la fourchette.

    Utilisé côté dashboard sur la cascade CA (`ca_net`) : la fourchette relative
    du CA est celle du volume (CA ∝ volume à hypothèses données), donc on
    applique le même facteur magasin×jour. Retourne [store_id, date, month,
    val, lo, hi] avec lo/hi = bornes au grain magasin×jour."""
    hist_end = pd.Timestamp(hist_end or config.HIST_END)
    sd = (df_lines.groupby(["store_id", date_col], as_index=False)[value].sum()
          .rename(columns={value: "val"}))
    horizon = (pd.to_datetime(sd[date_col]) - hist_end).dt.days.clip(lower=1)
    wk = _week_bucket(horizon)
    fmap = factors.set_index("wk")
    f10 = wk.map(fmap.f10).to_numpy()
    f90 = wk.map(fmap.f90).to_numpy()
    grow = np.sqrt((horizon / (HORIZON_WEEKS * 7)).clip(lower=1.0)).to_numpy()
    f10 = 1.0 - (1.0 - f10) * grow
    f90 = 1.0 + (f90 - 1.0) * grow
    sd["lo"] = (sd["val"] * f10).clip(lower=0)
    sd["hi"] = sd["val"] * f90
    sd["month"] = pd.to_datetime(sd[date_col]).dt.to_period("M").astype(str)
    return sd


def aggregate_band(store_day: pd.DataFrame, value="val", low="lo",
                   high="hi", rho: float = AGG_RHO) -> tuple[float, float, float]:
    """Agrège une fourchette sur un ensemble de lignes magasin×jour.

    Les aléas journaliers se compensent en partie mais pas totalement (une
    composante de biais reste corrélée). On agrège l'écart-type avec une
    corrélation moyenne ρ : σ_agg = √[(1−ρ)·Σσ² + ρ·(Σσ)²] — ρ=0 redonne la
    quadrature (indépendance, trop serré), ρ=1 la somme linéaire (comonotone,
    trop large). ρ est calibré sur le backtest (AGG_RHO). Attend un DataFrame
    déjà au grain magasin×jour (cf. store_day_bounds).
    Retourne (point, borne_basse, borne_haute)."""
    point = float(store_day[value].sum())
    dev_lo = (store_day[value] - store_day[low]).clip(lower=0)
    dev_hi = (store_day[high] - store_day[value]).clip(lower=0)
    s_lo = np.sqrt((1 - rho) * (dev_lo ** 2).sum() + rho * dev_lo.sum() ** 2)
    s_hi = np.sqrt((1 - rho) * (dev_hi ** 2).sum() + rho * dev_hi.sum() ** 2)
    return point, max(point - float(s_lo), 0.0), point + float(s_hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    fac = calibrate(a.scenario)
    print(f"=== Facteurs de fourchette P{int(P_LOW*100)}/P{int(P_HIGH*100)} "
          f"— scénario {a.scenario} ===")
    print(fac.round(3).to_string(index=False))
    print(f"\nCouverture visée {int((P_HIGH - P_LOW) * 100)}% ; "
          f"moyenne obtenue {fac.couverture.mean():.1%} (grain magasin×jour).")
    print(f"-> results/forecast/pi_factors_scenario{a.scenario}.csv")
    sys.exit(0)
