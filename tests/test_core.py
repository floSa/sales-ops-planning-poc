"""
Tests des invariants du POC — durcissent les briques de calcul métier.

Rapides (pas d'entraînement de modèle) : ils vérifient les propriétés
mathématiques garanties par construction. Lancer :

    python -m pytest tests/ -v      (ou : python -m tests.test_core)

Prérequis : le jeu de données doit avoir été généré (python -m src.generate_data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.baselines import seasonal_naive
from src.cascade_ca import cascade, cascade_monthly, default_hypotheses
from src.ecarts import decompose


# --------------------------------------------------------------------------- #
# Fixtures légères (données réelles du POC si dispo, sinon jouet)
# --------------------------------------------------------------------------- #
def _load_forecast_and_tables():
    from src.dataset import load_tables
    fc_path = config.RESULTS_DIR / "forecast" / "forecast_daily_scenario1.parquet"
    if not fc_path.exists():
        return None, None
    fc = pd.read_parquet(fc_path)
    fc["date"] = pd.to_datetime(fc["date"])
    return fc, load_tables()


# --------------------------------------------------------------------------- #
# 1. Cascade : hypothèses neutres => CA net == CA modèle (au centime)
# --------------------------------------------------------------------------- #
def test_cascade_neutre_egale_modele():
    fc, t = _load_forecast_and_tables()
    if fc is None:
        print("  [skip] prévision absente")
        return
    c = cascade(fc, t["sales"], t["traffic"])  # hypothèses neutres par défaut
    ecart_rel = abs(c.ca_net.sum() / c.ca_modele.sum() - 1)
    assert ecart_rel < 1e-6, f"cascade neutre != modèle : écart {ecart_rel:.2%}"
    print(f"  [ok] cascade neutre == modèle (écart {ecart_rel:.1e})")


def test_cascade_coef_pv_proportionnel():
    """PV × k sur un mois => CA net de ce mois × k (les autres leviers neutres)."""
    fc, t = _load_forecast_and_tables()
    if fc is None:
        print("  [skip] prévision absente")
        return
    c0 = cascade(fc, t["sales"], t["traffic"])
    mois = sorted(c0.month.unique())[1]
    hyp = default_hypotheses(sorted(c0.month.unique()))
    hyp.loc[hyp.month == mois, "coef_pv"] = 1.03
    c1 = cascade(fc, t["sales"], t["traffic"], hypotheses=hyp)
    ratio = c1[c1.month == mois].ca_net.sum() / c0[c0.month == mois].ca_net.sum()
    assert abs(ratio - 1.03) < 1e-6, f"PV×1.03 => ratio {ratio:.4f}, attendu 1.03"
    print(f"  [ok] PV × 1,03 => CA net × {ratio:.4f}")


# --------------------------------------------------------------------------- #
# 2. Décomposition d'écarts : identité exacte prix + volume + mix == écart
# --------------------------------------------------------------------------- #
def test_ecarts_identite_comptable():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "store_id": rng.choice(["S01", "S02"], n),
        "sku_id": rng.choice(["SKU001", "SKU002", "SKU003"], n),
        "date": pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        "q_reel": rng.poisson(5, n).astype(float),
        "q_prev": rng.poisson(5, n).astype(float),
        "p_reel": rng.uniform(5, 40, n),
        "is_promo": rng.integers(0, 2, n),
        "is_calendar": rng.integers(0, 2, n),
    })
    df["p_prev"] = df.p_reel * rng.uniform(0.9, 1.1, n)  # prix prévu != réel
    out = decompose(df)
    residu = (out.ecart_total - (out.effet_prix + out.effet_volume + out.effet_mix)).abs()
    assert residu.max() < 1e-6, f"identité prix+volume+mix rompue : {residu.max():.2e}"
    # la reventilation par driver recompose l'effet quantité
    qte = out.effet_volume + out.effet_mix
    driver = out.dont_promo + out.dont_calendaire + out.dont_autre
    assert (qte - driver).abs().max() < 1e-6, "reventilation par driver incohérente"
    print(f"  [ok] écart == prix+volume+mix (résidu {residu.max():.1e}) et drivers cohérents")


# --------------------------------------------------------------------------- #
# 3. Baseline naïve saisonnière : reproduit la valeur d'il y a 7 jours
# --------------------------------------------------------------------------- #
def test_seasonal_naive_lag7():
    dates = pd.date_range("2026-01-01", "2026-03-31", freq="D")
    hist = pd.DataFrame({
        "store_id": "S01", "sku_id": "SKU001", "date": dates,
        "quantity": np.arange(len(dates), dtype=float),
    })
    cutoff = pd.Timestamp("2026-03-15")
    ev = hist[hist.date > cutoff][["store_id", "sku_id", "date"]].copy()
    pred = seasonal_naive(hist[hist.date <= cutoff], ev, cutoff=cutoff, m=7)
    # 1er jour prévu (cutoff+1) : réfère à cutoff+1-7 => valeur connue
    ref = hist.set_index("date").quantity
    attendu = ref[ev.date.iloc[0] - pd.Timedelta(days=7)]
    assert abs(pred[0] - attendu) < 1e-9, f"naïve lag-7 : {pred[0]} != {attendu}"
    assert np.isfinite(pred).all(), "la naïve laisse des NaN"
    print(f"  [ok] naïve lag-7 : 1er point = {pred[0]:.0f} (= J-7 connu)")


# --------------------------------------------------------------------------- #
# 4. Cohérence horaire == journalier (données générées)
# --------------------------------------------------------------------------- #
def test_coherence_horaire_journalier():
    hp = config.DATA_TRX / "sales_hourly.parquet"
    sp = config.DATA_TRX / "sales_daily.parquet"
    if not hp.exists() or not sp.exists():
        print("  [skip] données absentes")
        return
    hourly = pd.read_parquet(hp)
    sales = pd.read_parquet(sp)
    ca_h = hourly.groupby(["store_id", "date"], observed=True)["ca"].sum()
    ca_d = sales.groupby(["store_id", "date"], observed=True)["revenue"].sum()
    diff = (ca_d - ca_h.reindex(ca_d.index).fillna(0)).abs()
    assert diff.max() < 1.0, f"Σ CA horaire != journalier : écart max {diff.max():.2f} €"
    print(f"  [ok] Σ CA horaire == journalier (écart max {diff.max():.3f} €)")


# --------------------------------------------------------------------------- #
def _main():
    tests = [test_cascade_neutre_egale_modele, test_cascade_coef_pv_proportionnel,
             test_ecarts_identite_comptable, test_seasonal_naive_lag7,
             test_coherence_horaire_journalier]
    print(f"=== {len(tests)} tests d'invariants ===")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__} : {e}")
            failed += 1
    print(f"\n{'TOUS VERTS' if not failed else str(failed) + ' EN ÉCHEC'}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(_main())
