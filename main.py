"""
Pipeline bout-en-bout du POC MaxiZoo — critère de succès n°1 du cadrage.

Enchaîne : génération des données synthétiques -> backtest rolling-origin
(scénarios 1 et 2) -> prévision opérationnelle S2 2026 -> analyse d'écarts ->
rolling forecast. Le dashboard se lance ensuite avec :

    streamlit run dashboard_simulation.py

Usage :
    python main.py                     # tout, scénarios 1 et 2
    python main.py --skip-data         # sans régénérer les données
    python main.py --scenario 1        # un seul scénario
    python main.py --mode ensemble     # ensemble 3-GBM (long)
"""
from __future__ import annotations

import argparse
import sys
import time


def run(skip_data: bool, scenarios: list[int], mode: str | None):
    t0 = time.time()

    if not skip_data:
        print("\n" + "=" * 70 + "\nÉTAPE 1 — Génération du jeu de données synthétique\n" + "=" * 70)
        from src.generate_data import main as gen
        gen()

    from src.backtest import run_backtest
    from src.forecast import run_forecast
    from src.ecarts import build_backtest_ecarts
    from src.rolling_forecast import atterrissage, suivi_mensuel

    for s in scenarios:
        print("\n" + "=" * 70 + f"\nÉTAPE 2 — Backtest rolling-origin, scénario {s}\n" + "=" * 70)
        run_backtest(scenario=s, mode=mode)
        print("\n" + "=" * 70 + f"\nÉTAPE 3 — Prévision S2, scénario {s}\n" + "=" * 70)
        run_forecast(scenario=s, mode=mode)
        print("\n" + "=" * 70 + f"\nÉTAPE 4 — Écarts + rolling forecast, scénario {s}\n" + "=" * 70)
        build_backtest_ecarts(scenario=s)
        atterrissage(scenario=s)
        suivi_mensuel(scenario=s)

    print(f"\nPipeline complet en {(time.time() - t0) / 60:.1f} min. "
          f"Dashboard : streamlit run dashboard_simulation.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--scenario", type=int, default=None, choices=[1, 2],
                    help="un seul scénario (défaut : les deux)")
    ap.add_argument("--mode", default=None, choices=[None, "fast", "ensemble"])
    a = ap.parse_args()
    run(a.skip_data, [a.scenario] if a.scenario else [1, 2], a.mode)
    sys.exit(0)
