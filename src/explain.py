"""
Explicabilité du moteur de prévision (Bloc A) — « qu'est-ce qui pilote la
prévision ? ».

Entraîne le modèle sur tout l'historique (scénario 2, météo incluse, pour voir
le poids réel des signaux météo) et extrait l'importance des variables au sens
du GAIN LightGBM (réduction d'erreur totale apportée par chaque variable).

Les noms techniques des variables sont traduits en libellés lisibles et
regroupés en familles métier, pour une lecture directe par un CDG :
  - Historique des ventes (lags, moyennes glissantes)
  - Calendrier & saisonnalité
  - Prix & promotions
  - Météo
  - Identité produit & magasin

Sortie : results/explain/feature_importance.csv (+ by_family.csv) et affichage.
Ne sert PAS à la prévision : c'est un diagnostic, lancé à part.

Usage :
    python -m src.explain [--scenario 2]
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src import config
from src.dataset import build_panel
from src.features_dynamic import DYNAMIC_FEATURES, add_dynamic_features
from src.modeling import train_models

# Libellés lisibles (français) par variable
LABELS = {
    "store_code": "Identité du magasin", "sku_code": "Identité du produit",
    "commodity_code": "Famille de produit", "brand_type_code": "Type de marque",
    "region_code": "Région", "store_type_code": "Taille de magasin",
    "is_online": "Canal Online", "surface_m2": "Surface du magasin",
    "is_eb": "Exclusivité enseigne (EB)", "is_pl": "Marque distributeur (PL)",
    "base_price": "Prix catalogue",
    "day_of_week": "Jour de la semaine", "day_of_month": "Jour du mois",
    "month": "Mois", "week_of_year": "Semaine de l'année", "is_weekend": "Week-end",
    "dow_sin": "Cycle hebdomadaire", "dow_cos": "Cycle hebdomadaire",
    "doy_sin": "Cycle annuel", "doy_cos": "Cycle annuel",
    "is_holiday": "Jour férié", "is_day_before_holiday": "Veille de férié",
    "is_pont": "Pont", "days_until_holiday": "Jours avant un férié",
    "days_since_holiday": "Jours depuis un férié",
    "is_open": "Magasin ouvert", "open_hours": "Heures d'ouverture",
    "days_since_start": "Ancienneté dans l'historique",
    "rel_price": "Prix relatif (vs habituel)", "promo_discount": "Profondeur de remise",
    "is_promo_produits": "Promo produits", "is_promo_mea": "Mise en avant",
    "is_promo_influence": "Promo influence", "influence_day": "Jour de la promo influence",
    "is_promo_seuils": "Promo à seuil", "is_promo_cadeau": "Cadeau à seuil",
    "is_promo_ouverture": "Promo ouverture magasin", "is_post_promo": "Contrecoup post-promo",
    "temp_anomaly": "Anomalie de température", "rain_mm": "Pluie",
    "lag_1": "Ventes de la veille", "lag_7": "Ventes il y a 1 semaine",
    "lag_14": "Ventes il y a 2 semaines", "lag_28": "Ventes il y a 4 semaines",
    "lag_364": "Ventes l'an dernier (même jour)",
    "roll_mean_7": "Niveau moyen 7 j", "roll_mean_28": "Niveau moyen 28 j",
    "roll_std_7": "Variabilité 7 j", "roll_std_28": "Variabilité 28 j",
    "ewma_7": "Tendance récente lissée",
}

# Famille métier par variable
FAMILY = {}
for f in ["lag_1", "lag_7", "lag_14", "lag_28", "lag_364",
          "roll_mean_7", "roll_mean_28", "roll_std_7", "roll_std_28", "ewma_7"]:
    FAMILY[f] = "Historique des ventes"
for f in ["day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
          "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday",
          "is_day_before_holiday", "is_pont", "days_until_holiday",
          "days_since_holiday", "is_open", "open_hours"]:
    FAMILY[f] = "Calendrier & saisonnalité"
for f in ["base_price", "rel_price", "promo_discount", "is_promo_produits",
          "is_promo_mea", "is_promo_influence", "influence_day", "is_promo_seuils",
          "is_promo_cadeau", "is_promo_ouverture", "is_post_promo"]:
    FAMILY[f] = "Prix & promotions"
for f in ["temp_anomaly", "rain_mm"]:
    FAMILY[f] = "Météo"
for f in ["store_code", "sku_code", "commodity_code", "brand_type_code",
          "region_code", "store_type_code", "is_online", "surface_m2",
          "is_eb", "is_pl", "days_since_start"]:
    FAMILY[f] = "Identité produit & magasin"


def run_explain(scenario: int = 2, mode: str | None = None) -> pd.DataFrame:
    t0 = time.time()
    include_weather = scenario == 2
    print(f"=== Explicabilité (scénario {scenario}, "
          f"{'avec' if include_weather else 'sans'} météo) ===")

    panel, features, _ = build_panel(config.HIST_START, config.HIST_END,
                                     include_weather=include_weather)
    feats = features + DYNAMIC_FEATURES
    print(f"Entraînement sur {len(panel):,} lignes…")
    panel_dyn = add_dynamic_features(panel)
    models = train_models(panel_dyn[feats], panel_dyn["quantity"],
                          weights=panel_dyn["sample_weight"], mode="fast")

    lgbm = models["lgbm"]
    imp = pd.DataFrame({
        "feature": feats,
        "gain": lgbm.booster_.feature_importance(importance_type="gain"),
    })
    imp["importance_pct"] = (imp.gain / imp.gain.sum() * 100).round(2)
    imp["libelle"] = imp.feature.map(LABELS).fillna(imp.feature)
    imp["famille"] = imp.feature.map(FAMILY).fillna("Autre")
    imp = imp.sort_values("importance_pct", ascending=False).reset_index(drop=True)

    by_family = (imp.groupby("famille")["importance_pct"].sum()
                 .sort_values(ascending=False).round(2).reset_index())

    out = config.RESULTS_DIR / "explain"
    out.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out / "feature_importance.csv", index=False)
    by_family.to_csv(out / "by_family.csv", index=False)

    print("\n--- Top 12 variables (part du gain, %) ---")
    print(imp.head(12)[["libelle", "famille", "importance_pct"]].to_string(index=False))
    print("\n--- Par famille métier ---")
    print(by_family.to_string(index=False))
    print(f"\n-> {out} | durée {time.time() - t0:.0f}s")
    return imp


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=int, default=2, choices=[1, 2])
    a = ap.parse_args()
    run_explain(scenario=a.scenario)
    sys.exit(0)
