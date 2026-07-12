# MaxiZoo — POC Prévision des ventes

POC **développé et fonctionnel** (V1 livrée le 2026-07-11). Voir [README.md](README.md)
pour l'architecture, le mode d'emploi, les résultats et les hypothèses 🟡.
Le cadrage d'origine est dans `00_Preparatif/` (spec : [03_Cadrage_Points_Ouverts.md](00_Preparatif/03_Cadrage_Points_Ouverts.md)).

## L'essentiel

- **Données 100 % synthétiques** (5 ans, 12 magasins villes FR + ONLINE, 60 SKU),
  générées par `src/generate_data.py` — seule la météo est réelle (Open-Meteo,
  figée dans `data/referentiel/weather.csv`). Doc : [data/README.md](data/README.md).
- **Pipeline bout-en-bout** : `python main.py` (génération → backtests 2 scénarios →
  prévisions S2 2026 → écarts → rolling forecast → comparaison scénarios, ~1 h).
- **Dashboard** : `streamlit run dashboard_simulation.py` (mono-profil CDG Ventes).
- **Résultats clés** : WAPE 0,69 vs 0,94 naïve lag-7 (+26 %, bat la baseline sur
  chaque magasin et catégorie) ; WAPE 0,16 à la maille pilotage magasin × jour ;
  apport scénario 2 (météo) : +5,1 % au global pilotage, +15,9 % les jours de pluie ;
  atterrissage 2026 ≈ 10,4 M€ (synthétique).

## Environnement (important)

Python : venv WSL `~/Projets/MaxiZoo/.venv` (pas de sudo — libgomp embarqué via
`.venv/lib/native/` + `.pth` de préchargement, cf. mémoire projet). Exécution :
`wsl.exe -d Ubuntu-24.04 -- bash -lc "cd ~/Projets/MaxiZoo && .venv/bin/python …"`.
Machine partagée avec d'autres services : ne pas paralléliser les entraînements.

## Conventions

- Métrique : WAPE + biais, jamais le MAPE. Baseline : naïve saisonnière lag-7.
- Grain natif unique : magasin × SKU × jour ; le reste = agrégations de reporting.
- Hypothèses non validées client marquées 🟡 (code, README, dashboard).
- Commits réguliers ; push prévu sur le GitHub perso de Florian en fin de projet.

## Lien avec le reste du dossier `Retail/` (OneDrive)

Socle repris de `Retail/Sales_Forecasting` (pipeline GBM Tweedie) et
`Pilotage_StoreItem` (dashboard, drift). Ce POC calcule un **besoin en ETP**
(effectif), pas une masse salariale — pont éventuel vers
`POC_Pilotage_CA_Stock_RH` : ETP × coût moyen (cadrage §10, non traité).
