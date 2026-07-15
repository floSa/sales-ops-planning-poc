# POC Prévision des ventes — MaxiZoo

Démonstrateur de faisabilité (POC) d'un moteur de prévision des ventes
magasin × SKU × jour avec simulation de scénarios, calendrier promotionnel,
cascade de calcul jusqu'au CA net, besoin en effectif (ETP) et analyse
d'écarts avec rolling forecast. Réalisé par AOSIS Consulting.

> ⚠️ **Toutes les données sont synthétiques** (générées par
> `src/generate_data.py`, cf. [data/README.md](data/README.md)). Aucune donnée
> réelle MaxiZoo n'est utilisée ; les chiffres n'ont aucune valeur métier.
> L'objectif est de démontrer la **mécanique de bout en bout**, pas de
> produire un chiffre exploitable tel quel.

## Ce que couvre le POC (cadrage V1 du 2026-07-11)

| Bloc du brief | Niveau | Contenu |
|---|---|---|
| **A — Moteur de prévision** | complet | GBM Tweedie + sample weights ruptures, grain natif magasin × SKU × jour, scénarios 1 (baseline calendaire+promo) et 2 (baseline + IA météo), cascade PV × PA → PM + inflation × transactions → CA net |
| **B — Dashboard de simulation** | dégradé « profil unique » | un seul profil (CDG Ventes), hypothèses mensuelles éditables, saisie de campagne promo avec uplift proposé — pas de moteur de droits DR/DV |
| **C — Module ETP** | dégradé « interne seul » | besoin ETP par magasin depuis CA/heure, fréquentation, horaires — pas de signal concurrentiel, ONLINE exclu |
| **D — Écarts + rolling forecast** | complet sauf données non structurées | décomposition prix/volume/mix/promo/calendaire, atterrissage fin d'année — Online agrégé (pas de sous-zones) |

Revue détaillée, point par point des consignes client d'origine : [00_Preparatif/05_Etat_Avancement_vs_Consignes.md](00_Preparatif/05_Etat_Avancement_vs_Consignes.md).
Feuille de route des prochaines étapes : [00_Preparatif/06_Prochaines_Etapes.md](00_Preparatif/06_Prochaines_Etapes.md).

## Démarrage rapide

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.generate_data      # ~20 s (Open-Meteo ou --synthetic-weather)
.venv/bin/python main.py --skip-data       # backtests + prévisions + écarts (~1-2 h)
.venv/bin/streamlit run dashboard_simulation.py
```

Chaque étape est relançable indépendamment : `python -m src.backtest --scenario 1`,
`python -m src.forecast --scenario 2`, `python -m src.ecarts --scenario 1`…

## Architecture

```
data/referentiel/   stores, horaires, produits, promos (6 typologies), inflation, météo
data/transactions/  sales_daily (SKU×magasin×jour), traffic_daily, sales_hourly
src/
  generate_data.py  générateur synthétique + 13 contrôles de cohérence
  weather.py        météo Open-Meteo figée / repli synthétique / climatologie future
  dataset.py        panel + features statiques (calendaire FR, promo, prix, magasin, météo)
  features_dynamic.py  lags / rollings / EWMA par série (anti-fuite, shift(1))
  baselines.py      naïve saisonnière hebdo (lag-7) — LA référence à battre
  modeling.py       LightGBM Tweedie (défaut) ou ensemble XGB+LGBM+CatBoost
  inferencing.py    prévision directe + récursive (hybride α calibré au backtest)
  backtest.py       rolling-origin 4 plis × 28 j, WAPE + biais par magasin/catégorie/SKU/horizon
  forecast.py       prévision opérationnelle juil→déc 2026
  cascade_ca.py     PV × PA → PM + inflation × transactions → CA net (hypothèses mensuelles)
  etp.py            besoin ETP interne par magasin (normes paramétrables 🟡)
  ecarts.py         décomposition écart réel/prévu : prix, volume, mix + reventilation promo/calendaire
  rolling_forecast.py  atterrissage fin d'année + suivi mensuel de crédibilité
  simulation.py     uplift promo proposé depuis l'historique, application au scénario
  prediction_intervals.py  fourchettes P10/P90 calibrées sur le backtest (proba)
  explain.py        explicabilité : importance des variables (gain LightGBM) par famille métier
  compare_scenarios.py  comparaison scénario 1 vs 2 (apport de la météo)
tests/test_core.py  invariants (cascade, décomposition d'écarts, naïve, cohérence horaire)
dashboard_simulation.py  Streamlit mono-profil CDG Ventes
.streamlit/config.toml   thème clair forcé (lisibilité)
results/            backtests, prévisions, écarts, explicabilité, rapports
```

Tests : `python -m tests.test_core` (rapides, sans entraînement — vérifient les
invariants mathématiques : cascade neutre == modèle, identité de la décomposition
d'écarts, naïve lag-7, cohérence horaire/journalier).

Le pipeline reprend l'architecture de `Retail/Sales_Forecasting` (même maille,
même stratégie de pondération des ruptures, même logique directe/récursive/hybride),
étendue en rolling-origin multi-plis comme préconisé par le wiki (§0.5, §7.1).

## Méthodologie d'évaluation (cadrage §9 — non négociable)

- **Métrique : WAPE + biais** — jamais le MAPE (explose sur les ventes à zéro ;
  ~45 % des lignes sont à zéro à cette maille).
- **Backtest rolling-origin** : 4 plis glissants de 28 jours au grain jour,
  réentraînement complet à chaque pli, déclinaison par magasin / catégorie /
  SKU / horizon.
- **Baseline à battre : naïve saisonnière hebdomadaire (lag-7)** — un modèle
  qui ne bat pas la naïve ne se justifie pas.
- Grain natif unique magasin × SKU × jour : semaine/mois, WTD/MTD/YTD et
  catégorie sont des **agrégations de reporting**, pas des modèles séparés.

## Résultats du backtest (rolling-origin 4 plis × 28 j, mars→juin 2026)

Au **grain natif SKU × magasin × jour** (le plus dur : ~46 % de zéros) :

| Scénario | WAPE hybride | WAPE naïve lag-7 | Gain | Biais CA (récursive) |
|---|---|---|---|---|
| 1 — Baseline (calendaire + promos) | **0,694** | 0,937 | **+25,9 %** | +2,7 % |
| 2 — Baseline + IA (météo) | **0,693** | 0,937 | **+26,1 %** | +2,5 % |

Le modèle bat la naïve sur **chacun des 13 magasins et des 8 catégories**
(détail : `results/backtest/scenario*/by_store.csv`, `by_category.csv`).

Au **grain de pilotage magasin × jour** (celui du CDG — agrégation des
prévisions SKU), le WAPE tombe à **0,16-0,17** et l'apport du scénario 2
devient net (`results/backtest/comparaison_scenarios.csv`) :

| Segment (magasin × jour) | WAPE scén. 1 | WAPE scén. 2 | Apport IA météo |
|---|---|---|---|
| Tous jours | 0,170 | 0,162 | **+5,1 %** |
| Anomalie température > 5 °C | 0,206 | 0,192 | **+6,9 %** |
| Pluie > 6 mm | 0,200 | 0,168 | **+15,9 %** |

Lecture : au grain SKU l'apport météo est marginal (+0,2 % global, +3 % sur
les épisodes) car le bruit des séries fines domine ; c'est à la maille de
pilotage qu'il se matérialise. ⚠️ L'ampleur dépend de la calibration du
générateur (cf. data/README.md) — à re-mesurer sur données réelles.

**Prévision opérationnelle** (récursive, biais +2,7 %) : atterrissage 2026 =
réel janv-juin **5,09 M€** + prévu juil-déc **5,33 M€** = **10,42 M€**
(scénario 1), profil mensuel aligné sur N-1 (+1,7 %, pic de Noël préservé).
L'atterrissage scénario 2 ressort à **10,89 M€** : au-delà du réalisé, la
météo future est la **climatologie** (anomalie nulle), donc une année « sans
surprise météo » — structurellement optimiste puisque les effets météo sont
surtout pénalisants. L'écart entre les deux atterrissages (~0,5 M€) chiffre
l'enjeu météo ; en V2, brancher des prévisions météo réelles à 15 j puis des
scénarios stochastiques au-delà.

## Ce qui pilote la prévision (explicabilité)

`python -m src.explain` extrait l'importance des variables (gain LightGBM),
regroupée par famille métier (`results/explain/`) :

| Famille | Part du gain |
|---|---|
| Historique des ventes (lags, moyennes glissantes) | **80,0 %** |
| Calendrier & saisonnalité | 16,0 % |
| Identité produit & magasin | 2,0 % |
| Prix & promotions | 1,6 % |
| Météo | 0,4 % |

Variable la plus influente : le **niveau moyen des 28 derniers jours**
(74 % à elle seule), puis le statut d'ouverture et les horaires du magasin.
Profil classique d'un modèle de demande retail : l'auto-régressif domine, le
calendrier structure, promo et météo affinent. La météo pèse peu **au global**
— cohérent avec le §Résultats : son apport se concentre sur les jours
d'anomalie et à la maille de pilotage, pas sur la moyenne SKU × jour.

## Hypothèses de travail 🟡 (à valider avec le client)

Reprises du cadrage (`00_Preparatif/03_Cadrage_Points_Ouverts.md` §8) et
complétées en développement :

1. **EB** = « exclusivité enseigne », **PL** = marque distributeur (flags binaires).
2. **PA** (Panier Article) = nombre moyen d'articles par ticket.
3. **Inflation** : paramètre central mensuel ; dans la cascade, le champ
   « Δ inflation » est un **écart** à la trajectoire CPI déjà incluse dans les
   prix prévus (défaut 0) — sinon double compte.
4. **Transactions prévues = volume prévu / PA projeté**, PA projeté = profil
   historique lissé (8 semaines, par magasin × jour de semaine) — pas de
   second modèle dédié en V1.
5. **Normes ETP synthétiques** : 220 € de CA/heure-vendeur, 12 tickets/heure,
   effectif plancher 2, 151,67 h/mois — paramétrables dans le dashboard.
6. **Uplift promo proposé** = uplift médian par SKU observé dans l'historique
   par typologie (approximation POC : baseline = ventes hors promo du même SKU,
   pas un contrefactuel modélisé — cf. wiki §9.1 pour la cible v2).
7. **Météo réelle Open-Meteo** dans le jeu synthétique (validé le 2026-07-11 :
   amendement du cadrage « tout signal synthétique » — la demande reste
   simulée, la météo est réelle et figée dans data/, aucune dépendance externe
   du pipeline). Au-delà du réalisé : climatologie jour-de-l'année.

## Adaptations vs plan initial (annoncées et validées)

- **Mode `fast` par défaut** (LightGBM seul, 300 arbres) : passage de 9 séries
  (Sales_Forecasting) à ~1 560 séries × 1,4 M lignes — l'ensemble 3-GBM reste
  disponible via `--mode ensemble` ou `config.MODEL_MODE`.
- **Cible en échelle réelle** (pas de log1p comme dans Sales_Forecasting) :
  la loss Tweedie gère nativement zéros et asymétrie, et on évite le biais de
  ré-exponentiation.

## Limites connues (POC, pas produit)

- Les prix réels == prix prévus dans le synthétique → l'effet **prix** de la
  décomposition d'écarts est ~0 en réel vs prévision (il devient non trivial
  en scénario simulé vs baseline via le coefficient PV).
- Les arbres n'extrapolent pas la tendance au-delà de l'historique : la
  croissance Online au-delà du réalisé est portée par les lags récursifs, pas
  par un terme de tendance explicite.
- La prévision du scénario 2 au-delà du réalisé repose sur la climatologie
  (anomalie météo nulle) : c'est un scénario « météo normale », borne haute
  quand les effets météo sont pénalisants (cf. Résultats).
- Pas de réconciliation hiérarchique (un seul niveau de modélisation,
  agrégations simples — MinT à considérer si plusieurs niveaux deviennent
  consommés simultanément, wiki §6).
- **Prévision probabiliste (fourchettes P10/P90)** : version POC livrée, mais
  **calibrée empiriquement sur le backtest** (pas des quantiles appris par le
  modèle — cf. `src/prediction_intervals.py`). Couverture ~80 % au grain
  magasin×jour ; au-delà de l'horizon de backtest (28 j) la fourchette est
  élargie en √(horizon), extrapolation à revalider. Des quantiles natifs
  (LightGBM `objective="quantile"`) restent la cible pour un usage réassort fin.
- Données non structurées (météo pour l'explication d'écarts, tendances,
  concurrence) : hors scope V1 (cadrage §6).

## Lien avec l'existant `Retail/`

Voir [00_Preparatif/02_Existant_Reutilisable.md](00_Preparatif/02_Existant_Reutilisable.md).
Ce POC calcule un **besoin en ETP** (effectif), pas une masse salariale (euros) —
le pont vers `POC_Pilotage_CA_Stock_RH` est « ETP × coût moyen = masse salariale »
(cadrage §10), non traité ici.
