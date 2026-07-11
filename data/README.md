# Données du POC — 100 % SYNTHÉTIQUES

⚠️ **Aucune donnée réelle MaxiZoo.** Tout ce dossier est généré par
`src/generate_data.py` (graine fixe : reproductible via
`python -m src.generate_data`). Les chiffres n'ont **aucune valeur métier** :
ils servent à démontrer la mécanique de bout en bout du POC.

**Seule exception, validée le 2026-07-11** : la météo (`referentiel/weather.csv`)
est de la vraie météo historique des 12 villes (API Open-Meteo), récupérée une
seule fois à la génération puis figée ici — le pipeline n'a aucune dépendance
externe. La demande synthétique est générée en fonction de cette météo (sur
l'**anomalie** de température, pas le niveau), pour que le scénario 2
« baseline + IA » ait un vrai signal à apprendre. Repli 100 % synthétique
disponible : `python -m src.generate_data --synthetic-weather`.

## Périmètre

- **12 magasins physiques** : villes françaises réelles, la taille de la ville
  dimensionne la taille du magasin (Paris 1 400 m² → Brive 450 m²)
  + **1 magasin virtuel `ONLINE`** (canal e-commerce agrégé, sans sous-zones — cadrage §6).
- **60 SKU** répartis sur 8 commodity groups, dont **4 lancés en cours
  d'historique** (démo cold start).
- **5 ans d'historique** : 2021-07-01 → 2026-06-30, grain natif magasin × SKU × jour.

## Tables

### `referentiel/` (CSV)

| Fichier | Grain | Colonnes clés |
|---|---|---|
| `stores.csv` | magasin | store_id, store_name, region, store_type, surface_m2, population, lat/lon, is_online |
| `store_hours.csv` | magasin × jour de semaine | open_hour, close_hour, is_closed — intrant du calcul ETP |
| `products.csv` | SKU | commodity_group, brand, brand_type, is_eb 🟡, is_pl, base_price, launch_date |
| `promo_calendar.csv` | campagne | promo_type (6 typologies du brief), mechanic, discount_rate, dates, perimeter (magasin/online/omnicanal), store_id (ouverture magasin) |
| `promo_scope.csv` | campagne × SKU | SKU ciblés (typologies produits / influence / mise en avant) |
| `inflation.csv` | mois | inflation_mm_pct, cpi_index (base 100 = 2021-07), is_projection — paramètre central 🟡 |
| `weather.csv` | magasin × jour | temp_mean_c, rain_mm, temp_anomaly, source (open-meteo / synthetique / climatologie) |

### `transactions/` (Parquet — régénérables, CSV exclus du git)

| Fichier | Grain | Colonnes |
|---|---|---|
| `sales_daily.parquet` | magasin × SKU × jour (~1,36 M lignes) | quantity, unit_price (promo déduite), revenue, is_rupture (ventes censurées), promo_id |
| `traffic_daily.parquet` | magasin × jour | nb_tickets (= commandes pour ONLINE) |
| `sales_hourly.parquet` | magasin × jour × heure ouvrée (~245 k lignes) | ca, nb_tickets — répartition horaire réelle, Σ heures = totaux jour (contrôlé) |

## Mécanismes encodés dans la demande (vérité terrain du générateur)

Le pipeline de prévision doit les retrouver — c'est le test du POC :

- Saisonnalité hebdo (pic samedi en magasin, dimanche/lundi en ligne) et
  annuelle par catégorie (antiparasitaires l'été, oiseaux l'hiver, jouets à Noël).
- Jours fériés France (fermetures 1er mai/Noël/jour de l'an, régime dimanche sinon),
  veilles de férié, course aux fêtes de décembre.
- Tendance par magasin (Online ≈ +14 %/an, Dijon en léger déclin).
- Inflation mensuelle 2021-2023 réaliste appliquée aux prix (cpi_index).
- 6 typologies promo avec signatures distinctes : produits (uplift ∝ remise +
  creux post-promo), seuils/cadeau (hausse du panier article), influence
  (effet décalé ~28 j, online), mise en avant, ouverture magasin (pic local).
- Météo : canicule → trafic magasin en baisse et online en hausse,
  antiparasitaires/reptiles dopés ; vague de froid → rayon oiseaux dopé.
  ⚠️ L'ampleur de ces effets est **calibrée pour être détectable** par le
  scénario 2 (jusqu'à -4,5 %/°C d'anomalie chaude sur le trafic magasin,
  plancher ×0,65) : avec des effets plus faibles, le signal est noyé par le
  bruit des séries SKU et les deux scénarios deviennent indiscernables —
  l'apport réel de la météo devra être mesuré sur données réelles MaxiZoo.
- Ruptures de stock (~1 % des lignes) : ventes censurées, flag `is_rupture`
  → sample weights dans le pipeline (demande ≠ ventes, wiki §0.1).
- Intermittence : ~45 % de lignes à zéro (longue traîne équipement).

## Contrôles de cohérence

`src/generate_data.py` exécute 13 contrôles à chaque génération (sommes
horaire/journalier, PA plausible, uplift promo détectable par SKU, cold start,
part de zéros/ruptures, croissance Online…) et écrit le rapport dans
`results/data_quality_report.txt`. La génération échoue si un contrôle est rouge.
