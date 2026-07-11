# Existant réutilisable — audit du dossier `Retail/`

Inventaire de ce qui, dans les projets déjà réalisés (`OneDrive.../Documents/Projets/Retail/`), peut nourrir ce POC — technique et données — mappé sur les 4 blocs identifiés dans le [brief](01_Brief_Consignes.md). Rien n'est copié à ce stade : ce document sert à décider, à la prochaine conversation, quoi reprendre vs quoi construire ex nihilo.

## Vue d'ensemble des sous-projets existants

| Projet | Contenu | Pertinence |
|---|---|---|
| **Sales_Forecasting** | Pipeline complet de prévision (nettoyage, features calendaire/lags, ensemble XGBoost/LightGBM/CatBoost Tweedie, backtest out-of-time, WAPE, SHAP) | ⭐⭐⭐ Bloc A |
| **Wiki_Prevision_Ventes** | Playbook méthodo 11 phases (cadrage, données, features, modélisation, probabiliste, réconciliation hiérarchique, évaluation, MLOps, sujets retail, gouvernance) | ⭐⭐⭐ Cadrage + Bloc A/D |
| **Pilotage_StoreItem** | Dashboard Streamlit magasin/article + intégration météo + suivi de dérive (`drift_monitor.py`) | ⭐⭐ Bloc B/D |
| **PowerBI_Pilotage** | Modèle en étoile (dim_article, dim_magasin, dim_horizon + fact_ventes/forecast/backtest/metrics), thème AOSIS | ⭐⭐ Bloc B (reporting) |
| **Walmart_Weather** | Structure proche du dataset Kaggle Walmart Store Sales : `stores.csv` (taille/type magasin), `features.csv` (**MarkDown1-5** = promos, CPI, chômage, carburant, température) | ⭐⭐ Proxy données promo + taille magasin |
| **Bike_Weather** | Données publiques Bike Sharing (météo → demande) | ⭐ Méthodo lien météo/demande uniquement |
| **Deck_AvantVente_Retail** | Génération de decks PPTX AOSIS via script Python | ⭐ Pour livrable de cadrage/synthèse client |
| **Synthese_Meteo** | Dashboard météo transverse | ⭐ Brique météo réutilisable pour scénario 2 |

## Détail par bloc

### Bloc A — Moteur de prévision (stat + IA)
**Directement réutilisable :**
- L'architecture du pipeline `Sales_Forecasting/src/` (cleaning → features statiques/dynamiques → sélection → modélisation ensembliste → backtest → inférence) est un socle technique transposable à la maille magasin×catégorie×semaine recommandée par le wiki (§0.3) plutôt que SKU×jour.
- `Wiki_Prevision_Ventes/09_Sujets_Retail.md` couvre déjà **promo/uplift** (décomposition baseline + incrémental, code `promo_uplift()`), **élasticité-prix** (régression log-log), **cannibalisation/halo** — c'est exactement la mécanique attendue en 1.1/1.2 du brief.
- `06_Reconciliation_Hierarchique.md` du wiki adresse la cohérence entre mailles (magasin/région, produit/catégorie) nécessaire pour les vues WTD/MTD/YTD multi-niveaux du brief.

**Absent, à construire :**
- La **cascade de calcul** propre au client (PV × PA → PM → +inflation × transactions → CA net) est une logique métier spécifique, pas présente dans l'existant — c'est un moteur de règles à écrire par-dessus le moteur statistique.
- Aucun mécanisme de **scénario baseline vs baseline+IA avec alertes** packagé — à construire (le drift monitoring de Pilotage_StoreItem donne une base pour la détection d'anomalies/alertes).

### Bloc B — Outil de simulation/saisie (UI métier)
**Réutilisable comme socle :**
- `Pilotage_StoreItem/dashboard_pilotage.py` (Streamlit) donne un point de départ technique (stack, structure) pour une UI de pilotage, mais **sans** simulation multi-scénarios ni saisie d'hypothèses éditables.
- Le modèle en étoile de `PowerBI_Pilotage` (dim_magasin, dim_article, dim_horizon + facts) est une bonne base de modélisation de données pour stocker scénarios/hypothèses/promos versionnés.

**Absent, à construire :**
- Simulateur multi-scénarios avec override par périmètre (CDG propose, DR/DV amendent hors inflation) → gestion de droits/rôles, aucune brique existante.
- Moteur de calendrier promo avec les 6 typologies de mécaniques → net new.

### Bloc C — Module RH (ETP)
**Rien de réutilisable en l'état.** C'était déjà le point ouvert non résolu du chantier voisin `POC_Pilotage_CA_Stock_RH` (dans `Retail/`) : aucun jeu de données ni brique technique liant CA/fréquentation/horaires à un besoin ETP. Le vrai bloquant reste les données externes (horaires + fréquentation des magasins concurrents) : aucune source identifiée dans l'existant AOSIS, à trancher (scraping, panel, étude terrain).

### Bloc D — Analyse d'écarts + rolling forecast
**Partiellement réutilisable :**
- `Pilotage_StoreItem/src/drift_monitor.py` + `results/drift_wape.csv`/`drift_features.csv` : suivi d'écarts prévision/réel et de dérive de features — base pour le rolling forecast et la détection de signaux de réatterrissage.
- `Sales_Forecasting/src/backtest.py` : logique de backtest out-of-time transposable au rolling-origin nécessaire pour un vrai rolling forecast (le wiki §0.5 note que le backtest actuel est mono-pli, à étendre).

**Absent, à construire :**
- Décomposition d'écart **price/volume/mix/promo/calendar** : pas implémentée telle quelle dans l'existant (l'élasticité-prix du wiki §9.2 est un ingrédient, pas la décomposition complète) → à construire, mais méthodologie standard (pas de risque R&D).
- Couche IA texte/non structuré (météo, tendances, événements, concurrence) pour enrichir l'explication des écarts → exploratoire, rien d'existant à réutiliser.

## Données — écart entre existant et besoin

Aucun jeu de données actuel dans `Retail/` ne couvre le **détail transactionnel** demandé par le brief (ticket/commande, panier article, quantités, par heure, par zone géo) :

| Donnée requise par le brief | Disponible dans l'existant ? |
|---|---|
| Ventes journalières magasin × produit | ✅ `Sales_Forecasting/Datas` (mais 9 séries seulement, échelle test technique) |
| Taille magasin, type magasin | ✅ proxy dans `Walmart_Weather/data/stores.csv` |
| Promos avec profondeur remise | ✅ proxy dans `Walmart_Weather/data/features.csv` (MarkDown1-5) — mais pas de typologie de mécanique (6 types du brief) |
| Détail ticket/commande par heure | ❌ aucune source |
| Zone géographique (online) | ❌ aucune source |
| Calendrier promo avec métadonnées (nom campagne, périmètre, type) | ❌ aucune source structurée |
| Horaires/fréquentation concurrents | ❌ aucune source, à définir |
| Données RH (masse salariale, ETP actuels) | ❌ aucune source (déjà noté dans `POC_Pilotage_CA_Stock_RH`) |
| Météo | ✅ `Pilotage_StoreItem/src/fetch_weather.py`, `Synthese_Meteo` |

**Conclusion** : l'existant AOSIS donne un socle méthodologique et un squelette technique solides pour le **Bloc A** (et une partie du D), mais aucune donnée réelle du client n'est encore disponible. Le POC nécessitera soit des données réelles du client (idéal), soit un jeu de données synthétique/public reconstitué pour démontrer la mécanique (cf. points ouverts).
