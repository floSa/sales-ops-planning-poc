# Prompt de démarrage — développement du POC MaxiZoo

Ce fichier contient le prompt à coller en tout début de la prochaine conversation (nouveau répertoire de travail : `MaxiZoo/`), pour attaquer directement le développement sans repasser par une phase de cadrage.

---

## Contexte

Tu interviens comme agent de développement data science pour AOSIS Consulting, sur un POC (Proof Of Concept — un démonstrateur de faisabilité, pas un produit final) de prévision des ventes pour le client MaxiZoo (magasins physiques + canal Online). Le cadrage a été fait dans une conversation précédente et est entièrement écrit dans `00_Preparatif/`. **Ne redemande pas de cadrage** : toutes les décisions nécessaires pour démarrer sont prises et documentées, y compris les hypothèses non validées par le client (marquées 🟡 dans les documents).

## Lectures obligatoires avant d'écrire la moindre ligne de code (dans l'ordre)

1. `00_Preparatif/03_Cadrage_Points_Ouverts.md` — **le document de référence**, c'est ta spec : scope exact, granularité, cascade de calcul, métriques, backtest, baseline, données à construire, étapes concrètes.
2. `00_Preparatif/01_Brief_Consignes.md` — le brief client original, pour le contexte métier.
3. `00_Preparatif/02_Existant_Reutilisable.md` — audit de ce qui est réutilisable dans `../Retail/` (dossier OneDrive voisin, pipelines et méthodologie déjà écrits pour d'autres projets AOSIS).
4. Le **code source réel** de `Retail/Sales_Forecasting/src/` (`cleaning.py`, `features_static_engineering.py`, `features_dynamic_engineering.py`, `modeling.py`, `backtest.py`, `inferencing.py`) — c'est le socle technique à reprendre et adapter, pas juste à survoler : la granularité retenue pour ce POC (magasin × article × jour) est exactement celle déjà traitée par ce pipeline.
5. `Retail/Wiki_Prevision_Ventes/00_Cadrage.md`, `06_Reconciliation_Hierarchique.md`, `07_Evaluation.md`, `09_Sujets_Retail.md` — méthodologie à respecter (métriques, protocole de backtest, promo/uplift).

## Objectif du POC

Démontrer la faisabilité d'un moteur de prévision des ventes par magasin × article (SKU — Stock Keeping Unit, une référence produit unique) × jour, avec : scénarios (historique corrigé vs. historique + signaux IA), calendrier promotionnel, cascade de calcul jusqu'au chiffre d'affaires (CA), calcul d'un besoin en effectif (ETP — Équivalent Temps Plein, une unité RH qui mesure un volume de travail normalisé sur un temps plein) en version interne simplifiée, et analyse des écarts réel/prévision avec rolling forecast (réestimation glissante de l'atterrissage de fin d'année). Le tout sur un jeu de données **synthétique** — aucune donnée réelle MaxiZoo n'est disponible à ce jour. L'objectif est de démontrer la mécanique bout-en-bout, pas de produire un chiffre exploitable tel quel par le client.

## Périmètre inclus, et à quel niveau

- **Moteur de prévision (Bloc A)** — complet. Prévision de la quantité vendue par magasin × SKU (référence produit unique) × jour, en reprenant l'architecture GBM (Gradient Boosting Machine — famille de modèles d'arbres boostés type XGBoost/LightGBM/CatBoost) + loss Tweedie + sample weights sur ruptures de `Sales_Forecasting`. Cascade `PV (Prix de Vente) × PA (Panier Article) → PM (Panier Moyen) + inflation × transactions → CA net` appliquée en agrégation au-dessus des prévisions SKU × jour.
- **Dashboard de simulation (Bloc B)** — version dégradée **"Profil unique"** : un seul profil utilisateur simulé (CDG Ventes — Contrôleur De Gestion), pas de moteur de droits multi-rôles.
- **Module ETP (Bloc C)** — version dégradée **"ETP interne seul"** : calcul du besoin en effectif à partir du CA prévisionnel par heure, de la fréquentation (nombre de tickets) et des horaires du magasin. Pas de signal concurrentiel (horaires/fréquentation des magasins concurrents).
- **Analyse d'écarts + rolling forecast (Bloc D)** — complet pour la décomposition price/volume/mix/promo/calendar et le rolling forecast mensuel. Version dégradée **"Online agrégé"** pour la ventilation géographique de l'Online (pas de sous-zones).

## Périmètre explicitement exclu

- Données non structurées (météo réelle, tendances, événements, concurrence promo/communication) pour enrichir l'analyse d'écarts.
- Moteur de droits multi-utilisateurs (CDG Online, DR — Directeur Régional —, DV).
- Signal concurrentiel dans le module ETP.
- Toute connexion à une vraie source de données externe : tout signal exogène (météo, tendances, marché) doit être **synthétique** et identifié comme tel dans le code/la doc.

## Contraintes techniques non négociables

- Métrique : **WAPE** (Weighted Absolute Percentage Error) + biais — jamais le MAPE.
- Backtest : rolling-origin (fenêtre glissante), multi-plis, au grain jour.
- Baseline à battre : naïve saisonnière hebdomadaire (valeur d'il y a 7 jours).
- Grain natif du modèle : magasin × SKU (référence produit unique) × jour. Les vues semaine/mois/année, WTD/MTD/YTD (Week/Month/Year To Date) et catégorie sont des agrégations de reporting au-dessus de ce grain, pas des niveaux de modélisation séparés.
- Toute donnée produite doit être clairement labellisée "synthétique" dans la documentation (README notamment) pour éviter toute confusion avec de vraies données MaxiZoo.

## Structure de projet attendue

Suivre la convention des autres sous-projets de `Retail/` : `data/` (ou `Datas/`), `src/`, `results/`, `README.md`, `requirements.txt`. Proposer une arborescence avant de commencer à écrire du code.

## Plan de travail suggéré

1. Générer le jeu de données synthétique : magasin × SKU × jour, avec répartition horaire réelle (nécessaire au calcul ETP), calendrier promo à 6 typologies (Produits, Seuils, Ouverture magasin, Influence, Mise en avant, Cadeau seuil), attributs produit (commodity group, brand, brand type, flags EB/PL), horaires magasin, magasin "Online" virtuel.
2. Adapter l'architecture `Sales_Forecasting` (GBM ensemble Tweedie + sample weights) à ce jeu de données, avec backtest rolling-origin multi-plis.
3. Implémenter la cascade PV × PA → PM + inflation × transactions → CA net, au-dessus des prévisions agrégées.
4. Implémenter le calcul ETP interne (CA/heure + fréquentation/heure + horaires magasin, sans signal concurrentiel).
5. Construire un dashboard léger mono-profil (CDG Ventes), en s'inspirant de `Pilotage_StoreItem/dashboard_pilotage.py`.
6. Implémenter la décomposition d'écarts price/volume/mix/promo/calendar + rolling forecast mensuel, en s'appuyant sur `Pilotage_StoreItem/src/drift_monitor.py`.

Tu peux réordonner ou paralléliser ces étapes si tu identifies une meilleure séquence, mais préviens-moi explicitement si tu t'écartes de ce plan.

## Comment traiter les zones d'incertitude

Tous les points marqués 🟡 dans `03_Cadrage_Points_Ouverts.md` (définitions de EB, PL, PA, rôle DV, source de l'inflation...) sont des hypothèses de travail, pas des blocages. Utilise-les sans t'arrêter pour demander confirmation — documente simplement, dans le code ou le README, qu'il s'agit d'hypothèses non validées par le client.

## Critères de succès du POC

- Pipeline exécutable de bout en bout : génération des données → prévision → cascade CA → calcul ETP → dashboard → analyse d'écarts.
- Backtest avec WAPE + biais, comparé explicitement à la baseline naïve.
- Documentation claire des hypothèses et limites (en particulier : "ceci n'est pas de la vraie donnée MaxiZoo").
- Démonstration présentable (dashboard fonctionnel, pas seulement des notebooks épars).

## Avant de coder

Présente d'abord un plan détaillé (todo list) issu de ta lecture des documents ci-dessus, avec en particulier le schéma des tables du jeu de données synthétique envisagé, pour validation rapide avant de lancer le développement.
