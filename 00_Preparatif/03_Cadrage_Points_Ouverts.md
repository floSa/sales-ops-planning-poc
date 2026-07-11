# Cadrage V1 — décisions arbitrées sans réponse client

Décisions prises le 2026-07-11, **sans validation client**, révisées le même jour après reformulation précise de chaque arbitrage (solution demandée par le brief vs solution(s) dégradée(s) proposées). Chaque décision est une hypothèse de travail raisonnable, pas une vérité confirmée — les points marqués `🟡 à valider avec le client` devront être revus dès que possible. Objectif : que la prochaine conversation attaque directement le développement, sans repasser par une phase de cadrage.

Convention utilisée dans ce document : pour chaque sujet, **Demandé** = ce que le brief client dit littéralement ; **Décision V1** = ce qu'on construit réellement, avec le nom de la version choisie si une version dégradée existe.

## 1. Scope fonctionnel V1

- **Bloc A — Moteur de prévision (stat + IA)** : INCLUS, sans dégradation majeure.
- **Bloc B — Outil de simulation/saisie (UI)** : INCLUS, en version dégradée **"Profil unique"** (cf. §3 — pas de moteur de droits multi-rôles).
- **Bloc C — Module ETP** (Équivalent Temps Plein — unité RH qui mesure un volume de travail normalisé sur un temps plein ; c'est le point 1.3 du brief, PAS la RH au sens large, ni la masse salariale) : INCLUS, en version dégradée **"ETP interne seul"** (cf. §5).
- **Bloc D — Analyse d'écarts + rolling forecast** : INCLUS, sans dégradation majeure (à l'exception de la couche données non structurées, cf. §6).

→ Tous les blocs du brief sont désormais couverts par le POC (Proof Of Concept), au moins en version dégradée. Aucun bloc n'est totalement hors scope.

## 2. Granularité (temps × produit)

- **Demandé** : le brief demande une donnée à la journée (axe temps "day", avec cumuls WTD — Week To Date, cumul depuis le début de la semaine — MTD — Month To Date — YTD — Year To Date) et au niveau article, c'est-à-dire au SKU (Stock Keeping Unit : une référence produit unique, ex. "Croquettes chien 5kg marque X").
- **Décision V1** : **Jour × Article (SKU)** — la solution demandée, retenue sans compromis. Le WTD/MTD/YTD (Week/Month/Year To Date) sont calculables nativement puisque la donnée de base est journalière.

**Conséquence architecturale (bonne nouvelle)** : cette maille correspond exactement au scope déjà traité par `Sales_Forecasting` dans `Retail/` (9 séries SKU × jour, ensemble de modèles de type GBM — Gradient Boosting Machine, une famille de modèles d'arbres boostés comme XGBoost/LightGBM/CatBoost — avec loss Tweedie et sample weights pour les ruptures de stock). Le pipeline existant est donc **directement transposable**, pas seulement une source d'inspiration méthodologique. Le wiki interne (`Wiki_Prevision_Ventes/00_Cadrage.md` §0.3) note que SKU × jour est la maille la plus dure à modéliser (beaucoup de séries avec beaucoup de zéros — "intermittence") : c'est un choix assumé, pas un choix par défaut.

**Architecture à deux niveaux qui en découle** :
1. Le moteur de prévision (Bloc A) prévoit la **quantité vendue par SKU (référence produit unique) × magasin × jour** — c'est le niveau natif du modèle, réutilisant directement l'architecture `Sales_Forecasting`.
2. Les agrégats magasin × jour (PA — Panier Article, PV — Prix de Vente, PM — Panier Moyen, nombre de transactions) utilisés dans la cascade de calcul (§3) sont obtenus en sommant/moyennant les prévisions SKU sur un magasin donné, un jour donné.
3. Les vues WTD/MTD/YTD (Week/Month/Year To Date) et la maille catégorie (commodity group) sont des agrégations de reporting au-dessus de ce niveau natif jour × SKU, pas des niveaux de modélisation séparés.

## 3. Cible et cascade de calcul

- **Demandé** : le brief décrit une cascade `PV (Prix de Vente) × PA (Panier Article, nombre moyen d'articles par ticket — 🟡 terme du brief non défini explicitement, hypothèse à confirmer) → PM (Panier Moyen) + inflation × nombre de transactions → CA (Chiffre d'Affaires) net`.
- **Décision V1** : pas de dégradation ici — le modèle prévoit la quantité vendue par SKU (référence produit unique) × jour (§2), ce qui permet de reconstituer transactions et PM (Panier Moyen) au niveau magasin × jour, puis d'appliquer la cascade du brief telle quelle pour obtenir le CA (Chiffre d'Affaires) net. C'est cohérent avec la recommandation du wiki (§0.2) de prévoir le volume d'abord et de reconstituer le CA (Chiffre d'Affaires) ensuite — il n'y a pas de vraie tension entre la demande du brief et la bonne pratique.

## 4. Consommateur simulé et droits multi-utilisateurs

- **Demandé** : le CDG (Contrôleur De Gestion) Ventes propose des hypothèses ; le DR (Directeur Régional, ~27 personnes) ou le DV (rôle non confirmé par le client, ~4 personnes 🟡) peuvent les modifier sur leur propre périmètre, sauf l'hypothèse d'inflation.
- **Décision V1** : version dégradée **"Profil unique"** — un seul profil simulé (CDG — Contrôleur De Gestion — Ventes), aucun mécanisme de droits ni de validation hiérarchique. Pas de CDG Online, pas de DR (Directeur Régional), pas de DV. Le canal Online reste présent **dans les données** (référentiel magasin) mais n'a pas d'écran ni de profil dédié en V1.

## 5. Module ETP (Bloc C)

- **Demandé** : calculer le besoin en ETP (Équivalent Temps Plein) par magasin, à partir du CA (Chiffre d'Affaires) prévisionnel par heure, de la fréquentation (nombre de tickets), des horaires du magasin, ET des horaires + pics de fréquentation des magasins concurrents dans la même zone commerciale.
- **Décision V1** : version dégradée **"ETP interne seul"** — calcul du besoin en ETP (Équivalent Temps Plein) à partir des seules données internes : CA (Chiffre d'Affaires) prévisionnel par heure, fréquentation (nombre de tickets), horaires d'ouverture/fermeture du magasin. **Pas de signal concurrentiel** (horaires et fréquentation des magasins concurrents) — aucune source de données identifiée pour ça (scraping, panel, étude terrain à envisager plus tard, hors POC).
- **Conséquence sur les données (§7)** : il faut désormais une vraie donnée horaire (CA et fréquentation par heure, par magasin) dans le jeu synthétique — ce n'est plus un détail cosmétique pour la démo, c'est un intrant nécessaire au calcul ETP (Équivalent Temps Plein).

## 6. Analyse d'écarts (Bloc D)

- **Demandé** : décomposition d'écart CA (Chiffre d'Affaires) réel/prévisionnel (effet prix, volume, mix, promo, calendaire) par magasin et par zone géographique pour l'Online ; enrichissement par données non structurées (météo, habitudes de consommation, tendances produits, événements, concurrence promo/communication) ; rolling forecast (atterrissage fin d'année).
- **Décision V1** :
  - Décomposition price/volume/mix/promo/calendar par magasin : INCLUS, sans dégradation.
  - Zone géographique pour l'Online : version dégradée **"Online agrégé"** — traité comme une seule entité, sans sous-zones. 🟡 à enrichir si des données réelles le permettent.
  - Données non structurées (météo, tendances, événements, concurrence) : version dégradée **"Hors scope"** — aucune source de ce type dans le jeu de données synthétique prévu (§7). Ce point du brief n'est pas traité en V1.
  - Rolling forecast (atterrissage fin d'année) : INCLUS, calculé mensuellement.

## 7. Données V1 : jeu synthétique inspiré de `Walmart_Weather`

Décision inchangée : pas d'attente d'un extrait réel du client — construction d'un jeu de données synthétique, structuré comme `Retail/Walmart_Weather` (déjà présent : `stores.csv` taille/type magasin, `features.csv` avec MarkDown promo, CPI, météo) mais **enrichi** pour coller au brief et aux décisions ci-dessus :

- Grain de base : magasin × SKU (référence produit unique) × jour, avec quantité vendue, PV (Prix de Vente).
- Répartition **horaire réelle** (pas cosmétique) de CA (Chiffre d'Affaires) et de fréquentation par magasin × jour, nécessaire au calcul ETP (Équivalent Temps Plein) interne (§5).
- Axe produit : commodity group / brand / brand type + flags binaires `EB` et `PL` (cf. §8, définitions provisoires).
- Calendrier promo structuré avec les **6 typologies** du brief (Produits, Seuils, Ouverture magasin, Influence, Mise en avant, Cadeau seuil) — pas seulement un pourcentage de remise agrégé comme dans MarkDown1-5.
- Horaires d'ouverture/fermeture par magasin (nécessaire au calcul ETP interne, §5).
- Un magasin "Online" virtuel dans le référentiel magasins.

**Première tâche de la prochaine conversation** : générer ce jeu de données (script dédié, à documenter comme les autres `Datas/`/`data/` du dossier `Retail/`).

## 8. Termes ambigus — hypothèses de travail

🟡 Toutes les définitions ci-dessous sont des hypothèses à faire confirmer par le client dès que possible ; elles ne bloquent pas le développement V1 mais peuvent nécessiter un ajustement du modèle de données ensuite.

- **EB** ("Enseigne / marque Exclusive Boutique") : flag binaire "produit vendu en exclusivité enseigne".
- **PL** (Private Label) : flag binaire "marque distributeur" — standard retail, moins ambigu que EB.
- **PA** (Panier Article) : le brief ne définit pas ce terme explicitement. Hypothèse retenue : nombre moyen d'articles par ticket/panier (pour que la formule `PV × PA → PM` soit cohérente : prix moyen × nombre d'articles = valeur moyenne du panier).
- **DV vs DR** (Directeur des Ventes/zone vs Directeur Régional) : non modélisé en V1 (profil unique, cf. §4) — la distinction hiérarchique exacte reste à clarifier avant toute V2 avec droits multi-niveaux.
- **Inflation** : en V1, paramètre saisi manuellement (% mensuel éditable), pas de flux externe (type indice INSEE) branché — cohérent avec le fait que c'est la seule hypothèse que DR (Directeur Régional)/DV ne peuvent *pas* amender dans le brief (donc plutôt une donnée de référence centrale que locale).

## 9. Métrique, backtest, baseline

- **Métrique** : WAPE (Weighted Absolute Percentage Error, une métrique d'erreur de prévision robuste aux ventes à zéro — jamais le MAPE, qui explose quand la vente réelle tend vers zéro) + biais, déclinés par magasin, par catégorie et par SKU (référence produit unique).
- **Backtest** : rolling-origin (fenêtre glissante), au jour, multi-plis — le backtest existant de `Sales_Forecasting` est mono-pli sur 7 jours (le wiki §0.5 note déjà ce manque), à étendre pour ce projet.
- **Baseline à battre** : naïve saisonnière hebdomadaire (valeur d'il y a 7 jours, "lag-7") — plus adaptée à une maille journalière que la naïve mensuelle (m-12) envisagée initialement.

## 10. Lien avec `POC_Pilotage_CA_Stock_RH`

Décision : chantiers gardés **distincts** pour l'instant, mais la frontière est maintenant plus précise qu'avant. Ce POC MaxiZoo calcule un **besoin en ETP** (Équivalent Temps Plein — un effectif, une unité de volume de travail), pas une **masse salariale** (un coût en euros). `POC_Pilotage_CA_Stock_RH` (dans `Retail/`) couvre la masse salariale : si les deux chantiers se rejoignent un jour, le pont naturel est "ETP (Équivalent Temps Plein) × coût moyen par ETP = masse salariale", mais ce calcul de coût n'est pas fait ici.

## Récapitulatif — livrable Phase 0 (format wiki §0.8)

| Point | Décision V1 |
|---|---|
| Cible | Quantité vendue par SKU (référence produit unique) × magasin × jour → agrégée en transactions + PM (Panier Moyen) → CA (Chiffre d'Affaires) net par cascade |
| Maille | Magasin × SKU (référence produit unique) × jour |
| Horizon | Mensuel glissant (agrégation de prévisions journalières) |
| Consommateur aval | CDG (Contrôleur De Gestion) Ventes (profil unique) |
| Métrique | WAPE (Weighted Absolute Percentage Error) + biais, par magasin/catégorie/SKU |
| Backtest | Rolling-origin journalier, multi-plis |
| Baseline | Naïve saisonnière hebdomadaire (lag-7) |
| Données | Synthétique, inspiré de Walmart_Weather + enrichissements (horaire réel, 6 typologies promo, EB/PL) |
| Scope fonctionnel | A (complet) + B (dégradé : profil unique) + C (dégradé : ETP interne seul) + D (complet, sauf données non structurées) |

## Prochaines étapes concrètes (prochaine conversation = développement)

1. Générer le jeu de données synthétique (§7) : magasin × SKU (référence produit unique) × jour, avec répartition horaire réelle, calendrier promo à 6 typologies, horaires magasin.
2. Reprendre directement l'architecture `Sales_Forecasting` (GBM — Gradient Boosting Machine — ensemble Tweedie + sample weights ruptures) pour le Bloc A, étendue en rolling-origin multi-plis.
3. Implémenter la cascade PV (Prix de Vente) × PA (Panier Article) → PM (Panier Moyen) + inflation × transactions → CA (Chiffre d'Affaires) net, au-dessus des prévisions SKU × jour agrégées.
4. Construire le calcul ETP (Équivalent Temps Plein) interne (CA/heure + fréquentation/heure + horaires magasin), sans signal concurrentiel.
5. Construire un dashboard léger de simulation mono-profil (CDG Ventes), réutilisant la base `Pilotage_StoreItem`.
6. Implémenter la décomposition d'écarts price/volume/mix/promo/calendar + rolling forecast mensuel (Bloc D), en s'appuyant sur `drift_monitor.py` existant.
