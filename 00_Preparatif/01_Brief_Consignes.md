# Brief client — Prévision des ventes (POC)

Brief posé le 2026-07-11 à l'issue d'un tour d'horizon des use cases prévision des ventes. Version nettoyée/structurée du compte-rendu transmis, **sans reformulation du fond** — les zones ambiguës du CR original sont signalées telles quelles avec `⚠️`.

## Axes d'analyse

- **Axe Organisationnel** : Magasin (le canal Online est traité comme un magasin) ; taille magasin — déjà géré via le module RH existant côté client.
- **Axe Produit** : commodity group, brand, brand type, avec deux attributs transverses :
  - **EB** ⚠️ *Enseigne / marque Exclusive Boutique — terme à confirmer avec le client, ambigu tel quel.*
  - **PL** — Private Label (marque distributeur).
- **Axe Temps** : day, week, year, avec analyses WTD (Week To Date), MTD (Month To Date), YTD (Year To Date).
- **Axe Transactionnel** : détail par magasin des tickets (prix de vente, panier article, quantités) par heure et par zone géographique. Pour l'Online, l'équivalent du ticket est la **commande** (mêmes données, vocabulaire différent).

## Périmètre & acteurs

- **Périmètre** : Magasin + Online.
- **People** :
  - CDG Ventes (Contrôleur De Gestion Ventes) — Julien
  - CDG Online (Contrôleur De Gestion Online) — Alexia
  - DR (Directeur Régional) — ~27 personnes
  - DV ⚠️ *Directeur des Ventes / zone — rôle et périmètre exact à confirmer, notamment vs DR — ~4 personnes*

## Use case 1 — Prévision des ventes (forecast et budget = même process)

### 1.1 Scénarios proposés par l'outil (IA, stat…)
Le CDG doit pouvoir simuler plusieurs scénarios :
- **Scénario 1 — Baseline** : historique corrigé des effets calendaires et des promos paramétrées.
- **Scénario 2 — Baseline + IA** : ajout d'éléments anticipés par l'IA (météo, tendances, etc.) avec système d'alertes.

Paramétrage d'hypothèses à la **maille mois**, pré-saisies par l'outil selon les 2 scénarios :
- **PV** (Prix de Vente) : saisie agrégée ou à la maille article.
- **PA** (Panier Article) → PV + PA donnent le **PM** (Panier Moyen).
- **Inflation** appliquée au PM.
- **Nombre de transactions**.
- Cascade de calcul : `PM + inflation × nb transactions → CA net`.
- Inputs % impactant le CA (liste extensible) : trend marché + cannibalisation, competition, compétition fight.
- 🚩 Calcul automatique des impacts promos et calendaires (ex. nb de dimanches/mois).
- 🚩 Les DR/DV peuvent modifier les hypothèses sur leur périmètre, **hors inflation**.

### 1.2 Calendrier commercial (promos)
Saisie : nom campagne, date début/fin, type (remise ou non), périmètre (magasin / online / omnicanal). L'outil propose une variation prix/qté basée sur l'historique (scénario 1) ou historique + facteurs externes (scénario 2), modifiable par l'utilisateur.

Types de promo à gérer (cf. annexe client) :
- Promotion Produits (mécaniques variées : Tract, Cat Days, Dog Days…)
- Promotions seuils (remise €, %, multi-seuils)
- Promotions ouverture magasin
- Influence (produits offerts à influenceurs — effet décalé dans le temps à modéliser)
- Mise en avant
- Cadeau seuil d'achat

### 1.3 Output module RH
Une fois le scénario définitif choisi, calcul du besoin ETP (Équivalent Temps Plein) par magasin à partir de :
- CA / magasin / heure
- Fréquentation par magasin (nb tickets)
- Horaires d'ouverture/fermeture par magasin
- Horaires des magasins concurrents sur la même zone commerciale + leurs pics de fréquentation
- Autres signaux d'aide à la décision (concurrence, météo/canicule…)

## Use case 2 — Analyse réel vs prévision

- Écarts CA réel/prévisionnel : effet prix, volume, mix, promo, calendaire — par magasin, et par zone géographique pour l'Online.
- Analyse enrichie par données non structurées (météo, habitudes conso, tendances produits, événements, concurrence promo/comm).
- Rolling forecast (atterrissage fin d'année) selon écarts observés, promos à venir, signaux externes.

## Clarification du besoin — 4 blocs à découpler

Le CR mélange 3 niveaux qu'il faut distinguer pour cadrer le projet :

| Bloc | Contenu | Nature |
|---|---|---|
| **A — Moteur de prévision (stat + IA)** | Baseline calendaire/promo, modèle enrichi IA (météo, tendances) + détection d'anomalies, cascade PV×PA→PM→+inflation×transactions→CA paramétrable | Cœur produit, data science |
| **B — Outil de simulation/saisie (UI métier)** | Simulateur multi-scénarios avec override, moteur calendrier promo (6 typologies), gestion des droits (CDG propose / DR-DV amendent sur leur périmètre) | Couche interaction, produit |
| **C — Module d'aide à la décision RH** | Croisement CA/fréquentation/horaires → recommandation ETP. Nécessite données concurrentes (horaires, fréquentation) → **source à définir** (scraping, panel, étude terrain ?) | Dérivé du forecast, dépendance data externe forte |
| **D — Analyse d'écarts + rolling forecast** | Décomposition d'écart classique (price/volume/mix/promo/calendar) = modélisation statistique standard. Couche IA texte/non structuré pour enrichir l'explication = plus exploratoire, ROI à valider | Post-forecast, monitoring |

→ Le cadrage de la prochaine conversation doit trancher **quel(s) bloc(s) attaquer en priorité pour le POC** (voir [03_Cadrage_Points_Ouverts.md](03_Cadrage_Points_Ouverts.md)).
