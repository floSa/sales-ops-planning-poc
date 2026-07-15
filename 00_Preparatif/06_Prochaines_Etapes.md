# Prochaines étapes — feuille de route

Suite logique de l'[état d'avancement](05_Etat_Avancement_vs_Consignes.md). La V1
livrée démontre la **mécanique de bout en bout sur données synthétiques**. Ce
document liste ce qui reste à faire pour passer du démonstrateur à un outil
exploitable, par ordre de priorité.

Effort indicatif : **S** = quelques jours · **M** = 1-2 semaines · **L** = plusieurs semaines.

---

## Priorité 1 — Le préalable à tout : les vraies données

Rien de ce qui suit n'a de valeur métier tant que le POC tourne sur du
synthétique. C'est le point bloquant numéro un.

- [ ] **Brancher un extrait réel MaxiZoo** (ventes ticket/commande, horaires, calendrier promo, référentiel produit/magasin) à la place du jeu synthétique. **L**
- [ ] **Re-mesurer les performances sur vraies données** : WAPE, biais, et surtout **l'apport réel de la météo** (le scénario 2) — son effet a été volontairement accentué dans la simulation, il faut le vérifier pour de bon. **M**
- [ ] **Valider les hypothèses 🟡** avec le client : définitions EB / PA, normes de productivité ETP, trajectoire d'inflation. **S**

---

## Priorité 2 — Compléter les fonctionnalités du brief

Les points du brief laissés de côté en V1 (les ⬜ de l'état d'avancement), une
fois les vraies données disponibles.

- [ ] **Gestion des droits multi-profils** — CDG Online, DR (~27), DV (~4) : chacun amende les hypothèses sur son périmètre, l'inflation restant centrale (brief 1.1, bloc B). **L**
- [ ] **Vues WTD / MTD / YTD** dans le dashboard (cumuls semaine/mois/année à date) — la donnée le permet déjà, il manque l'écran. **S**
- [ ] **Saisie du prix de vente à la maille article** (en plus de la saisie agrégée actuelle, brief 1.1). **S**
- [ ] **Inputs % impactant le CA** — trend marché, cannibalisation, concurrence (liste extensible, brief 1.1). **M**
- [ ] **Alertes automatiques + signal « tendances »** du scénario 2 (le brief prévoit des alertes IA ; seule la météo est branchée aujourd'hui). **M**
- [ ] **Online par zone géographique** dans l'analyse d'écarts (aujourd'hui agrégé, brief use case 2). **M**
- [ ] **Signal concurrentiel dans le calcul ETP** — horaires et pics de fréquentation des concurrents (brief 1.3). ⚠️ Nécessite une **source de données externe à définir** (scraping, panel, étude terrain). **L**

---

## Priorité 3 — Fiabiliser le moteur de prévision (data science)

Améliorations de la qualité prédictive, indépendantes du brief mais à forte
valeur.

- [x] **Prévision probabiliste** (fourchettes P10 / P90) — **version POC livrée** (`src/prediction_intervals.py`, affichée dans le dashboard) : intervalles calibrés empiriquement sur les erreurs du backtest (couverture ~80 % au grain magasin×jour), agrégés en quadrature avec corrélation calibrée. Reste à faire pour la cible pleine : **quantiles natifs** (LightGBM `objective="quantile"`) et validation de l'élargissement √(horizon) au-delà de 28 j. **M**
- [ ] **Réconciliation hiérarchique** — garantir que la somme des prévisions SKU = prévision catégorie = prévision magasin = national (cohérence multi-niveaux). **M**
- [ ] **Cannibalisation & effet halo** entre produits (une promo sur A déporte les ventes de son substitut B, ou dope un complément C). **L**
- [ ] **Amélioration du cold-start** — meilleure prévision des nouveaux produits / nouvelles ouvertures (similarité, attributs produit). **M**

---

## Priorité 4 — Industrialisation (MLOps)

Passage d'un script lancé à la main à un outil qui tourne tout seul.

- [ ] **Réentraînement automatique planifié** (batch nocturne / hebdomadaire). **M**
- [ ] **Monitoring de dérive en production** — surveiller que le modèle ne vieillit pas (la logique existe déjà dans `drift_monitor` côté `Retail/`, à brancher ici). **M**
- [ ] **Intégration continue** — les tests d'invariants (`tests/test_core.py`) tournent déjà ; les faire tourner automatiquement à chaque modification. **S**
- [ ] **Historisation des scénarios** — versionner les hypothèses et scénarios saisis (qui a proposé quoi, quand), pour la traçabilité et le suivi budget. **M**

---

## Priorité 5 — Analyse enrichie (exploratoire, ROI à valider)

- [ ] **Couche IA texte / données non structurées** pour expliquer les écarts — météo (au-delà du chiffre), habitudes conso, tendances produits, événements, communication concurrente (brief use case 2). Le plus exploratoire ; à cadrer et chiffrer avant de s'y engager. **L**

---

## Note de cadrage

- Les priorités 2 à 5 **dépendent toutes de la priorité 1** : sans vraies
  données, on continue d'optimiser une mécanique sur du synthétique.
- Deux chantiers nécessitent des **sources de données externes encore à
  identifier** : le signal concurrentiel de l'ETP (P2) et la couche non
  structurée (P5). À trancher tôt, car le sourcing peut être long.
- L'ordre reste indicatif : selon la priorité métier du client, on peut par
  exemple traiter les droits multi-profils (P2) avant les améliorations
  data science (P3).
