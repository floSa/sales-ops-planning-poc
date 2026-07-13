# État d'avancement vs consignes client

Revue point par point des consignes client d'origine (cf. [01_Brief_Consignes.md](01_Brief_Consignes.md)), établie le 2026-07-13 sur la base de la V1 livrée. Légende : ✅ fait · 🟡 fait partiellement (précisé) · ⬜ pas fait / reste à faire.

**Rappel transverse** : tout ce qui est ✅ ci-dessous tourne sur des **données 100 % synthétiques** (cf. [data/README.md](../data/README.md)) — la mécanique est démontrée bout en bout, mais aucun chiffre n'a de valeur métier tant que de vraies données MaxiZoo n'auront pas été branchées.

## Axes d'analyse

- ✅ **Axe Organisationnel** — Magasin (12 magasins + Online traité comme un magasin virtuel), taille magasin (dimensionnée par la taille de la ville)
- ✅ **Axe Produit** — commodity group, brand, brand type, flags EB et PL (🟡 définitions non validées client, documenté)
- 🟡 **Axe Temps** — grain jour natif, semaine/mois/année calculables par agrégation (cascade mensuelle, atterrissage annuel) ; **⬜ pas de bascule WTD/MTD/YTD dédiée dans le dashboard** — la donnée le permet, l'écran ne le propose pas
- 🟡 **Axe Transactionnel** — détail magasin par ticket (PV, panier article, quantités, par heure) ✅ complet pour les magasins physiques ; **⬜ zone géographique pour l'Online non traitée** (dégradé "Online agrégé", décidé au cadrage)

## Périmètre & acteurs

- ✅ **Périmètre Magasin + Online** — les deux présents dans la donnée et le moteur
- ✅ **CDG Ventes** (profil unique simulé)
- ⬜ **CDG Online, DR (~27), DV (~4)** — aucun moteur de droits multi-profils, aucune vue dédiée (dégradé "Profil unique", décidé au cadrage)

## Use case 1.1 — Scénarios proposés par l'outil

- ✅ **Scénario 1 — Baseline** (historique corrigé calendaire + promos)
- 🟡 **Scénario 2 — Baseline + IA** — météo intégrée et son apport mesuré (+5,1 % à la maille pilotage, jusqu'à +15,9 % les jours de pluie) ; **⬜ "tendances" et alertes automatiques non implémentées**
- ✅ **Paramétrage à la maille mois** (PV, PA, transactions, inflation éditables)
- 🟡 **PV** — saisie agrégée ✅ ; **⬜ saisie à la maille article non proposée**
- ✅ **PA + PV → PM**
- ✅ **Inflation appliquée au PM**
- ✅ **Nombre de transactions**
- ✅ **PM + inflation × transactions → CA net** (cascade exacte, testée)
- ⬜ **Inputs % impactant le CA** (trend marché, cannibalisation, competition, compétition fight) — non implémentés
- ✅ **Calcul auto des impacts promos et calendaires** — dans le moteur de prévision (features calendaires) et dans la décomposition d'écarts (effet promo/calendaire isolé)
- ⬜ **DR/DV modifient les hypothèses sur leur périmètre** — pas de moteur de droits (dégradé, décidé au cadrage)

## Use case 1.2 — Calendrier commercial (promos)

- ✅ **Saisie campagne** — nom, dates, type, périmètre (magasin/online/omnicanal)
- 🟡 **Proposition prix/qté par l'outil** — uplift proposé depuis l'historique ✅ ; **⬜ la proposition ne varie pas encore selon le scénario 1 vs 2** (pas de facteurs externes injectés dans la proposition elle-même)
- ✅ **Modifiable par l'utilisateur**
- ✅ **Les 6 typologies de promo** — Produits, Seuils, Ouverture magasin, Influence (avec effet décalé dans le temps modélisé), Mise en avant, Cadeau seuil

## Use case 1.3 — Output module RH (ETP)

- ✅ **CA / magasin / heure**
- ✅ **Fréquentation par magasin (tickets)**
- ✅ **Horaires d'ouverture/fermeture par magasin**
- ⬜ **Horaires et pics de fréquentation des concurrents** — aucune source identifiée (dégradé "ETP interne seul", décidé au cadrage)
- ⬜ **Autres signaux (concurrence, météo/canicule) dans le calcul ETP** — la météo sert au forecast des ventes, pas au staffing

## Use case 2 — Analyse réel vs prévision

- ✅ **Décomposition d'écart** — effet prix, volume, mix (identité comptable exacte, testée) + reventilation promo/calendaire/autre
- ✅ **Par magasin**
- ⬜ **Par zone géographique pour l'Online** — dégradé "Online agrégé" (décidé au cadrage)
- 🟡 **Enrichissement données non structurées** — météo ✅ (repliée dans "effet autre") ; **⬜ habitudes conso, tendances produits, événements, concurrence promo/comm** — hors scope (décidé au cadrage, aucune source)
- 🟡 **Rolling forecast / atterrissage fin d'année** — réel YTD + prévision reste d'année ✅ ; **⬜ pas de recalibration dynamique du reste d'année à partir des écarts observés récents** (l'atterrissage additionne, il ne réajuste pas automatiquement)

## Vue synthétique par bloc (A/B/C/D)

| Bloc | Statut |
|---|---|
| **A — Moteur de prévision** | ✅ baseline + cascade complètes · 🟡 IA météo oui, tendances/alertes non |
| **B — Simulation/saisie (UI)** | ✅ multi-scénarios + 6 typologies promo · ⬜ gestion des droits DR/DV |
| **C — Module RH (ETP)** | ✅ CA/fréquentation/horaires internes · ⬜ signal concurrentiel |
| **D — Écarts + rolling forecast** | ✅ décomposition classique · ⬜ couche IA texte/non structuré |
