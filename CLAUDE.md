# MaxiZoo — POC Prévision des ventes

Dossier créé le 2026-07-11. Le brief client, l'audit de l'existant et le **cadrage V1 sont finalisés** (sans réponse client — décisions documentées comme hypothèses à valider). **La prochaine conversation attaque directement le développement.**

## Point de départ pour la prochaine conversation

Coller le prompt de [00_Preparatif/04_Prompt_Demarrage_POC.md](00_Preparatif/04_Prompt_Demarrage_POC.md) en tout début de conversation — il référence tout ce qui est nécessaire dans le bon ordre :
1. [03_Cadrage_Points_Ouverts.md](00_Preparatif/03_Cadrage_Points_Ouverts.md) — la spec V1 actionnable (scope, cible, maille, données, étapes concrètes).
2. [01_Brief_Consignes.md](00_Preparatif/01_Brief_Consignes.md) — brief client structuré, zones ambiguës signalées `⚠️`.
3. [02_Existant_Reutilisable.md](00_Preparatif/02_Existant_Reutilisable.md) — audit de ce qui, dans `Retail/` (OneDrive), est réutilisable.

## Scope V1 (décidé le 2026-07-11, révisé le même jour)

Les 4 blocs du brief sont tous couverts, au moins en version dégradée — aucun n'est hors scope :

- **Bloc A — moteur de prévision** : complet. Maille **magasin × SKU (Stock Keeping Unit, référence produit unique) × jour** (pas de compromis sur la granularité — réutilise directement l'architecture existante de `Sales_Forecasting`), cascade PV (Prix de Vente) × PA (Panier Article) → PM (Panier Moyen) + inflation × transactions → CA (Chiffre d'Affaires) net.
- **Bloc B — dashboard de simulation** : dégradé "**Profil unique**" — un seul profil (CDG Ventes, Contrôleur De Gestion), pas de moteur de droits multi-rôles (DR/DV).
- **Bloc C — module ETP** (Équivalent Temps Plein, ≠ RH au sens large, ≠ masse salariale) : dégradé "**ETP interne seul**" — calcul basé sur CA/fréquentation/horaires internes, sans signal concurrentiel (données concurrentes non disponibles).
- **Bloc D — analyse d'écarts + rolling forecast** : complet, sauf la couche données non structurées (météo/tendances/concurrence pour expliquer les écarts) qui reste hors scope V1.

Horizon mensuel glissant (agrégé depuis des prévisions journalières), données synthétiques inspirées de `Walmart_Weather`. Détail complet, rationale et définitions de chaque sigle : [03_Cadrage_Points_Ouverts.md](00_Preparatif/03_Cadrage_Points_Ouverts.md).

## Lien avec le reste du dossier `Retail/`

Ce chantier recoupe le brief déjà posé dans `Retail/POC_Pilotage_CA_Stock_RH/CLAUDE.md` (CA + stock + masse salariale) — les deux restent distincts pour l'instant : ce POC calcule un besoin en ETP (effectif), pas une masse salariale (coût en euros) — voir §10 de [03_Cadrage_Points_Ouverts.md](00_Preparatif/03_Cadrage_Points_Ouverts.md) pour le pont entre les deux si besoin plus tard. L'existant technique/méthodologique mobilisable se trouve dans `Retail/` (`Sales_Forecasting`, `Wiki_Prevision_Ventes`, `Pilotage_StoreItem`, `PowerBI_Pilotage`, `Walmart_Weather`).
