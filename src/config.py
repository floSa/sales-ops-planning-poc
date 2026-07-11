"""
Configuration centrale du POC Prévision des ventes MaxiZoo.

Toutes les données produites par ce projet sont SYNTHÉTIQUES (aucune donnée
réelle MaxiZoo). Seule exception, validée avec le donneur d'ordre : la météo
des villes-magasins est de la vraie météo historique (Open-Meteo), récupérée
UNE fois à la génération puis figée dans data/ — la demande, elle, est
générée synthétiquement en fonction de cette météo.

Les hypothèses de travail non validées par le client sont marquées 🟡
(cf. 00_Preparatif/03_Cadrage_Points_Ouverts.md §8).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_REF = DATA_DIR / "referentiel"
DATA_TRX = DATA_DIR / "transactions"
RESULTS_DIR = BASE_DIR / "results"

SEED = 42

# --------------------------------------------------------------------------- #
# Périmètre temporel
# --------------------------------------------------------------------------- #
HIST_START = "2021-07-01"   # 5 ans d'historique (validé le 2026-07-11)
HIST_END = "2026-06-30"     # dernier jour "réalisé"
FORECAST_END = "2026-12-31" # horizon de démo : atterrissage fin d'année

# --------------------------------------------------------------------------- #
# Référentiel magasins — villes françaises réelles ; la taille de la ville
# dimensionne la taille du magasin (validé le 2026-07-11)
# --------------------------------------------------------------------------- #
# (store_id, ville, region, type, surface_m2, population_ville, lat, lon)
STORES = [
    ("S01", "Paris",              "Île-de-France & Est", "grand", 1400, 2_133_000, 48.8566,  2.3522),
    ("S02", "Marseille",          "Sud",                 "grand", 1250,   870_000, 43.2965,  5.3698),
    ("S03", "Lyon",               "Sud",                 "grand", 1200,   520_000, 45.7640,  4.8357),
    ("S04", "Toulouse",           "Sud",                 "grand", 1150,   500_000, 43.6047,  1.4442),
    ("S05", "Nantes",             "Ouest",               "moyen",  950,   320_000, 47.2184, -1.5536),
    ("S06", "Strasbourg",         "Île-de-France & Est", "moyen",  900,   290_000, 48.5734,  7.7521),
    ("S07", "Bordeaux",           "Ouest",               "moyen",  880,   260_000, 44.8378, -0.5792),
    ("S08", "Rennes",             "Ouest",               "moyen",  820,   220_000, 48.1173, -1.6778),
    ("S09", "Dijon",              "Île-de-France & Est", "petit",  600,   160_000, 47.3220,  5.0415),
    ("S10", "Angers",             "Ouest",               "petit",  580,   155_000, 47.4784, -0.5632),
    ("S11", "Colmar",             "Île-de-France & Est", "petit",  520,    68_000, 48.0790,  7.3585),
    ("S12", "Brive-la-Gaillarde", "Sud",                 "petit",  450,    46_000, 45.1580,  1.5339),
]
ONLINE_STORE_ID = "ONLINE"  # magasin virtuel : canal e-commerce agrégé (cadrage §6)

# --------------------------------------------------------------------------- #
# Référentiel produits — 60 SKU / 8 commodity groups (validé le 2026-07-11)
# --------------------------------------------------------------------------- #
# commodity_group -> nombre de SKU
COMMODITY_GROUPS = {
    "Chien": 14,
    "Chat": 14,
    "Aquariophilie": 7,
    "Oiseau": 6,
    "Rongeur": 6,
    "Reptile": 4,
    "Hygiène & Soins": 5,
    "Accessoires & Jouets": 4,
}
N_SKU_LAUNCHES = 4          # SKU lancés en cours d'historique (démo cold start)

# 🟡 définitions provisoires (cadrage §8) : EB = exclusivité enseigne, PL = marque distributeur
SHARE_EB = 0.15
SHARE_PL = 0.35

# --------------------------------------------------------------------------- #
# Promotions — les 6 typologies du brief (1.2)
# --------------------------------------------------------------------------- #
PROMO_TYPES = [
    "produits",           # mécaniques remise sur SKU ciblés (Tract, Cat Days…)
    "seuils",             # remise dès X € d'achat -> dope le PA (articles/ticket)
    "ouverture_magasin",  # événement local -> pic de trafic du magasin
    "influence",          # produits offerts à influenceurs -> uplift décalé, sans remise
    "mise_en_avant",      # tête de gondole -> uplift modéré, sans remise
    "cadeau_seuil",       # cadeau dès X € -> dope le PA, sans remise
]

# --------------------------------------------------------------------------- #
# Modélisation (Bloc A) — reprise architecture Sales_Forecasting
# --------------------------------------------------------------------------- #
MODEL_MODE = "fast"         # "fast" = LightGBM seul | "ensemble" = XGB + LGBM + CatBoost
TWEEDIE_VARIANCE_POWER = 1.3
BACKTEST_N_FOLDS = 4
BACKTEST_HORIZON_DAYS = 28  # plis de 4 semaines, agrégeables en vue mensuelle
HIST_WINDOW_DAYS = 90       # fenêtre d'historique pour initialiser l'inférence récursive
SEASONAL_NAIVE_LAG = 7      # baseline à battre : naïve saisonnière hebdomadaire

# --------------------------------------------------------------------------- #
# Module ETP (Bloc C, version "interne seul") — 🟡 normes synthétiques,
# paramétrables dans le dashboard. Le canal ONLINE est exclu du calcul ETP
# (pas de magasin physique ; la logistique e-commerce est hors scope V1).
# --------------------------------------------------------------------------- #
ETP_PROD_CA_PER_HOUR = 220.0     # € de CA gérés par heure-vendeur 🟡
ETP_TICKETS_PER_HOUR = 12.0      # tickets encaissables par heure-vendeur 🟡
ETP_MIN_STAFF = 2.0              # effectif minimum de sécurité par heure ouvrée 🟡
ETP_HOURS_PER_MONTH = 151.67     # heures mensuelles d'un temps plein (35 h/sem)

# --------------------------------------------------------------------------- #
# Cascade CA (Bloc A §3 du cadrage) : PV × PA -> PM, (PM + inflation) × nb
# transactions -> CA net. L'inflation est un paramètre central mensuel 🟡.
# --------------------------------------------------------------------------- #
PA_SMOOTHING_WEEKS = 8      # 🟡 PA projeté = profil historique lissé (pas de 2e modèle)
