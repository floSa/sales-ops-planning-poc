"""
Génération du jeu de données SYNTHÉTIQUE du POC MaxiZoo.

⚠️ Aucune donnée réelle MaxiZoo : tout est simulé (cf. data/README.md).
Seule la météo des villes est réelle (Open-Meteo, figée à la génération).

Grain natif : magasin × SKU × jour, sur 5 ans (2021-07-01 -> 2026-06-30),
12 magasins physiques (villes françaises réelles, taille ville -> taille
magasin) + 1 magasin virtuel ONLINE.

La demande générée encode des mécanismes que le pipeline de prévision devra
retrouver : saisonnalités hebdo/annuelle par catégorie, jours fériés France,
tendance par magasin (Online en forte croissance), inflation des prix,
6 typologies de promotions avec signatures d'effet distinctes, effets météo
(sur l'ANOMALIE de température, pas le niveau), ruptures de stock censurant
les ventes, lancements de produits en cours d'historique (cold start).

Usage :
    python -m src.generate_data [--synthetic-weather]
"""
from __future__ import annotations

import argparse
import sys

import holidays as pyholidays
import numpy as np
import pandas as pd

from src import config
from src.weather import add_climatology_anomaly, build_weather

# --------------------------------------------------------------------------- #
# Vérité terrain du générateur (les modèles ne la voient jamais directement)
# --------------------------------------------------------------------------- #
NEGBIN_DISPERSION = 2.2          # sur-dispersion gamma-poisson
STORE_DAY_SHOCK_SIGMA = 0.06     # choc de trafic magasin×jour partagé entre SKU

# Jours de la semaine (0 = lundi) — canal physique vs online
DOW_FACTOR_PHYSICAL = np.array([0.85, 0.90, 1.10, 0.90, 1.05, 1.50, 0.55])
DOW_FACTOR_ONLINE = np.array([1.10, 1.00, 1.00, 0.95, 0.95, 0.85, 1.15])

# Saisonnalité mensuelle par commodity group (janv -> déc)
MONTH_FACTOR = {
    "Chien":                [1.00, 0.97, 1.00, 1.00, 1.02, 1.03, 1.05, 1.03, 1.00, 1.00, 1.00, 1.08],
    "Chat":                 [1.00, 0.97, 1.00, 1.00, 1.02, 1.02, 1.04, 1.02, 1.00, 1.00, 1.02, 1.10],
    "Aquariophilie":        [0.95, 1.00, 1.15, 1.20, 1.15, 1.05, 0.95, 0.90, 0.95, 1.00, 0.95, 1.00],
    "Oiseau":               [1.25, 1.20, 1.05, 0.90, 0.85, 0.80, 0.80, 0.80, 0.90, 1.05, 1.20, 1.30],
    "Rongeur":              [0.98, 0.98, 1.00, 1.02, 1.02, 1.00, 1.00, 1.00, 1.00, 1.00, 1.02, 1.15],
    "Reptile":              [0.90, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.15, 1.05, 0.95, 0.90, 0.95],
    "Hygiène & Soins":      [0.75, 0.75, 0.90, 1.10, 1.25, 1.35, 1.40, 1.35, 1.15, 0.95, 0.80, 0.75],
    "Accessoires & Jouets": [0.90, 0.85, 0.90, 0.95, 1.00, 1.00, 0.95, 0.95, 0.95, 1.00, 1.15, 1.90],
}

# Tendance annuelle par magasin (croissance/décroissance structurelle)
STORE_TREND = {"S01": 0.020, "S02": 0.010, "S03": 0.025, "S04": 0.015,
               "S05": 0.020, "S06": 0.005, "S07": 0.010, "S08": 0.015,
               "S09": -0.015, "S10": 0.010, "S11": 0.000, "S12": -0.005,
               config.ONLINE_STORE_ID: 0.140}

# Panier article (PA) de base par type de magasin 🟡
PA_BASE = {"grand": 2.35, "moyen": 2.20, "petit": 2.05, "online": 2.75}

# Signatures d'effet des 6 typologies promo (vérité terrain)
UPLIFT_ELASTICITY_PRODUITS = 2.2   # facteur = 1 + 2.2 × profondeur de remise
PULL_FORWARD_FACTOR, PULL_FORWARD_DAYS = 0.88, 7
UPLIFT_MISE_EN_AVANT = 1.25
UPLIFT_INFLUENCE_PEAK = 0.50       # +50 % au pic, courbe décalée sur 28 jours
UPLIFT_SEUILS_QTY, SEUILS_PA_BOOST = 1.10, 1.08
UPLIFT_CADEAU_QTY, CADEAU_PA_BOOST = 1.08, 1.06
UPLIFT_OUVERTURE = 1.60
PROMO_PRIORITY = {"produits": 6, "influence": 5, "mise_en_avant": 4,
                  "cadeau_seuil": 3, "seuils": 2, "ouverture_magasin": 1}

RUPTURE_START_PROB = 0.004         # ~1 % de jours en rupture (épisodes de 1 à 5 j)

# --------------------------------------------------------------------------- #
# Catalogue produits : (libellé, prix de base €, ventes/jour magasin "moyen")
# --------------------------------------------------------------------------- #
PRODUCT_CATALOG = {
    "Chien": [
        ("Croquettes chien adulte 12 kg", 42.99, 6.0),
        ("Croquettes chiot poulet 4 kg", 19.99, 3.0),
        ("Croquettes chien senior light 8 kg", 34.99, 2.0),
        ("Pâtée chien bœuf 6×400 g", 9.49, 8.0),
        ("Friandises dentaires chien x28", 12.99, 5.0),
        ("Os à mâcher naturel", 4.49, 7.0),
        ("Laisse rétractable 5 m", 17.99, 0.8),
        ("Collier cuir taille M", 14.99, 0.6),
        ("Harnais confort taille L", 24.99, 0.7),
        ("Panier orthopédique 80 cm", 59.99, 0.25),
        ("Jouet corde nœud XL", 7.99, 1.2),
        ("Shampooing chien poils longs 300 ml", 9.99, 0.9),
        ("Sac de transport chien 10 kg", 39.99, 0.2),
        ("Gamelle antidérapante inox 1,5 L", 11.99, 0.5),
    ],
    "Chat": [
        ("Croquettes chat stérilisé 10 kg", 38.99, 6.0),
        ("Croquettes chaton 2 kg", 12.99, 3.0),
        ("Pâtée chat saumon 12×85 g", 8.99, 10.0),
        ("Friandises chat au malt x60", 6.99, 6.0),
        ("Litière agglomérante 15 L", 13.99, 9.0),
        ("Litière végétale 10 L", 11.49, 4.0),
        ("Maison de toilette XL", 34.99, 0.3),
        ("Arbre à chat 120 cm", 69.99, 0.35),
        ("Griffoir carton double", 9.99, 1.4),
        ("Jouet plumeau interactif", 5.99, 1.5),
        ("Herbe à chat pot 120 g", 3.49, 2.5),
        ("Fontaine à eau 2 L", 29.99, 0.6),
        ("Panier douillet gris", 22.99, 0.5),
        ("Collier réfléchissant chat", 6.49, 0.8),
    ],
    "Aquariophilie": [
        ("Aquarium 120 L équipé", 189.00, 0.08),
        ("Filtre interne 600 L/h", 27.99, 0.4),
        ("Chauffage aquarium 150 W", 21.99, 0.35),
        ("Flocons poissons tropicaux 250 ml", 6.99, 3.0),
        ("Granulés poissons de fond 120 g", 4.99, 2.0),
        ("Conditionneur d'eau 250 ml", 8.99, 1.5),
        ("Gravier naturel 5 kg", 7.49, 0.9),
    ],
    "Oiseau": [
        ("Mélange graines perruche 3 kg", 9.99, 3.0),
        ("Pâtée aux œufs canaris 1 kg", 7.49, 1.5),
        ("Boules de graisse x30", 8.99, 4.0),
        ("Cage perruche 60 cm", 49.99, 0.12),
        ("Mangeoire extérieure bois", 15.99, 0.7),
        ("Millet en grappes 300 g", 4.29, 1.8),
    ],
    "Rongeur": [
        ("Granulés lapin nain 3 kg", 8.99, 2.5),
        ("Foin de Crau 2,5 kg", 10.99, 3.0),
        ("Litière chanvre 30 L", 12.99, 2.0),
        ("Cage rongeur 100 cm", 64.99, 0.1),
        ("Roue d'exercice silencieuse", 13.99, 0.4),
        ("Friandises carotte rongeur", 3.99, 1.6),
    ],
    "Reptile": [
        ("Terrarium verre 80×40", 129.00, 0.05),
        ("Lampe UVB 100 W", 32.99, 0.3),
        ("Substrat tropical 20 L", 16.99, 0.35),
        ("Grillons lyophilisés 70 g", 5.99, 1.1),
    ],
    "Hygiène & Soins": [
        ("Pipettes antiparasitaires chien x6", 22.99, 2.5),
        ("Collier antiparasitaire chat", 18.99, 1.8),
        ("Spray antipuces habitat 500 ml", 14.99, 1.2),
        ("Vermifuge chien-chat x4", 11.99, 1.5),
        ("Brosse démêlante double", 12.49, 0.7),
    ],
    "Accessoires & Jouets": [
        ("Jouet peluche couinante", 6.99, 1.8),
        ("Balle distributrice de friandises", 9.99, 1.1),
        ("Tunnel de jeu pliable", 14.99, 0.6),
        ("Coffret cadeau Noël animaux", 19.99, 0.5),
    ],
}

# Lancements en cours d'historique (cold start) — validé le 2026-07-11
SKU_LAUNCHES = {
    "Croquettes chien senior light 8 kg": "2023-03-15",
    "Fontaine à eau 2 L": "2024-09-01",
    "Lampe UVB 100 W": "2025-02-15",
    "Coffret cadeau Noël animaux": "2025-10-01",
}

NATIONAL_BRANDS = {
    "Chien": "NutriPaws", "Chat": "FelixCo", "Aquariophilie": "AquaSphère",
    "Oiseau": "PlumeDor", "Rongeur": "ZooVital", "Reptile": "ReptiCare",
    "Hygiène & Soins": "SanoPet", "Accessoires & Jouets": "ZooVital",
}

# Inflation mensuelle m/m (%) — trajectoire inspirée de l'épisode 2021-2023 🟡
INFLATION_MM = {
    2021: [None] * 6 + [0.10, 0.15, 0.15, 0.20, 0.25, 0.20],
    2022: [0.30, 0.45, 0.60, 0.55, 0.50, 0.55, 0.45, 0.40, 0.50, 0.45, 0.35, 0.30],
    2023: [0.40, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.25, 0.20, 0.15, 0.10, 0.15],
    2024: [0.15, 0.20, 0.15, 0.15, 0.10, 0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.15],
    2025: [0.10, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10, 0.08, 0.10, 0.10, 0.08, 0.10],
    2026: [0.10, 0.10, 0.08, 0.10, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
}


# --------------------------------------------------------------------------- #
# Référentiels
# --------------------------------------------------------------------------- #
def build_stores() -> pd.DataFrame:
    rows = [dict(store_id=s, store_name=v, region=r, store_type=t, surface_m2=m2,
                 population=pop, latitude=la, longitude=lo, is_online=0)
            for s, v, r, t, m2, pop, la, lo in config.STORES]
    rows.append(dict(store_id=config.ONLINE_STORE_ID, store_name="Canal Online",
                     region="ONLINE", store_type="online", surface_m2=0,
                     population=0, latitude=np.nan, longitude=np.nan, is_online=1))
    return pd.DataFrame(rows)


def build_store_hours(stores: pd.DataFrame) -> pd.DataFrame:
    """Horaires d'ouverture par magasin × jour de semaine (intrant ETP)."""
    sunday_open_moyen = {"S05", "S06"}  # deux "moyens" ouvrent le dimanche matin
    rows = []
    for _, s in stores.iterrows():
        for dow in range(7):
            if s.is_online:
                rows.append(dict(store_id=s.store_id, day_of_week=dow,
                                 open_hour=0.0, close_hour=24.0, is_closed=0))
                continue
            close = {"grand": 20.0, "moyen": 19.5, "petit": 19.0}[s.store_type]
            if dow < 6:
                rows.append(dict(store_id=s.store_id, day_of_week=dow,
                                 open_hour=9.0, close_hour=close, is_closed=0))
            else:  # dimanche
                opens = (s.store_type == "grand"
                         or (s.store_type == "moyen" and s.store_id in sunday_open_moyen))
                rows.append(dict(store_id=s.store_id, day_of_week=6,
                                 open_hour=9.0 if opens else 0.0,
                                 close_hour=13.0 if opens else 0.0,
                                 is_closed=0 if opens else 1))
    return pd.DataFrame(rows)


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    i = 0
    for group, items in PRODUCT_CATALOG.items():
        for label, price, base_qty in items:
            i += 1
            u = rng.random()
            if u < config.SHARE_PL:
                brand, brand_type, is_pl, is_eb = "MaxiZoo Sélection", "distributeur", 1, 0
            elif u < config.SHARE_PL + config.SHARE_EB:
                brand, brand_type, is_pl, is_eb = "Ligne Prestige", "exclusive", 0, 1
            else:
                brand, brand_type, is_pl, is_eb = NATIONAL_BRANDS[group], "nationale", 0, 0
            rows.append(dict(sku_id=f"SKU{i:03d}", sku_label=label, commodity_group=group,
                             brand=brand, brand_type=brand_type, is_eb=is_eb, is_pl=is_pl,
                             base_price=price, base_qty=base_qty,
                             launch_date=SKU_LAUNCHES.get(label, "")))
    return pd.DataFrame(rows)


def build_inflation() -> pd.DataFrame:
    months = pd.period_range(config.HIST_START, config.FORECAST_END, freq="M")
    rates = [INFLATION_MM[m.year][m.month - 1] for m in months]
    cpi = 100.0 * np.cumprod(1.0 + np.asarray(rates) / 100.0)
    return pd.DataFrame({
        "month": months.astype(str),
        "inflation_mm_pct": rates,
        "cpi_index": cpi.round(3),
        "is_projection": [int(m > pd.Period(config.HIST_END, freq="M")) for m in months],
    })


def build_promos(products: pd.DataFrame, rng: np.random.Generator):
    """Calendrier promo à 6 typologies + périmètre SKU, historique ET futur (démo)."""
    equipment = products.loc[products.base_qty < 1.0, "sku_id"].tolist()
    chat = products.loc[products.commodity_group == "Chat", "sku_id"].tolist()
    chien = products.loc[products.commodity_group == "Chien", "sku_id"].tolist()
    hygiene = products.loc[products.commodity_group == "Hygiène & Soins", "sku_id"].tolist()
    noel = (products.loc[products.commodity_group == "Accessoires & Jouets", "sku_id"].tolist()
            + products.loc[products.sku_label.str.contains("Jouet"), "sku_id"].tolist())
    influence_pool = products.loc[
        products.commodity_group.isin(["Chien", "Chat", "Accessoires & Jouets"]), "sku_id"].tolist()

    promos, scope = [], []

    def add(name, ptype, mechanic, disc, start, end, perim, skus=None, store_id=""):
        pid = f"P{len(promos) + 1:03d}"
        promos.append(dict(promo_id=pid, campaign_name=name, promo_type=ptype,
                           mechanic=mechanic, discount_rate=disc,
                           date_start=start, date_end=end, perimeter=perim,
                           store_id=store_id))
        for sku in (skus or []):
            scope.append(dict(promo_id=pid, sku_id=sku))

    for y in range(2021, 2027):
        add(f"Soldes d'hiver {y}", "produits", "-25 % sélection", 0.25,
            f"{y}-01-08", f"{y}-02-04", "omnicanal",
            sorted(rng.choice(equipment, 12, replace=False).tolist()))
        add(f"Cat Days {y}", "produits", "-20 % univers chat", 0.20,
            f"{y}-03-10", f"{y}-03-23", "omnicanal",
            sorted(rng.choice(chat, 8, replace=False).tolist()))
        add(f"Anniversaire MaxiZoo {y}", "seuils", "-10 € dès 60 € d'achat", 0.0,
            f"{y}-05-12", f"{y}-05-25", "omnicanal")
        add(f"Été sans parasites {y}", "produits", "-15 % antiparasitaires", 0.15,
            f"{y}-06-05", f"{y}-06-25", "omnicanal", hygiene)
        add(f"Dog Days {y}", "produits", "-20 % univers chien", 0.20,
            f"{y}-09-08", f"{y}-09-21", "omnicanal",
            sorted(rng.choice(chien, 8, replace=False).tolist()))
        add(f"Black Friday {y}", "produits", "-30 % sélection", 0.30,
            f"{y}-11-22", f"{y}-11-29", "omnicanal",
            sorted(rng.choice(products.sku_id.tolist(), 15, replace=False).tolist()))
        add(f"Noël des animaux {y}", "mise_en_avant", "Têtes de gondole fêtes", 0.0,
            f"{y}-12-01", f"{y}-12-24", "omnicanal", sorted(set(noel)))
        add(f"Cadeau de Noël {y}", "cadeau_seuil", "Peluche offerte dès 40 €", 0.0,
            f"{y}-12-10", f"{y}-12-24", "omnicanal")
        if y >= 2022:  # le marketing d'influence démarre en 2022
            for mth, tag in [(4, "printemps"), (10, "automne")]:
                add(f"Influenceurs #MonAnimalEtMoi {tag} {y}", "influence",
                    "Produits offerts à 15 influenceurs", 0.0,
                    f"{y}-{mth:02d}-05", f"{y}-{mth:02d}-18", "online",
                    sorted(rng.choice(influence_pool, 5, replace=False).tolist()))

    add("Réouverture magasin Dijon", "ouverture_magasin", "Animations + goodies", 0.0,
        "2023-10-05", "2023-10-08", "magasin", store_id="S09")
    add("Magasin rénové Brive", "ouverture_magasin", "Animations + goodies", 0.0,
        "2025-04-10", "2025-04-13", "magasin", store_id="S12")

    promos_df = pd.DataFrame(promos)
    promos_df["date_start"] = pd.to_datetime(promos_df["date_start"])
    promos_df["date_end"] = pd.to_datetime(promos_df["date_end"])
    # on ne garde que les campagnes qui touchent [HIST_START, FORECAST_END]
    keep = (promos_df.date_end >= config.HIST_START) & (promos_df.date_start <= config.FORECAST_END)
    promos_df = promos_df[keep].reset_index(drop=True)
    scope_df = pd.DataFrame(scope)
    scope_df = scope_df[scope_df.promo_id.isin(promos_df.promo_id)].reset_index(drop=True)
    return promos_df, scope_df


# --------------------------------------------------------------------------- #
# Cœur : matrices (séries × jours)
# --------------------------------------------------------------------------- #
def _french_holidays(years):
    return pyholidays.France(years=list(years))


def generate_facts(stores, hours, products, promos, scope, inflation, wx,
                   rng: np.random.Generator):
    dates = pd.date_range(config.HIST_START, config.HIST_END, freq="D")
    D = len(dates)
    store_ids = stores.store_id.tolist()
    sku_ids = products.sku_id.tolist()
    n_stores, n_skus = len(store_ids), len(sku_ids)
    S = n_stores * n_skus  # séries = magasin × SKU
    store_of_series = np.repeat(np.arange(n_stores), n_skus)
    sku_of_series = np.tile(np.arange(n_skus), n_stores)

    dow = dates.dayofweek.values
    month_idx = dates.month.values - 1

    # --- facteur magasin (taille) et canal ---
    is_online_store = stores.is_online.values.astype(bool)
    surf = stores.surface_m2.values.astype(float)
    store_factor = np.where(is_online_store, 1.55, (np.maximum(surf, 1) / 950.0) ** 0.85)

    # --- jour de semaine par canal ---
    dow_factor = np.where(is_online_store[:, None],
                          DOW_FACTOR_ONLINE[dow][None, :],
                          DOW_FACTOR_PHYSICAL[dow][None, :])            # (stores, D)

    # --- saisonnalité mensuelle par groupe ---
    group_of_sku = products.commodity_group.values
    month_matrix = np.stack([np.asarray(MONTH_FACTOR[g])[month_idx] for g in group_of_sku])  # (skus, D)

    # --- tendance ---
    yrs = np.arange(D) / 365.25
    trend = np.stack([(1.0 + STORE_TREND[s]) ** yrs for s in store_ids])  # (stores, D)

    # --- statut d'ouverture (fériés France, fermetures) ---
    fr = _french_holidays(range(dates[0].year, dates[-1].year + 1))
    is_holiday = np.array([d.date() in fr for d in dates])
    is_closed_holiday = np.array([(d.month, d.day) in [(5, 1), (12, 25), (1, 1)] for d in dates])
    day_before_holiday = np.roll(is_holiday, -1)
    sunday_open = (hours[(hours.day_of_week == 6)].set_index("store_id").is_closed == 0)
    sunday_open = sunday_open.reindex(store_ids).values

    open_factor = np.ones((n_stores, D))
    phys = ~is_online_store
    # dimanche fermé pour certains magasins
    closed_sunday = np.outer(phys & ~sunday_open, dow == 6)
    open_factor[closed_sunday] = 0.0
    # fériés : fermé si 1er mai/Noël/jour de l'an ; sinon régime dimanche (0.7 si ouvert ce régime)
    holiday_closed = np.outer(phys, is_closed_holiday)
    open_factor[holiday_closed] = 0.0
    other_holiday = is_holiday & ~is_closed_holiday
    open_factor[np.outer(phys & ~sunday_open, other_holiday)] = 0.0
    open_factor[np.outer(phys & sunday_open, other_holiday)] *= 0.70
    # veille de férié : petit boost d'anticipation
    open_factor[np.outer(phys, day_before_holiday & ~is_holiday)] *= 1.10
    # ONLINE : jamais fermé, léger boost les fériés
    open_factor[np.outer(is_online_store, is_holiday)] *= 1.05

    # gradient de décembre (course aux fêtes puis creux post-Noël)
    dec_boost = np.ones(D)
    dec_days = (dates.month == 12) & (dates.day >= 15) & (dates.day <= 24)
    post_noel = (dates.month == 12) & (dates.day >= 26)
    dec_boost[dec_days], dec_boost[post_noel] = 1.12, 0.80

    # --- météo : effets sur l'ANOMALIE de température + pluie ---
    wx_h = wx[wx.date.isin(dates)]
    anom = wx_h.pivot(index="store_id", columns="date", values="temp_anomaly").reindex(store_ids).values
    rain = wx_h.pivot(index="store_id", columns="date", values="rain_mm").reindex(store_ids).values
    temp = wx_h.pivot(index="store_id", columns="date", values="temp_mean_c").reindex(store_ids).values

    hot = np.maximum(anom - 2.0, 0.0)      # épisodes nettement plus chauds que le normal
    cold = np.maximum(-anom - 2.0, 0.0)    # nettement plus froids
    traffic_weather = np.where(is_online_store[:, None],
                               np.minimum(1.0 + 0.012 * hot, 1.10) * np.where(rain > 6, 1.04, 1.0),
                               np.maximum(1.0 - 0.020 * hot, 0.84)
                               * np.where(rain > 6, 0.93, 1.0)
                               * np.maximum(1.0 - 0.008 * cold, 0.92))   # (stores, D)
    # effets par groupe : chaleur -> antiparasitaires/reptiles/aquario ; froid -> oiseaux du ciel
    grp_weather = np.ones((S, D), dtype=np.float32)
    warm_groups = {"Hygiène & Soins": 0.030, "Reptile": 0.020, "Aquariophilie": 0.012}
    for g, coef in warm_groups.items():
        mask = np.isin(sku_of_series, np.flatnonzero(group_of_sku == g))
        grp_weather[mask] = np.minimum(1.0 + coef * hot[store_of_series[mask]], 1.45)
    mask_oiseau = np.isin(sku_of_series, np.flatnonzero(group_of_sku == "Oiseau"))
    grp_weather[mask_oiseau] = np.minimum(1.0 + 0.045 * cold[store_of_series[mask_oiseau]], 1.60)

    # --- promotions : facteurs qty, remise prix, boost PA, id affiché ---
    promo_factor = np.ones((S, D), dtype=np.float32)
    price_discount = np.zeros((S, D), dtype=np.float32)
    pa_boost = np.ones((n_stores, D), dtype=np.float32)
    promo_idx = np.full((S, D), -1, dtype=np.int16)
    promo_prio = np.zeros((S, D), dtype=np.int8)

    sku_pos = {k: i for i, k in enumerate(sku_ids)}
    store_pos = {k: i for i, k in enumerate(store_ids)}
    scope_by_promo = scope.groupby("promo_id")["sku_id"].apply(list).to_dict()

    for p_i, p in promos.iterrows():
        dmask = (dates >= p.date_start) & (dates <= p.date_end)
        if not dmask.any():
            continue
        if p.perimeter == "magasin":
            smask = ~is_online_store
        elif p.perimeter == "online":
            smask = is_online_store.copy()
        else:
            smask = np.ones(n_stores, dtype=bool)
        if p.store_id:  # ouverture_magasin : un seul magasin
            smask = np.zeros(n_stores, dtype=bool)
            smask[store_pos[p.store_id]] = True
        scoped = scope_by_promo.get(p.promo_id)
        kmask = np.zeros(n_skus, dtype=bool)
        if scoped:
            kmask[[sku_pos[k] for k in scoped]] = True
        else:
            kmask[:] = True
        series_mask = smask[store_of_series] & kmask[sku_of_series]

        prio = PROMO_PRIORITY[p.promo_type]
        cell = np.outer(series_mask, dmask)

        if p.promo_type == "produits":
            promo_factor[cell] *= 1.0 + UPLIFT_ELASTICITY_PRODUITS * p.discount_rate
            price_discount[cell] = np.maximum(price_discount[cell], p.discount_rate)
            # pull-forward : creux sur les SKU ciblés la semaine suivant la promo
            end_i = int(np.searchsorted(dates, p.date_end))
            post = np.zeros(D, dtype=bool)
            post[end_i + 1:end_i + 1 + PULL_FORWARD_DAYS] = True
            promo_factor[np.outer(series_mask, post)] *= PULL_FORWARD_FACTOR
        elif p.promo_type == "mise_en_avant":
            promo_factor[cell] *= UPLIFT_MISE_EN_AVANT
        elif p.promo_type == "influence":
            # effet décalé : courbe en cloche sur 28 jours depuis le début
            start_i = int(np.searchsorted(dates, p.date_start))
            t = np.arange(D) - start_i
            curve = 1.0 + UPLIFT_INFLUENCE_PEAK * np.exp(-((t - 10) / 8.0) ** 2)
            curve[(t < 0) | (t > 28)] = 1.0
            promo_factor[series_mask] *= curve[None, :]
        elif p.promo_type == "seuils":
            promo_factor[cell] *= UPLIFT_SEUILS_QTY
            pa_boost[np.outer(smask, dmask)] *= SEUILS_PA_BOOST
        elif p.promo_type == "cadeau_seuil":
            promo_factor[cell] *= UPLIFT_CADEAU_QTY
            pa_boost[np.outer(smask, dmask)] *= CADEAU_PA_BOOST
        elif p.promo_type == "ouverture_magasin":
            promo_factor[cell] *= UPLIFT_OUVERTURE

        take = cell & (promo_prio < prio)
        promo_idx[take] = p_i
        promo_prio[take] = prio

    promo_factor = np.minimum(promo_factor, 3.5)

    # --- lancements (cold start) : absent avant, montée en charge 60 j après ---
    launched = np.ones((S, D), dtype=bool)
    ramp = np.ones((S, D), dtype=np.float32)
    for label, ld in SKU_LAUNCHES.items():
        k = sku_pos[products.loc[products.sku_label == label, "sku_id"].iloc[0]]
        li = int(np.searchsorted(dates, pd.Timestamp(ld)))
        rows = sku_of_series == k
        launched[rows, :li] = False
        t = np.maximum(np.arange(D) - li, 0)
        ramp[rows] = np.minimum(0.35 + 0.65 * t / 60.0, 1.0)[None, :]

    # --- intensité finale et tirage ---
    base = products.base_qty.values[sku_of_series] * store_factor[store_of_series]
    shock = rng.lognormal(mean=0.0, sigma=STORE_DAY_SHOCK_SIGMA, size=(n_stores, D))
    lam = (base[:, None]
           * dow_factor[store_of_series]
           * month_matrix[sku_of_series]
           * trend[store_of_series]
           * open_factor[store_of_series]
           * dec_boost[None, :]
           * traffic_weather[store_of_series]
           * grp_weather
           * promo_factor
           * ramp
           * shock[store_of_series]).astype(np.float64)
    lam[~launched] = 0.0

    qty_true = rng.poisson(rng.gamma(NEGBIN_DISPERSION, np.maximum(lam, 1e-12) / NEGBIN_DISPERSION))

    # --- ruptures : épisodes censurant la vente ---
    starts = rng.random((S, D)) < RUPTURE_START_PROB
    durations = rng.integers(1, 6, size=(S, D))
    rupture = np.zeros((S, D), dtype=bool)
    for k in range(5):
        rupture[:, k:] |= starts[:, :D - k] & (durations[:, :D - k] > k)
    rupture &= launched & (lam > 0)
    tau = rng.uniform(0.15, 0.60, size=(S, D))
    qty_obs = np.where(rupture, rng.binomial(qty_true, tau), qty_true)

    # --- prix : base × CPI mensuel × (1 - remise produits) ---
    cpi_by_month = inflation.set_index("month")["cpi_index"]
    cpi_day = cpi_by_month.reindex(dates.to_period("M").astype(str)).values / 100.0
    price = (products.base_price.values[sku_of_series][:, None]
             * cpi_day[None, :] * (1.0 - price_discount))
    price = np.round(price, 2)
    revenue = np.round(qty_obs * price, 2)

    # --- table de faits longue (on retire les jours pré-lancement) ---
    keep = launched.ravel()
    promo_ids_arr = promos.promo_id.values
    pid_flat = promo_idx.ravel()
    sales = pd.DataFrame({
        "date": np.tile(dates.values, S)[keep],
        "store_id": np.repeat(np.array(store_ids)[store_of_series], D)[keep],
        "sku_id": np.repeat(np.array(sku_ids)[sku_of_series], D)[keep],
        "quantity": qty_obs.ravel()[keep].astype(np.int32),
        "unit_price": price.ravel()[keep].astype(np.float32),
        "revenue": revenue.ravel()[keep].astype(np.float32),
        "is_rupture": rupture.ravel()[keep].astype(np.int8),
        "promo_id": np.where(pid_flat[keep] >= 0,
                             promo_ids_arr[np.maximum(pid_flat[keep], 0)], None),
    })

    # --- trafic (tickets/commandes) par magasin × jour ---
    qty_store_day = np.zeros((n_stores, D))
    np.add.at(qty_store_day, store_of_series, qty_obs * launched)
    doy = dates.dayofyear.values
    pa_base = np.array([PA_BASE["online"] if o else PA_BASE[t]
                        for o, t in zip(is_online_store, stores.store_type)])
    pa_day = (pa_base[:, None]
              * (1.0 + 0.02 * np.sin(2 * np.pi * doy / 365.25))[None, :]
              * pa_boost
              * rng.normal(1.0, 0.04, size=(n_stores, D)))
    tickets = np.round(qty_store_day / np.maximum(pa_day, 1.2)).astype(int)
    tickets = np.where(qty_store_day > 0, np.maximum(tickets, 1), 0)
    traffic = pd.DataFrame({
        "date": np.tile(dates.values, n_stores),
        "store_id": np.repeat(store_ids, D),
        "nb_tickets": tickets.ravel().astype(np.int32),
    })

    ca_store_day = np.zeros((n_stores, D))
    np.add.at(ca_store_day, store_of_series, revenue * launched)

    # --- répartition horaire réelle (intrant ETP) ---
    hourly = build_hourly(stores, hours, dates, ca_store_day, tickets,
                          is_holiday, is_closed_holiday, other_holiday, rng)

    return sales, traffic, hourly


def build_hourly(stores, hours, dates, ca_store_day, tickets,
                 is_holiday, is_closed_holiday, other_holiday, rng):
    """
    CA et tickets par magasin × jour × heure. Profils : creux 14-15 h, pics
    11-12 h et 17-19 h en semaine, après-midi chargé le samedi, soirée pour
    l'ONLINE. Sommes horaires == totaux journaliers (contrôlé).
    """
    D = len(dates)
    dow = dates.dayofweek.values
    hh = np.arange(24)
    store_ids = stores.store_id.tolist()
    frames = []

    for si, s in stores.iterrows():
        weights = np.zeros((D, 24))
        if s.is_online:
            base = (0.10 + 0.18 * (hh >= 7)
                    + 0.55 * np.exp(-((hh - 13.0) / 2.5) ** 2)
                    + 0.90 * np.exp(-((hh - 21.0) / 1.8) ** 2))
            weights[:] = base[None, :]
        else:
            sh = hours[(hours.store_id == s.store_id)].set_index("day_of_week")
            for d in range(D):
                dw = dow[d]
                if is_closed_holiday[d]:
                    continue
                if other_holiday[d]:  # régime dimanche : 9-13 si ouvert le dimanche
                    if sh.loc[6, "is_closed"]:
                        continue
                    o, c = 9.0, 13.0
                elif sh.loc[dw, "is_closed"]:
                    continue
                else:
                    o, c = sh.loc[dw, "open_hour"], sh.loc[dw, "close_hour"]
                if dw == 5:      # samedi : après-midi dense
                    prof = 0.50 + 0.55 * np.exp(-((hh - 11.5) / 1.8) ** 2) \
                                + 0.95 * np.exp(-((hh - 16.5) / 2.2) ** 2)
                elif c <= 13.5:  # matinée courte (dimanche/férié)
                    prof = 0.70 + 0.45 * (hh - o) / 4.0
                else:
                    prof = 0.55 + 0.50 * np.exp(-((hh - 11.3) / 1.6) ** 2) \
                                + 0.75 * np.exp(-((hh - 17.8) / 1.5) ** 2)
                open_mask = (hh >= o) & (hh < c)
                w = np.where(open_mask, prof, 0.0)
                # dernière heure partielle (ex. fermeture 19 h 30)
                frac = c - np.floor(c)
                if frac > 0:
                    w[int(np.floor(c))] = prof[int(np.floor(c))] * frac
                weights[d] = w

        # bruit Dirichlet (via gamma) pour varier le profil jour par jour
        alpha = weights * 60.0
        g = np.where(alpha > 0, rng.gamma(np.maximum(alpha, 1e-9)), 0.0)
        rs = g.sum(axis=1, keepdims=True)
        pvals = np.divide(g, rs, out=np.zeros_like(g), where=rs > 0)

        n_day = tickets[si]
        safe_pvals = np.where(pvals.sum(axis=1, keepdims=True) > 0, pvals, 1.0 / 24)
        tick_h = rng.multinomial(n_day, safe_pvals)
        tick_h[pvals.sum(axis=1) == 0] = 0
        ca_h = ca_store_day[si][:, None] * pvals

        open_cells = weights > 0
        d_idx, h_idx = np.nonzero(open_cells)
        frames.append(pd.DataFrame({
            "date": dates.values[d_idx],
            "store_id": store_ids[si],
            "hour": h_idx.astype(np.int8),
            "ca": np.round(ca_h[d_idx, h_idx], 2).astype(np.float32),
            "nb_tickets": tick_h[d_idx, h_idx].astype(np.int32),
        }))

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Contrôles de cohérence
# --------------------------------------------------------------------------- #
def run_checks(sales, traffic, hourly, products, promos, scope, wx) -> str:
    lines, failed = ["=== Contrôles de cohérence — jeu synthétique MaxiZoo ==="], []

    def check(name, ok, detail=""):
        lines.append(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failed.append(name)

    # 1. cohérence horaire <-> journalier
    ca_day = sales.groupby(["store_id", "date"], observed=True)["revenue"].sum()
    ca_h = hourly.groupby(["store_id", "date"], observed=True)["ca"].sum()
    diff = (ca_day - ca_h.reindex(ca_day.index).fillna(0)).abs()
    check("Σ CA horaire == CA journalier", bool((diff < 0.5).all()),
          f"écart max {diff.max():.3f} €")
    tick_day = traffic.set_index(["store_id", "date"])["nb_tickets"]
    tick_h = hourly.groupby(["store_id", "date"], observed=True)["nb_tickets"].sum()
    dt = (tick_day - tick_h.reindex(tick_day.index).fillna(0)).abs()
    check("Σ tickets horaires == tickets journaliers", bool((dt == 0).all()),
          f"écart max {dt.max()}")

    # 2. PA (articles/ticket) plausible
    qty_day = sales.groupby(["store_id", "date"], observed=True)["quantity"].sum()
    m = pd.DataFrame({"qty": qty_day, "tickets": tick_day}).query("tickets > 0")
    pa = (m.qty / m.tickets)
    check("PA moyen dans [1.2, 4.5]", bool(pa.mean() > 1.2 and pa.mean() < 4.5),
          f"PA moyen {pa.mean():.2f}")

    # 3. intermittence, ruptures, prix
    zero_share = float((sales.quantity == 0).mean())
    check("Part de lignes à zéro dans [0.15, 0.65]", 0.15 < zero_share < 0.65,
          f"{zero_share:.1%}")
    rupt = float(sales.is_rupture.mean())
    check("Part de lignes en rupture dans [0.3 %, 3 %]", 0.003 < rupt < 0.03, f"{rupt:.2%}")
    check("Aucun prix/CA/quantité négatif",
          bool((sales.unit_price > 0).all() and (sales.quantity >= 0).all()
               and (sales.revenue >= 0).all()))

    # 4. uplift promo produits détectable — ratio PAR SKU (médiane) pour
    # neutraliser l'effet de mix (les campagnes longues portent sur des SKU
    # d'équipement à faible volume, une moyenne brute écraserait le signal)
    prod_promos = promos[promos.promo_type == "produits"]
    on_promo = sales.promo_id.isin(prod_promos.promo_id)
    per_sku_promo = sales.loc[on_promo].groupby("sku_id", observed=True)["quantity"].mean()
    per_sku_base = (sales.loc[sales.promo_id.isna() & sales.sku_id.isin(per_sku_promo.index)]
                    .groupby("sku_id", observed=True)["quantity"].mean())
    ratios = (per_sku_promo / per_sku_base.reindex(per_sku_promo.index)).dropna()
    uplift = float(ratios.median())
    check("Uplift médian par SKU des promos 'produits' > 1.25", uplift > 1.25,
          f"×{uplift:.2f} (sur {len(ratios)} SKU)")

    # 5. cold start : rien avant le lancement
    for label, ld in SKU_LAUNCHES.items():
        sku = products.loc[products.sku_label == label, "sku_id"].iloc[0]
        before = sales[(sales.sku_id == sku) & (sales.date < pd.Timestamp(ld))]
        check(f"Cold start {sku} ({label[:30]}…) absent avant {ld}", len(before) == 0)

    # 6. météo complète et croissance Online
    check("Météo complète jusqu'à FORECAST_END",
          bool(wx.groupby("store_id")["date"].max().min() >= pd.Timestamp(config.FORECAST_END)))
    online = sales[sales.store_id == config.ONLINE_STORE_ID]
    y1 = online[online.date < pd.Timestamp("2022-07-01")].revenue.sum()
    y5 = online[online.date >= pd.Timestamp("2025-07-01")].revenue.sum()
    check("CA Online en forte croissance (dernière année > 1.5× première)",
          y5 > 1.5 * y1, f"×{y5 / max(y1, 1):.2f}")

    lines.append("")
    lines.append(f"Lignes sales_daily : {len(sales):,} | traffic : {len(traffic):,} "
                 f"| hourly : {len(hourly):,}")
    lines.append(f"CA total 5 ans : {sales.revenue.sum() / 1e6:.1f} M€ (synthétique)")
    report = "\n".join(lines)
    if failed:
        raise AssertionError("Contrôles en échec : " + ", ".join(failed) + "\n" + report)
    return report


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(synthetic_weather: bool = False):
    rng = np.random.default_rng(config.SEED)
    config.DATA_REF.mkdir(parents=True, exist_ok=True)
    config.DATA_TRX.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 Référentiels…")
    stores = build_stores()
    hours = build_store_hours(stores)
    products = build_products(rng)
    promos, scope = build_promos(products, rng)
    inflation = build_inflation()

    print("2/5 Météo des villes…")
    if synthetic_weather:
        import src.weather as W
        _orig = W._fetch_city_open_meteo
        W._fetch_city_open_meteo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forcé --synthetic-weather"))
        wx = build_weather(rng)
        W._fetch_city_open_meteo = _orig
    else:
        wx = build_weather(rng)
    wx = add_climatology_anomaly(wx)

    print("3/5 Faits (ventes, trafic, horaire)…")
    sales, traffic, hourly = generate_facts(stores, hours, products, promos, scope,
                                            inflation, wx, rng)

    print("4/5 Contrôles de cohérence…")
    report = run_checks(sales, traffic, hourly, products, promos, scope, wx)
    print(report)
    (config.RESULTS_DIR / "data_quality_report.txt").write_text(report, encoding="utf-8")

    print("5/5 Écriture des fichiers…")
    stores.to_csv(config.DATA_REF / "stores.csv", index=False)
    hours.to_csv(config.DATA_REF / "store_hours.csv", index=False)
    products.drop(columns=["base_qty"]).to_csv(config.DATA_REF / "products.csv", index=False)
    promos.to_csv(config.DATA_REF / "promo_calendar.csv", index=False)
    scope.to_csv(config.DATA_REF / "promo_scope.csv", index=False)
    inflation.to_csv(config.DATA_REF / "inflation.csv", index=False)
    wx.to_csv(config.DATA_REF / "weather.csv", index=False)
    sales.to_parquet(config.DATA_TRX / "sales_daily.parquet", index=False)
    traffic.to_parquet(config.DATA_TRX / "traffic_daily.parquet", index=False)
    hourly.to_parquet(config.DATA_TRX / "sales_hourly.parquet", index=False)
    print(f"Terminé. Données dans {config.DATA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-weather", action="store_true",
                    help="force le repli météo synthétique (pas d'appel Open-Meteo)")
    args = ap.parse_args()
    sys.exit(main(synthetic_weather=args.synthetic_weather))
