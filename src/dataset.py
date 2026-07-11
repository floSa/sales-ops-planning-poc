"""
Construction du panel de modélisation (Bloc A) — adaptation de
Retail/Sales_Forecasting (cleaning.py + features_static_engineering.py)
à la maille magasin × SKU × jour du POC MaxiZoo.

Rôles :
  - chargement des tables synthétiques (data/),
  - sample weights sur ruptures (demande ≠ ventes, wiki §0.1 : on ne corrige
    pas la cible, on signale au modèle sa fiabilité variable),
  - features STATIQUES (connues à l'avance, sans fuite temporelle) :
    calendaire France, horaires/ouverture, promos par typologie, prix relatif,
    attributs produit/magasin, météo (scénario 2 uniquement).

Les features DYNAMIQUES (lags/rollings, dépendantes de la cible) sont dans
features_dynamic.py et recalculées à chaque pli/inférence (anti-fuite).
"""
from __future__ import annotations

import holidays as pyholidays
import numpy as np
import pandas as pd

from src import config

KEYS = ["store_id", "sku_id"]

# Fermetures totales des magasins physiques (le reste des fériés = régime dimanche)
FULL_CLOSE_DAYS = [(5, 1), (12, 25), (1, 1)]

WEATHER_FEATURES = ["temp_anomaly", "rain_mm"]

CATEGORICAL_FEATURES = ["store_code", "sku_code", "commodity_code", "brand_type_code",
                        "region_code", "store_type_code"]

STATIC_FEATURES = CATEGORICAL_FEATURES + [
    "is_online", "surface_m2", "is_eb", "is_pl", "base_price",
    "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_holiday", "is_day_before_holiday", "is_pont",
    "days_until_holiday", "days_since_holiday",
    "is_open", "open_hours", "days_since_start",
    "rel_price", "promo_discount", "is_promo_produits", "is_promo_mea",
    "is_promo_influence", "influence_day", "is_promo_seuils",
    "is_promo_cadeau", "is_promo_ouverture", "is_post_promo",
]


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_tables() -> dict:
    t = {
        "stores": pd.read_csv(config.DATA_REF / "stores.csv"),
        "hours": pd.read_csv(config.DATA_REF / "store_hours.csv"),
        "products": pd.read_csv(config.DATA_REF / "products.csv"),
        "promos": pd.read_csv(config.DATA_REF / "promo_calendar.csv",
                              parse_dates=["date_start", "date_end"]),
        "scope": pd.read_csv(config.DATA_REF / "promo_scope.csv"),
        "inflation": pd.read_csv(config.DATA_REF / "inflation.csv"),
        "weather": pd.read_csv(config.DATA_REF / "weather.csv", parse_dates=["date"]),
        "sales": pd.read_parquet(config.DATA_TRX / "sales_daily.parquet"),
        "traffic": pd.read_parquet(config.DATA_TRX / "traffic_daily.parquet"),
        "hourly": pd.read_parquet(config.DATA_TRX / "sales_hourly.parquet"),
    }
    t["sales"]["date"] = pd.to_datetime(t["sales"]["date"])
    t["traffic"]["date"] = pd.to_datetime(t["traffic"]["date"])
    t["hourly"]["date"] = pd.to_datetime(t["hourly"]["date"])
    t["promos"]["store_id"] = t["promos"]["store_id"].fillna("")
    return t


# --------------------------------------------------------------------------- #
# Features calendaires (par date, vectorisé)
# --------------------------------------------------------------------------- #
def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    fr = pyholidays.France(years=range(dates[0].year, dates[-1].year + 1))
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day
    df["month"] = dates.month
    df["week_of_year"] = dates.isocalendar().week.astype(int).values
    df["is_weekend"] = (dates.dayofweek >= 5).astype(np.int8)
    df["dow_sin"] = np.sin(2 * np.pi * dates.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dates.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dates.dayofyear / 365.25)

    is_hol = np.array([d.date() in fr for d in dates])
    df["is_holiday"] = is_hol.astype(np.int8)
    df["is_day_before_holiday"] = (np.roll(is_hol, -1) & ~is_hol).astype(np.int8)
    # ponts : lundi avec mardi férié, vendredi avec jeudi férié
    dow = dates.dayofweek.values
    df["is_pont"] = (((dow == 0) & np.roll(is_hol, -1)) |
                     ((dow == 4) & np.roll(is_hol, 1))).astype(np.int8)
    # comptes à rebours vers/depuis le férié le plus proche (vectorisé)
    hol_idx = np.flatnonzero(is_hol)
    idx = np.arange(len(dates))
    if len(hol_idx):
        nxt = hol_idx[np.minimum(np.searchsorted(hol_idx, idx), len(hol_idx) - 1)]
        prv = hol_idx[np.maximum(np.searchsorted(hol_idx, idx, side="right") - 1, 0)]
        df["days_until_holiday"] = np.clip(nxt - idx, 0, 60).astype(np.int16)
        df["days_since_holiday"] = np.clip(idx - prv, 0, 60).astype(np.int16)
    else:
        df["days_until_holiday"] = 60
        df["days_since_holiday"] = 60
    df["days_since_start"] = (dates - pd.Timestamp(config.HIST_START)).days
    return df


# --------------------------------------------------------------------------- #
# Ouverture effective magasin × date (mêmes règles que le générateur)
# --------------------------------------------------------------------------- #
def build_open_hours(stores: pd.DataFrame, hours: pd.DataFrame,
                     dates: pd.DatetimeIndex) -> pd.DataFrame:
    fr = pyholidays.France(years=range(dates[0].year, dates[-1].year + 1))
    is_hol = np.array([d.date() in fr for d in dates])
    full_close = np.array([(d.month, d.day) in FULL_CLOSE_DAYS for d in dates])
    other_hol = is_hol & ~full_close
    dow = dates.dayofweek.values

    frames = []
    for _, s in stores.iterrows():
        sh = hours[hours.store_id == s.store_id].set_index("day_of_week")
        oh = (sh.close_hour - sh.open_hour).reindex(dow).values  # heures par dow
        closed_dow = sh.is_closed.reindex(dow).values.astype(bool)
        oh = np.where(closed_dow, 0.0, oh)
        if not s.is_online:
            sunday_open = sh.loc[6, "is_closed"] == 0
            oh = np.where(full_close, 0.0, oh)
            # autres fériés : régime dimanche (9-13 si ouvert le dimanche, sinon fermé)
            oh = np.where(other_hol, 4.0 if sunday_open else 0.0, oh)
        frames.append(pd.DataFrame({
            "date": dates, "store_id": s.store_id,
            "open_hours": oh.astype(np.float32),
            "is_open": (oh > 0).astype(np.int8),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Features promo par typologie (magasin × SKU × date, vectorisé)
# --------------------------------------------------------------------------- #
def build_promo_features(stores: pd.DataFrame, products: pd.DataFrame,
                         promos: pd.DataFrame, scope: pd.DataFrame,
                         dates: pd.DatetimeIndex) -> pd.DataFrame:
    store_ids, sku_ids = stores.store_id.tolist(), products.sku_id.tolist()
    n_st, n_sk, D = len(store_ids), len(sku_ids), len(dates)
    S = n_st * n_sk
    st_of = np.repeat(np.arange(n_st), n_sk)
    sk_of = np.tile(np.arange(n_sk), n_st)
    is_online = stores.is_online.values.astype(bool)
    sku_pos = {k: i for i, k in enumerate(sku_ids)}
    store_pos = {k: i for i, k in enumerate(store_ids)}
    scope_by_promo = scope.groupby("promo_id")["sku_id"].apply(list).to_dict()

    cols = {
        "promo_discount": np.zeros((S, D), np.float32),
        "is_promo_produits": np.zeros((S, D), np.int8),
        "is_promo_mea": np.zeros((S, D), np.int8),
        "is_promo_influence": np.zeros((S, D), np.int8),
        "influence_day": np.full((S, D), -1, np.int16),
        "is_promo_seuils": np.zeros((S, D), np.int8),
        "is_promo_cadeau": np.zeros((S, D), np.int8),
        "is_promo_ouverture": np.zeros((S, D), np.int8),
        "is_post_promo": np.zeros((S, D), np.int8),
    }

    for _, p in promos.iterrows():
        dmask = (dates >= p.date_start) & (dates <= p.date_end)
        if p.perimeter == "magasin":
            smask = ~is_online
        elif p.perimeter == "online":
            smask = is_online.copy()
        else:
            smask = np.ones(n_st, dtype=bool)
        if p.store_id:
            smask = np.zeros(n_st, dtype=bool)
            smask[store_pos[p.store_id]] = True
        scoped = scope_by_promo.get(p.promo_id)
        kmask = np.zeros(n_sk, dtype=bool)
        if scoped:
            kmask[[sku_pos[k] for k in scoped if k in sku_pos]] = True
        else:
            kmask[:] = True
        series = smask[st_of] & kmask[sk_of]
        if not dmask.any() or not series.any():
            continue
        cell = np.outer(series, dmask)

        if p.promo_type == "produits":
            cols["is_promo_produits"][cell] = 1
            cols["promo_discount"][cell] = np.maximum(cols["promo_discount"][cell],
                                                      p.discount_rate)
            end_i = int(np.searchsorted(dates, p.date_end))
            post = np.zeros(D, dtype=bool)
            post[end_i + 1:end_i + 8] = True
            cols["is_post_promo"][np.outer(series, post)] = 1
        elif p.promo_type == "mise_en_avant":
            cols["is_promo_mea"][cell] = 1
        elif p.promo_type == "influence":
            start_i = int(np.searchsorted(dates, p.date_start))
            t = np.arange(D) - start_i
            win = (t >= 0) & (t <= 28)
            cols["is_promo_influence"][np.outer(series, win)] = 1
            tday = np.where(win, t, -1).astype(np.int16)
            block = cols["influence_day"][series]
            cols["influence_day"][series] = np.maximum(block, tday[None, :])
        elif p.promo_type == "seuils":
            cols["is_promo_seuils"][cell] = 1
        elif p.promo_type == "cadeau_seuil":
            cols["is_promo_cadeau"][cell] = 1
        elif p.promo_type == "ouverture_magasin":
            cols["is_promo_ouverture"][cell] = 1

    out = pd.DataFrame({
        "store_id": np.repeat(np.array(store_ids)[st_of], D),
        "sku_id": np.repeat(np.array(sku_ids)[sk_of], D),
        "date": np.tile(dates.values, S),
    })
    for name, mat in cols.items():
        out[name] = mat.ravel()
    return out


# --------------------------------------------------------------------------- #
# Panel complet
# --------------------------------------------------------------------------- #
def build_panel(start: str, end: str, include_weather: bool = False,
                tables: dict | None = None):
    """
    Grille magasin × SKU × jour sur [start, end], features statiques jointes.

    Les jours antérieurs au lancement d'un SKU sont exclus (cold start).
    Les jours sans réalisé (futur) ont quantity=NaN — à prédire.

    Returns:
        (df, features, categoricals) : panel trié par clés+date, liste des
        features à donner au modèle (selon scénario), liste des catégorielles.
    """
    t = tables or load_tables()
    dates = pd.date_range(start, end, freq="D")
    stores, products = t["stores"], t["products"]

    grid = pd.MultiIndex.from_product(
        [stores.store_id, products.sku_id, dates],
        names=["store_id", "sku_id", "date"]).to_frame(index=False)

    # cold start : pas de ligne avant lancement
    launch = products.set_index("sku_id")["launch_date"]
    launch = pd.to_datetime(launch.replace("", pd.NaT))
    grid = grid[~(grid.date < grid.sku_id.map(launch))].reset_index(drop=True)

    df = grid.merge(t["sales"][KEYS + ["date", "quantity", "unit_price", "is_rupture"]],
                    on=KEYS + ["date"], how="left")

    # --- sample weights (stratégie Sales_Forecasting adaptée : pas de NaN de
    # quantité dans le synthétique, la fiabilité vient du flag rupture) ---
    df["is_rupture"] = df["is_rupture"].fillna(0).astype(np.int8)
    w = np.ones(len(df), dtype=np.float32)
    w[(df.is_rupture == 1) & (df.quantity > 0)] = 0.5   # vente tronquée
    w[(df.is_rupture == 1) & (df.quantity == 0)] = 0.1  # rupture totale
    df["sample_weight"] = w

    # --- attributs magasin / produit + encodages catégoriels stables ---
    df = df.merge(stores[["store_id", "region", "store_type", "surface_m2", "is_online"]],
                  on="store_id", how="left")
    df = df.merge(products[["sku_id", "commodity_group", "brand_type",
                            "is_eb", "is_pl", "base_price"]], on="sku_id", how="left")
    df["store_code"] = pd.Categorical(df.store_id, categories=sorted(stores.store_id)).codes
    df["sku_code"] = pd.Categorical(df.sku_id, categories=sorted(products.sku_id)).codes
    df["commodity_code"] = pd.Categorical(
        df.commodity_group, categories=sorted(products.commodity_group.unique())).codes
    df["brand_type_code"] = pd.Categorical(
        df.brand_type, categories=sorted(products.brand_type.unique())).codes
    df["region_code"] = pd.Categorical(
        df.region, categories=sorted(stores.region.unique())).codes
    df["store_type_code"] = pd.Categorical(
        df.store_type, categories=sorted(stores.store_type.unique())).codes

    # --- calendrier, ouverture, promos ---
    df = df.merge(build_calendar(dates), on="date", how="left")
    df = df.merge(build_open_hours(stores, t["hours"], dates),
                  on=["store_id", "date"], how="left")
    df = df.merge(build_promo_features(stores, products, t["promos"], t["scope"], dates),
                  on=KEYS + ["date"], how="left")

    # --- prix relatif : réalisé si dispo, sinon prix théorique (base × CPI ×
    # (1 - remise)) — même formule passé/futur, pas de fuite ---
    cpi = t["inflation"].set_index("month")["cpi_index"]
    df["cpi"] = df.date.dt.to_period("M").astype(str).map(cpi).astype(float) / 100.0
    theo = df.base_price * df.cpi * (1.0 - df.promo_discount)
    df["unit_price"] = df.unit_price.astype(float).fillna(theo.round(2))
    df["rel_price"] = (df.unit_price / (df.base_price * df.cpi)).astype(np.float32)

    features = list(STATIC_FEATURES)
    if include_weather:
        wx = t["weather"][["date", "store_id"] + WEATHER_FEATURES]
        df = df.merge(wx, on=["store_id", "date"], how="left")
        features += WEATHER_FEATURES

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    return df, features, list(CATEGORICAL_FEATURES)
