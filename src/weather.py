"""
Météo des villes-magasins.

Décision validée le 2026-07-11 : ce qui compte est la COHÉRENCE du signal
(la demande générée doit réellement dépendre de la météo pour que le
scénario 2 « baseline + IA » ait quelque chose à apprendre). On privilégie
donc la vraie météo historique Open-Meteo (récupérée UNE fois, puis figée
dans data/referentiel/weather.csv — aucune dépendance externe ensuite),
avec un repli 100 % synthétique calibré par latitude si le réseau manque.

Trois sources possibles, tracées dans la colonne `source` :
  - "open-meteo"    : archive réelle (historique).
  - "synthetique"   : repli simulé (sinusoïde saisonnière + bruit autocorrélé).
  - "climatologie"  : au-delà du dernier jour réalisé, moyenne jour-de-l'année
                      de l'historique (= "prévision météo saisonnière" pour
                      les features futures du scénario 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_city_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Récupère température moyenne et précipitations journalières d'une ville."""
    import requests

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "Europe/Paris",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temp_mean_c": np.asarray(d["temperature_2m_mean"], dtype=float),
        "rain_mm": np.asarray(d["precipitation_sum"], dtype=float),
    })


def _synthetic_city(lat: float, lon: float, dates: pd.DatetimeIndex,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Repli synthétique : climat plausible piloté par la latitude (nord = plus froid)."""
    doy = dates.dayofyear.values.astype(float)
    # Moyenne annuelle ~17°C à Marseille (43°N), ~12°C à Paris (49°N)
    t_mean = 17.0 - 0.85 * (lat - 43.3)
    amplitude = 8.0 + 0.3 * (lat - 43.3)          # continentalité grossière
    season = t_mean + amplitude * np.sin(2 * np.pi * (doy - 105) / 365.25)
    # Bruit autocorrélé (AR(1)) pour des épisodes chauds/froids de quelques jours
    eps = rng.normal(0, 1.9, len(dates))
    noise = np.zeros(len(dates))
    for i in range(1, len(dates)):
        noise[i] = 0.72 * noise[i - 1] + eps[i]
    temp = season + noise
    # Pluie : jours de pluie ~35 %, quantités exponentielles, hiver plus arrosé
    p_rain = 0.30 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    is_rain = rng.random(len(dates)) < p_rain
    rain = np.where(is_rain, rng.exponential(5.0, len(dates)), 0.0)
    return pd.DataFrame({"date": dates, "temp_mean_c": temp.round(1), "rain_mm": rain.round(1)})


def build_weather(rng: np.random.Generator) -> pd.DataFrame:
    """
    Construit la table météo par magasin × jour, de HIST_START à FORECAST_END.

    - Magasins physiques : météo de leur ville (Open-Meteo, repli synthétique).
    - ONLINE : moyenne nationale pondérée par la population des villes.
    - Au-delà du dernier jour disponible : climatologie jour-de-l'année.
    """
    hist_dates = pd.date_range(config.HIST_START, config.HIST_END, freq="D")
    all_dates = pd.date_range(config.HIST_START, config.FORECAST_END, freq="D")

    frames = []
    use_fallback = False
    for store_id, city, _reg, _typ, _surf, _pop, lat, lon in config.STORES:
        df = None
        if not use_fallback:
            try:
                df = _fetch_city_open_meteo(lat, lon, config.HIST_START, config.HIST_END)
                print(f"  météo {city:<20s} : Open-Meteo ({len(df)} jours)")
            except Exception as e:  # réseau/API indisponible -> tout en synthétique
                print(f"  météo {city} : échec Open-Meteo ({e}) -> repli synthétique pour toutes les villes")
                use_fallback = True
        if df is None:
            df = _synthetic_city(lat, lon, hist_dates, rng)
        df["source"] = "synthetique" if use_fallback else "open-meteo"

        # L'archive peut s'arrêter quelques jours avant HIST_END : on complète
        # (et on étend jusqu'à FORECAST_END) par la climatologie jour-de-l'année.
        df = df.dropna(subset=["temp_mean_c"])
        clim = (df.assign(doy=df["date"].dt.dayofyear)
                  .groupby("doy")[["temp_mean_c", "rain_mm"]].mean())
        # lissage circulaire à 15 jours de la climatologie
        clim = pd.concat([clim.tail(15), clim, clim.head(15)]).rolling(15, center=True).mean().iloc[15:-15]
        missing = all_dates.difference(df["date"])
        if len(missing):
            fill = pd.DataFrame({
                "date": missing,
                "temp_mean_c": clim["temp_mean_c"].reindex(missing.dayofyear).values.round(1),
                "rain_mm": clim["rain_mm"].reindex(missing.dayofyear).values.round(1),
                "source": "climatologie",
            })
            df = pd.concat([df, fill], ignore_index=True).sort_values("date")
        df["store_id"] = store_id
        frames.append(df)

    wx = pd.concat(frames, ignore_index=True)

    # ONLINE = météo « France entière » : moyenne pondérée par population
    pops = {s[0]: s[5] for s in config.STORES}
    w = wx.assign(pop=wx["store_id"].map(pops))
    grp = w.groupby("date")
    online = pd.DataFrame({
        "temp_mean_c": grp.apply(lambda g: np.average(g["temp_mean_c"], weights=g["pop"])).round(1),
        "rain_mm": grp.apply(lambda g: np.average(g["rain_mm"], weights=g["pop"])).round(1),
        "source": grp["source"].first(),
    }).reset_index()
    online["store_id"] = config.ONLINE_STORE_ID

    out = pd.concat([wx.drop(columns=[c for c in ["doy"] if c in wx.columns]), online],
                    ignore_index=True)
    return out[["date", "store_id", "temp_mean_c", "rain_mm", "source"]].sort_values(
        ["store_id", "date"]).reset_index(drop=True)


def add_climatology_anomaly(wx: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute l'anomalie de température vs climatologie jour-de-l'année du magasin.

    C'est l'anomalie (et pas la température brute) qui pilote les effets météo
    du générateur ET les features du scénario 2 : le niveau saisonnier est déjà
    capté par les features calendaires, la valeur ajoutée météo est l'écart au
    normal (canicule, vague de froid).
    """
    wx = wx.copy()
    wx["doy"] = wx["date"].dt.dayofyear
    clim = (wx[wx["source"] != "climatologie"]
            .groupby(["store_id", "doy"])["temp_mean_c"].mean()
            .rename("temp_clim"))
    wx = wx.join(clim, on=["store_id", "doy"])
    # lissage grossier : rolling sur doy déjà moyenné multi-années, reste du bruit acceptable
    wx["temp_anomaly"] = (wx["temp_mean_c"] - wx["temp_clim"]).fillna(0.0).round(2)
    return wx.drop(columns=["doy", "temp_clim"])
