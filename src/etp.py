"""
Module ETP (Bloc C, version dégradée "ETP interne seul" — cadrage §5).

Calcule le besoin en ETP (Équivalent Temps Plein — un effectif, PAS une masse
salariale, cf. cadrage §10) par magasin, à partir des seules données internes :
  - CA prévisionnel par heure (CA journalier de la cascade × profil horaire
    historique du magasin),
  - fréquentation par heure (transactions prévues × profil horaire tickets),
  - horaires d'ouverture du magasin.

PAS de signal concurrentiel (aucune source identifiée — hors scope V1).
Le canal ONLINE est exclu (pas de magasin physique à staffer).

Besoin en personnel de l'heure h (normes 🟡 synthétiques, paramétrables) :

    staff(h) = max( CA(h) / prod_ca,  tickets(h) / cap_tickets,  effectif_min )

    ETP mensuel = Σ heures-personnes du mois / 151,67 h
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DEFAULT_PARAMS = dict(
    prod_ca_per_hour=config.ETP_PROD_CA_PER_HOUR,     # € de CA gérés par heure-vendeur
    tickets_per_hour=config.ETP_TICKETS_PER_HOUR,     # tickets encaissables par heure-vendeur
    min_staff=config.ETP_MIN_STAFF,                   # effectif plancher par heure ouvrée
    hours_per_month=config.ETP_HOURS_PER_MONTH,       # heures mensuelles d'un temps plein
)


def hourly_profiles(hourly_hist: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """
    Profils horaires par magasin × jour de semaine : part du CA et des tickets
    réalisée sur chaque heure ouvrée (moyenne des n_weeks dernières semaines).
    """
    cutoff = hourly_hist["date"].max() - pd.Timedelta(weeks=n_weeks)
    h = hourly_hist[hourly_hist["date"] > cutoff].copy()
    h["day_of_week"] = h["date"].dt.dayofweek
    prof = (h.groupby(["store_id", "day_of_week", "hour"], observed=True)
            .agg(ca=("ca", "mean"), tickets=("nb_tickets", "mean")).reset_index())
    tot = prof.groupby(["store_id", "day_of_week"], observed=True)[["ca", "tickets"]].transform("sum")
    prof["share_ca"] = np.where(tot.ca > 0, prof.ca / tot.ca, 0.0)
    prof["share_tickets"] = np.where(tot.tickets > 0, prof.tickets / tot.tickets, 0.0)
    return prof[["store_id", "day_of_week", "hour", "share_ca", "share_tickets"]]


def compute_etp(cascade_daily: pd.DataFrame, hourly_hist: pd.DataFrame,
                stores: pd.DataFrame, params: dict | None = None):
    """
    Args:
        cascade_daily : sortie de cascade_ca.cascade (ca_net, transactions par
                        magasin × jour prévisionnel).
        hourly_hist   : sales_hourly historique (profils).
        stores        : référentiel magasins (exclusion ONLINE).
        params        : surcharge des normes (dashboard).

    Returns:
        (etp_hourly, etp_daily, etp_monthly)
          etp_hourly  : magasin × date × heure — ca_h, tickets_h, staff besoin
          etp_daily   : magasin × date — heures-personnes nécessaires
          etp_monthly : magasin × mois — heures et ETP
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    physical = stores.loc[stores.is_online == 0, "store_id"]
    daily = cascade_daily[cascade_daily.store_id.isin(physical)].copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek

    prof = hourly_profiles(hourly_hist)
    h = daily.merge(prof, on=["store_id", "day_of_week"], how="left")
    h = h.dropna(subset=["hour"])  # jours fermés : pas de profil -> pas d'heures
    h["ca_h"] = h.ca_net * h.share_ca
    h["tickets_h"] = h.transactions * h.share_tickets
    h["staff"] = np.maximum.reduce([
        h.ca_h / p["prod_ca_per_hour"],
        h.tickets_h / p["tickets_per_hour"],
        np.full(len(h), float(p["min_staff"])),
    ])

    etp_hourly = h[["store_id", "date", "hour", "ca_h", "tickets_h", "staff"]]
    etp_daily = (etp_hourly.groupby(["store_id", "date"], observed=True)["staff"]
                 .sum().rename("heures_personnes").reset_index())
    etp_daily["month"] = etp_daily["date"].dt.to_period("M").astype(str)
    etp_monthly = (etp_daily.groupby(["store_id", "month"], observed=True)["heures_personnes"]
                   .sum().reset_index())
    etp_monthly["etp"] = (etp_monthly.heures_personnes / p["hours_per_month"]).round(2)
    return etp_hourly, etp_daily, etp_monthly
