"""
Dashboard de simulation — POC Prévision des ventes MaxiZoo (Bloc B).

Version dégradée "Profil unique" (cadrage §4) : UN seul profil simulé,
le CDG (Contrôleur De Gestion) Ventes. Pas de moteur de droits, pas de
validation hiérarchique, pas de profils CDG Online / DR / DV.

⚠️ Toutes les données affichées sont SYNTHÉTIQUES (cf. data/README.md).

Structure (une page, charte AOSIS reprise de Pilotage_StoreItem) :
  1. Scénario & périmètre (baseline vs baseline + IA météo)
  2. Hypothèses mensuelles éditables (PV / PA / transactions / Δ inflation)
     -> cascade PV × PA -> PM + inflation × transactions -> CA net
  3. Calendrier promotionnel + saisie d'une campagne (uplift proposé, modifiable)
  4. Prévision CA & atterrissage fin d'année (rolling forecast)
  5. Besoin ETP par magasin (normes éditables)
  6. Analyse d'écarts réel vs prévision (waterfall price/volume/mix/promo/calendaire)

Lancement :  streamlit run dashboard_simulation.py
Pré-requis : python -m src.backtest --scenario N  et  python -m src.forecast --scenario N
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.cascade_ca import HYPOTHESIS_COLS, cascade, cascade_monthly, default_hypotheses
from src.ecarts import EFFECT_COLS
from src.etp import DEFAULT_PARAMS as ETP_DEFAULTS
from src.etp import compute_etp
from src.simulation import apply_promo_to_forecast, propose_uplift, typology_uplift_reference

# --------------------------------------------------------------------------- #
# Charte AOSIS (reprise de Pilotage_StoreItem/dashboard_pilotage.py)
# --------------------------------------------------------------------------- #
NAVY, ORANGE, OK, WARN = "#211948", "#E84E24", "#809C30", "#F99500"
INK, MUTED, LINE, BG = "#1A1A1A", "#4B5563", "#E5E5E5", "#F2F2F7"
SHADOW = "0 2px 8px rgba(0,0,0,0.30)"

st.set_page_config(page_title="AOSIS — POC MaxiZoo Prévision", layout="wide", page_icon="🐾")
st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: Arial, sans-serif; }}
  .stApp {{ background: {BG}; }}
  [data-testid="stToolbar"] {{ display: none; }}
  header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
  [data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 700 !important; font-size: 13px !important; }}
  h1,h2,h3 {{ color: {INK}; font-weight: 700; }}
  .hero {{ background:{NAVY}; border-radius:14px; padding:18px 26px; box-shadow:{SHADOW};
           display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }}
  .hero-t {{ color:#fff; font-size:21px; font-weight:700; }}
  .hero-s {{ color:rgba(255,255,255,.82); font-size:12.5px; margin-top:4px; }}
  .badge {{ background:{ORANGE}; color:#fff; font-size:11px; font-weight:700;
            padding:4px 10px; border-radius:20px; }}
  .sec {{ font-size:11px; font-weight:700; color:{MUTED}; letter-spacing:.1em; text-transform:uppercase;
          margin:26px 0 12px; padding-bottom:8px; border-bottom:2px solid {ORANGE}; }}
  .kpi {{ background:#fff; border-radius:12px; padding:14px 18px; box-shadow:{SHADOW}; }}
  .kpi-l {{ font-size:11.5px; color:{MUTED}; font-weight:700; text-transform:uppercase; }}
  .kpi-v {{ font-size:24px; color:{INK}; font-weight:700; margin-top:2px; }}
  .kpi-d {{ font-size:12px; margin-top:2px; }}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_static():
    from src.dataset import load_tables
    t = load_tables()
    ref = typology_uplift_reference(t["sales"], t["promos"], t["scope"])
    return t, ref


@st.cache_data
def load_results(scenario: int):
    r = {}
    bt = config.RESULTS_DIR / "backtest" / f"scenario{scenario}"
    fc = config.RESULTS_DIR / "forecast" / f"forecast_daily_scenario{scenario}.parquet"
    r["summary"] = pd.read_csv(bt / "summary.csv") if (bt / "summary.csv").exists() else None
    r["by_store"] = pd.read_csv(bt / "by_store.csv") if (bt / "by_store.csv").exists() else None
    r["forecast"] = pd.read_parquet(fc) if fc.exists() else None
    if r["forecast"] is not None:
        r["forecast"]["date"] = pd.to_datetime(r["forecast"]["date"])
    ec = config.RESULTS_DIR / "ecarts" / f"ecarts_scenario{scenario}.csv"
    r["ecarts"] = pd.read_csv(ec) if ec.exists() else None
    return r


TABLES, UPLIFT_REF = load_static()
STORES = TABLES["stores"]
YEAR = pd.Timestamp(config.HIST_END).year

st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-t">POC Prévision des ventes — MaxiZoo</div>
    <div class="hero-s">Profil : CDG Ventes (profil unique V1) · grain natif magasin × SKU × jour ·
    horizon mensuel glissant · atterrissage {YEAR}</div>
  </div>
  <div class="badge">DONNÉES 100 % SYNTHÉTIQUES</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. Scénario & périmètre
# --------------------------------------------------------------------------- #
c1, c2, _ = st.columns([1.4, 1.4, 3])
scenario = c1.radio("Scénario de prévision (brief 1.1)",
                    [1, 2], horizontal=True,
                    format_func=lambda s: "1 — Baseline (calendaire + promos)" if s == 1
                    else "2 — Baseline + IA (signaux météo)")
sel_store = c2.selectbox("Magasin", ["Tous"] + STORES.store_id.tolist(),
                         format_func=lambda s: "Réseau entier" if s == "Tous"
                         else f"{s} — {STORES.set_index('store_id').store_name.get(s, s)}")

RES = load_results(scenario)
if RES["forecast"] is None:
    st.error(f"Prévision du scénario {scenario} absente. Lancer :  "
             f"`python -m src.forecast --scenario {scenario}`")
    st.stop()

# --------------------------------------------------------------------------- #
# 2. Hypothèses mensuelles + simulation promo -> cascade
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Hypothèses du scénario (saisie à la maille mois — brief 1.1)</div>',
            unsafe_allow_html=True)
months = sorted(RES["forecast"]["date"].dt.to_period("M").astype(str).unique())
if "hyp" not in st.session_state or list(st.session_state.hyp.month) != months:
    st.session_state.hyp = default_hypotheses(months)

hcol, pcol = st.columns([1.15, 1])
with hcol:
    st.caption("Coefficients multiplicatifs sur les valeurs proposées par l'outil. "
               "Δ inflation = écart vs trajectoire CPI centrale déjà incluse dans les prix 🟡. "
               "L'inflation n'est PAS modifiable par les DR/DV dans le brief — profil unique ici.")
    hyp = st.data_editor(
        st.session_state.hyp, hide_index=True, use_container_width=True,
        column_config={
            "month": st.column_config.TextColumn("Mois", disabled=True),
            "coef_pv": st.column_config.NumberColumn("PV ×", min_value=0.5, max_value=1.5, step=0.01),
            "coef_pa": st.column_config.NumberColumn("PA ×", min_value=0.5, max_value=1.5, step=0.01),
            "coef_transactions": st.column_config.NumberColumn("Transactions ×", min_value=0.5,
                                                               max_value=1.5, step=0.01),
            "inflation_delta_pct": st.column_config.NumberColumn("Δ inflation %", min_value=-5.0,
                                                                 max_value=5.0, step=0.1),
        })

with pcol:
    st.caption("Saisie d'une campagne promotionnelle (brief 1.2) — l'outil propose un uplift "
               "issu de l'historique, modifiable avant application.")
    with st.form("promo_form"):
        f1, f2 = st.columns(2)
        p_name = f1.text_input("Nom de campagne", "Ma campagne test")
        p_type = f2.selectbox("Typologie", config.PROMO_TYPES)
        f3, f4, f5 = st.columns(3)
        p_start = f3.date_input("Début", pd.Timestamp(f"{YEAR}-10-01"),
                                min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                                max_value=pd.Timestamp(config.FORECAST_END))
        p_end = f4.date_input("Fin", pd.Timestamp(f"{YEAR}-10-14"),
                              min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                              max_value=pd.Timestamp(config.FORECAST_END))
        p_perim = f5.selectbox("Périmètre", ["omnicanal", "magasin", "online"])
        p_disc = st.slider("Profondeur de remise (typologie 'produits')", 0.0, 0.5, 0.20, 0.05)
        p_cat = st.multiselect("Catégories ciblées",
                               sorted(TABLES["products"].commodity_group.unique()),
                               default=["Chien"])
        proposed = propose_uplift(UPLIFT_REF, p_type, p_disc)
        p_uplift = st.slider(f"Uplift quantités (proposé par l'outil : ×{proposed})",
                             0.8, 3.0, float(proposed), 0.05)
        p_apply = st.form_submit_button("Appliquer la campagne au scénario")

fc = RES["forecast"]
if p_apply:
    skus = TABLES["products"].loc[
        TABLES["products"].commodity_group.isin(p_cat), "sku_id"].tolist()
    fc = apply_promo_to_forecast(fc, promo_type=p_type, date_start=p_start, date_end=p_end,
                                 perimeter=p_perim, sku_ids=skus, uplift=p_uplift,
                                 discount_rate=p_disc if p_type == "produits" else 0.0)
    st.session_state.fc_sim = fc
    st.session_state.promo_name = p_name
elif "fc_sim" in st.session_state:
    fc = st.session_state.fc_sim

# cascade sur la prévision (éventuellement modifiée)
casc = cascade(fc.rename(columns={}), TABLES["sales"], TABLES["traffic"], hypotheses=hyp)
casc_base = cascade(RES["forecast"], TABLES["sales"], TABLES["traffic"], hypotheses=None)
if sel_store != "Tous":
    casc_view, casc_base_view = casc[casc.store_id == sel_store], casc_base[casc_base.store_id == sel_store]
else:
    casc_view, casc_base_view = casc, casc_base

# --------------------------------------------------------------------------- #
# 3. KPI + courbe CA net
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="sec">Prévision & cascade — CA net S2 {YEAR} '
            f'({ "réseau" if sel_store == "Tous" else sel_store })</div>', unsafe_allow_html=True)
ca_scn, ca_ref = casc_view.ca_net.sum(), casc_base_view.ca_net.sum()
delta = ca_scn / ca_ref - 1 if ca_ref else 0
wape_txt = "—"
if RES["summary"] is not None:
    s = RES["summary"]
    wh = s.loc[s.modele == "hybride", "WAPE"].iloc[0]
    wn = s.loc[s.modele == "naive_saiso", "WAPE"].iloc[0]
    wape_txt = f"{wh:.1%} (naïve {wn:.1%})"

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><div class="kpi-l">CA net simulé S2 {YEAR}</div>'
            f'<div class="kpi-v">{ca_scn/1e6:,.2f} M€</div>'
            f'<div class="kpi-d" style="color:{OK if delta>=0 else ORANGE}">'
            f'{delta:+.1%} vs proposition outil</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><div class="kpi-l">Proposition outil (hypothèses neutres)</div>'
            f'<div class="kpi-v">{ca_ref/1e6:,.2f} M€</div>'
            f'<div class="kpi-d" style="color:{MUTED}">scénario {scenario}</div></div>',
            unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><div class="kpi-l">WAPE backtest (hybride)</div>'
            f'<div class="kpi-v">{wape_txt.split(" ")[0]}</div>'
            f'<div class="kpi-d" style="color:{MUTED}">{wape_txt}</div></div>',
            unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="kpi-l">Transactions prévues S2</div>'
            f'<div class="kpi-v">{casc_view.transactions.sum()/1e3:,.0f} k</div>'
            f'<div class="kpi-d" style="color:{MUTED}">PM moyen '
            f'{casc_view.ca_net.sum()/max(casc_view.transactions.sum(),1):,.1f} €</div></div>',
            unsafe_allow_html=True)

mm = cascade_monthly(casc_view)
mm_base = cascade_monthly(casc_base_view)
mfig = go.Figure()
mfig.add_bar(x=mm.groupby("month").ca_net.sum().index,
             y=mm.groupby("month").ca_net.sum().values, name="CA net simulé",
             marker_color=NAVY)
mfig.add_scatter(x=mm_base.groupby("month").ca_net.sum().index,
                 y=mm_base.groupby("month").ca_net.sum().values,
                 name="Proposition outil", mode="lines+markers", line=dict(color=ORANGE, width=3))
mfig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                   paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
st.plotly_chart(mfig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 4. Atterrissage fin d'année (rolling forecast)
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="sec">Rolling forecast — atterrissage {YEAR}</div>', unsafe_allow_html=True)
sales = TABLES["sales"]
reel = sales[sales.date.dt.year == YEAR]
if sel_store != "Tous":
    reel = reel[reel.store_id == sel_store]
reel_m = reel.assign(month=reel.date.dt.to_period("M").astype(str)).groupby("month").revenue.sum()
prev_m = mm.groupby("month").ca_net.sum()
att = pd.concat([reel_m.rename("CA"), prev_m.rename("CA")])
afig = go.Figure()
afig.add_bar(x=reel_m.index, y=reel_m.values, name="Réel (janv-juin)", marker_color=OK)
afig.add_bar(x=prev_m.index, y=prev_m.values, name="Prévision (juil-déc)", marker_color=NAVY)
afig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                   paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
st.plotly_chart(afig, use_container_width=True)
st.markdown(f"**Atterrissage {YEAR} : {(reel_m.sum() + prev_m.sum())/1e6:,.2f} M€** "
            f"(réel S1 {reel_m.sum()/1e6:,.2f} M€ + prévision S2 {prev_m.sum()/1e6:,.2f} M€)")

# --------------------------------------------------------------------------- #
# 5. Besoin ETP (Bloc C — interne seul)
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Besoin ETP par magasin (interne seul — sans signal concurrentiel, '
            'ONLINE exclu)</div>', unsafe_allow_html=True)
e1, e2, e3, e4 = st.columns(4)
p_ca = e1.number_input("€ CA / heure-vendeur 🟡", 100.0, 500.0, float(ETP_DEFAULTS["prod_ca_per_hour"]), 10.0)
p_tk = e2.number_input("Tickets / heure-vendeur 🟡", 5.0, 30.0, float(ETP_DEFAULTS["tickets_per_hour"]), 1.0)
p_min = e3.number_input("Effectif minimum / h 🟡", 1.0, 5.0, float(ETP_DEFAULTS["min_staff"]), 0.5)
p_hm = e4.number_input("Heures / mois / ETP", 100.0, 200.0, float(ETP_DEFAULTS["hours_per_month"]), 0.01)

etp_hourly, etp_daily, etp_monthly = compute_etp(
    casc, TABLES["hourly"], STORES,
    params=dict(prod_ca_per_hour=p_ca, tickets_per_hour=p_tk, min_staff=p_min, hours_per_month=p_hm))
etp_avg = etp_monthly.groupby("store_id").etp.mean().sort_values(ascending=False)
efig = go.Figure(go.Bar(x=etp_avg.index, y=etp_avg.values, marker_color=NAVY,
                        text=etp_avg.round(1), textposition="outside"))
efig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                   paper_bgcolor="rgba(0,0,0,0)",
                   yaxis_title=f"ETP moyen / mois (S2 {YEAR})")
st.plotly_chart(efig, use_container_width=True)

etp_store = sel_store if sel_store != "Tous" else etp_avg.index[0]
day_prof = (etp_hourly[etp_hourly.store_id == etp_store]
            .assign(dow=lambda d: pd.to_datetime(d.date).dt.dayofweek)
            .query("dow == 5").groupby("hour").staff.mean())
pfig = go.Figure(go.Scatter(x=day_prof.index, y=day_prof.values, fill="tozeroy",
                            line=dict(color=ORANGE, width=3)))
pfig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#fff",
                   paper_bgcolor="rgba(0,0,0,0)",
                   title=f"Courbe de staffing type — samedi, {etp_store}",
                   xaxis_title="heure", yaxis_title="personnes")
st.plotly_chart(pfig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 6. Analyse d'écarts (Bloc D)
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Écarts réel vs prévision — fenêtre de backtest '
            '(price / volume / mix / promo / calendaire)</div>', unsafe_allow_html=True)
if RES["ecarts"] is None:
    st.info(f"Décomposition absente. Lancer :  `python -m src.ecarts --scenario {scenario}`")
else:
    ec = RES["ecarts"]
    if sel_store != "Tous":
        ec = ec[ec.store_id == sel_store]
    agg = ec[["ca_prev", "effet_prix", "effet_volume", "effet_mix"]].sum()
    drivers = ec[["dont_promo", "dont_calendaire", "dont_autre"]].sum()
    wfig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["CA prévu", "Effet prix", "Effet volume", "Effet mix", "CA réel"],
        y=[agg.ca_prev, agg.effet_prix, agg.effet_volume, agg.effet_mix, 0],
        connector=dict(line=dict(color=LINE)),
        increasing=dict(marker=dict(color=OK)), decreasing=dict(marker=dict(color=ORANGE)),
        totals=dict(marker=dict(color=NAVY))))
    wfig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(wfig, use_container_width=True)
    d1, d2, d3 = st.columns(3)
    for col, (lab, v) in zip((d1, d2, d3),
                             [("dont effet promo", drivers.dont_promo),
                              ("dont effet calendaire", drivers.dont_calendaire),
                              ("dont autre (tendance, météo, aléa)", drivers.dont_autre)]):
        col.markdown(f'<div class="kpi"><div class="kpi-l">{lab}</div>'
                     f'<div class="kpi-v" style="color:{OK if v>=0 else ORANGE}">{v:+,.0f} €</div>'
                     f'<div class="kpi-d" style="color:{MUTED}">reventilation de l\'effet quantité</div>'
                     f'</div>', unsafe_allow_html=True)

st.markdown(f"""<br><div style="color:{MUTED};font-size:12px">
POC AOSIS Consulting — données synthétiques, chiffres sans valeur métier.
Hypothèses 🟡 à valider client : définitions EB/PA, normes ETP, PA projeté sans modèle dédié,
Δ inflation vs CPI central. Backtest : rolling-origin {config.BACKTEST_N_FOLDS} plis ×
{config.BACKTEST_HORIZON_DAYS} j, WAPE + biais vs naïve saisonnière lag-7.</div>""",
            unsafe_allow_html=True)
