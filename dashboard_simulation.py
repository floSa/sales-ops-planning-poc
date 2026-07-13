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
from src.cascade_ca import HYPOTHESIS_COLS, cascade as _cascade
from src.cascade_ca import cascade_monthly as _cascade_monthly
from src.cascade_ca import default_hypotheses
from src.ecarts import EFFECT_COLS
from src.etp import DEFAULT_PARAMS as ETP_DEFAULTS
from src.etp import compute_etp as _compute_etp
from src.simulation import apply_promo_to_forecast, propose_uplift, typology_uplift_reference

# --------------------------------------------------------------------------- #
# Charte AOSIS (reprise de Pilotage_StoreItem/dashboard_pilotage.py)
# --------------------------------------------------------------------------- #
NAVY, ORANGE, OK, WARN = "#211948", "#E84E24", "#809C30", "#F99500"
INK, INK_SOFT, MUTED, LINE, BG = "#1A1A1A", "#33404F", "#4B5563", "#E5E5E5", "#F2F2F7"
SHADOW = "0 2px 8px rgba(0,0,0,0.30)"

st.set_page_config(page_title="Prévision des ventes", layout="wide", page_icon="🐾")
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
  .expl {{ color:{INK_SOFT}; font-size:13px; line-height:1.55; margin:2px 0 14px; }}
  .intro {{ background:#fff; border-radius:12px; padding:16px 20px; box-shadow:{SHADOW};
            color:{INK_SOFT}; font-size:13px; line-height:1.65; margin-bottom:10px; }}
  .intro b {{ color:{INK}; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
      color:{INK_SOFT} !important; font-size:12.5px !important; }}

  /* Champs de saisie (select, multi-select, date, texte, nombre) : fond blanc,
     ressortent du fond gris de la page au lieu de s'y fondre. */
  div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
  div[data-testid="stDateInput"] input, div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div:has(> input),
  [data-testid="stMultiSelect"] div:has(> input) {{
      background-color: #fff !important; border: 1px solid {LINE} !important; }}

  /* Encart de saisie d'une campagne promo : carte blanche comme les KPI/graphes. */
  [data-testid="stForm"] {{
      background: #fff; border-radius: 12px; padding: 18px 20px 6px; box-shadow: {SHADOW}; }}

  /* Tableau d'hypothèses mensuelles : même traitement carte blanche. */
  [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {{
      background: #fff; border-radius: 12px; padding: 4px; box-shadow: {SHADOW}; }}

  /* Bouton "Appliquer la campagne" : gris neutre, pas la couleur d'accent orange. */
  [data-testid="stFormSubmitButton"] button {{
      background-color: {MUTED} !important; color: #fff !important; border: none !important; }}
  [data-testid="stFormSubmitButton"] button:hover {{
      background-color: {INK_SOFT} !important; }}
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


# Cache manuel léger (clé = id() des objets stables + signature bon marché des
# hypothèses/paramètres), PAS st.cache_data : st.cache_data hacherait le
# CONTENU complet des tables à chaque interaction (143 k lignes de prévision +
# 1,36 M lignes d'historique + 130 k lignes horaires) sur CHAQUE re-exécution
# du script, y compris quand rien de pertinent n'a changé — plus coûteux que
# le calcul lui-même. id() est une comparaison quasi gratuite, valide ici car
# TABLES/RES sont eux-mêmes des objets stables retournés par des fonctions
# @st.cache_data (même référence tant que le scénario ne change pas).
def _hyp_signature(hyp_df):
    if hyp_df is None:
        return None
    return tuple(map(tuple, hyp_df.itertuples(index=False)))


def cascade_cached(forecast_df, sales_df, traffic_df, hyp_df):
    cache = st.session_state.setdefault("_casc_cache", {})
    key = (id(forecast_df), id(sales_df), id(traffic_df), _hyp_signature(hyp_df))
    if key not in cache:
        cache[key] = _cascade(forecast_df, sales_df, traffic_df, hypotheses=hyp_df)
    return cache[key]


def cascade_monthly_cached(casc_df):
    cache = st.session_state.setdefault("_cascm_cache", {})
    key = id(casc_df)
    if key not in cache:
        cache[key] = _cascade_monthly(casc_df)
    return cache[key]


def compute_etp_cached(casc_df, hourly_df, stores_df, params_tuple):
    cache = st.session_state.setdefault("_etp_cache", {})
    key = (id(casc_df), id(hourly_df), id(stores_df), params_tuple)
    if key not in cache:
        cache[key] = _compute_etp(casc_df, hourly_df, stores_df, params=dict(params_tuple))
    return cache[key]


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
    <div class="hero-t">Prévision des ventes</div>
    <div class="hero-s">Profil Contrôleur de gestion (CDG) Ventes · prévision par magasin et par
    référence, au jour · vue mensuelle · atterrissage fin {YEAR}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="intro">
<b>Ce que fait l'outil.</b> À partir de 5 ans d'historique, il prévoit la demande future par
magasin et par produit, en déduit le chiffre d'affaires (par une cascade de calcul), le besoin
en personnel, et compare la prévision au réalisé. Deux scénarios sont proposés : une base
calendrier + promotions, ou cette base enrichie de signaux météo.<br>
<b>Périmètre.</b> 12 magasins physiques (villes françaises) + 1 canal Online ·
60 références produit réparties en 8 familles · historique quotidien sur 5 ans.<br>
<b>Sigles.</b> <b>SKU</b> = référence produit · <b>CA</b> = chiffre d'affaires ·
<b>PV</b> = prix de vente · <b>PA</b> = panier article (nombre d'articles par ticket) ·
<b>PM</b> = panier moyen (PV × PA) · <b>ETP</b> = équivalent temps plein ·
<b>WAPE</b> = erreur moyenne de prévision, en % (plus le chiffre est bas, meilleure est la prévision).
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. Scénario & périmètre
# --------------------------------------------------------------------------- #
c1, c2, _ = st.columns([1.6, 1.4, 3])
scenario = c1.radio("Scénario de prévision",
                    [1, 2], horizontal=True,
                    format_func=lambda s: "1 — Base (calendrier + promos)" if s == 1
                    else "2 — Base + météo")
sel_store = c2.selectbox("Magasin", ["Tous"] + STORES.store_id.tolist(),
                         format_func=lambda s: "Réseau entier" if s == "Tous"
                         else f"{s} — {STORES.set_index('store_id').store_name.get(s, s)}")
st.markdown('<div class="expl">Choisissez un scénario de prévision et, si besoin, un magasin '
            '(par défaut : tout le réseau). Le scénario 2 ajoute l\'effet de la météo à la base '
            'calendrier + promotions du scénario 1. Tous les écrans ci-dessous se recalculent '
            'selon ces deux choix.</div>', unsafe_allow_html=True)

RES = load_results(scenario)
if RES["forecast"] is None:
    st.error(f"Prévision du scénario {scenario} absente. Lancer :  "
             f"`python -m src.forecast --scenario {scenario}`")
    st.stop()

# --------------------------------------------------------------------------- #
# 2. Hypothèses mensuelles + simulation promo -> cascade
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Hypothèses du scénario — saisie par mois</div>',
            unsafe_allow_html=True)
st.markdown('<div class="expl">Le contrôleur de gestion ajuste ici les hypothèses mois par mois. '
            'Les coefficients multiplient les valeurs proposées par l\'outil : par exemple PV × 1,02 '
            'relève de 2 % le prix de vente moyen du mois. Le chiffre d\'affaires en découle par la '
            'cascade : PV × PA = PM (panier moyen), puis PM × nombre de transactions = CA. À gauche, '
            'les hypothèses ; à droite, une campagne promotionnelle à simuler.</div>',
            unsafe_allow_html=True)
months = sorted(RES["forecast"]["date"].dt.to_period("M").astype(str).unique())
if "hyp" not in st.session_state or list(st.session_state.hyp.month) != months:
    st.session_state.hyp = default_hypotheses(months)

hcol, pcol = st.columns([1.15, 1])
with hcol:
    st.caption("Coefficients multiplicateurs appliqués aux valeurs proposées par l'outil "
               "(1,00 = inchangé). « Δ inflation » = point d'inflation ajouté ou retiré par rapport "
               "à la trajectoire déjà intégrée aux prix 🟡.")
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

# cascade sur la prévision (éventuellement modifiée) — mise en cache (voir cascade_cached)
casc = cascade_cached(fc, TABLES["sales"], TABLES["traffic"], hyp)
casc_base = cascade_cached(RES["forecast"], TABLES["sales"], TABLES["traffic"], None)
if sel_store != "Tous":
    casc_view, casc_base_view = casc[casc.store_id == sel_store], casc_base[casc_base.store_id == sel_store]
else:
    casc_view, casc_base_view = casc, casc_base

# --------------------------------------------------------------------------- #
# 3. KPI + courbe CA net
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="sec">Prévision et chiffre d\'affaires — 2ᵉ semestre {YEAR} '
            f'({ "réseau entier" if sel_store == "Tous" else sel_store })</div>',
            unsafe_allow_html=True)
st.markdown('<div class="expl">Chiffre d\'affaires net attendu sur le 2ᵉ semestre après application '
            'de vos hypothèses, comparé à la proposition de l\'outil (hypothèses neutres). '
            'Le WAPE indique la fiabilité de la prévision, mesurée sur le passé (plus bas = mieux ; '
            'la « naïve » est la prévision de référence à battre). Sur le graphe : barres = CA simulé '
            'mois par mois, ligne orange = proposition de l\'outil.</div>', unsafe_allow_html=True)
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
k3.markdown(f'<div class="kpi"><div class="kpi-l">Fiabilité prévision (WAPE)</div>'
            f'<div class="kpi-v">{wape_txt.split(" ")[0]}</div>'
            f'<div class="kpi-d" style="color:{MUTED}">{wape_txt}</div></div>',
            unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><div class="kpi-l">Transactions prévues S2</div>'
            f'<div class="kpi-v">{casc_view.transactions.sum()/1e3:,.0f} k</div>'
            f'<div class="kpi-d" style="color:{MUTED}">PM moyen '
            f'{casc_view.ca_net.sum()/max(casc_view.transactions.sum(),1):,.1f} €</div></div>',
            unsafe_allow_html=True)

mm = cascade_monthly_cached(casc_view)
mm_base = cascade_monthly_cached(casc_base_view)
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
st.markdown(f'<div class="sec">Atterrissage {YEAR} (rolling forecast)</div>', unsafe_allow_html=True)
st.markdown('<div class="expl">Projection de fin d\'année : on additionne le chiffre d\'affaires déjà '
            'réalisé sur janvier–juin (barres vertes) et la prévision de juillet–décembre '
            '(barres bleues). Le total est l\'« atterrissage » — l\'estimation de clôture de l\'année, '
            'réactualisée à mesure que les mois s\'écoulent.</div>', unsafe_allow_html=True)
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
st.markdown('<div class="sec">Besoin en personnel (ETP) par magasin</div>', unsafe_allow_html=True)
st.markdown('<div class="expl">Traduction de la prévision en effectif : à partir du CA et de la '
            'fréquentation prévus heure par heure et des horaires d\'ouverture, l\'outil estime le '
            'nombre d\'équivalents temps plein (ETP) nécessaires par magasin. Réglez les normes '
            'ci-dessous (marquées 🟡, à caler avec le métier). Le canal Online est exclu (pas de '
            'magasin physique à armer). Le second graphe montre une journée type (samedi) et sert '
            'à dimensionner les plannings heure par heure.</div>', unsafe_allow_html=True)
e1, e2, e3, e4 = st.columns(4)
p_ca = e1.number_input("€ CA / heure-vendeur 🟡", 100.0, 500.0, float(ETP_DEFAULTS["prod_ca_per_hour"]), 10.0)
p_tk = e2.number_input("Tickets / heure-vendeur 🟡", 5.0, 30.0, float(ETP_DEFAULTS["tickets_per_hour"]), 1.0)
p_min = e3.number_input("Effectif minimum / h 🟡", 1.0, 5.0, float(ETP_DEFAULTS["min_staff"]), 0.5)
p_hm = e4.number_input("Heures / mois / ETP", 100.0, 200.0, float(ETP_DEFAULTS["hours_per_month"]), 0.01)

_etp_params = (("prod_ca_per_hour", p_ca), ("tickets_per_hour", p_tk),
               ("min_staff", p_min), ("hours_per_month", p_hm))
etp_hourly, etp_daily, etp_monthly = compute_etp_cached(
    casc, TABLES["hourly"], STORES, _etp_params)
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
st.markdown('<div class="sec">Analyse des écarts réel vs prévision</div>', unsafe_allow_html=True)
st.markdown('<div class="expl">Sur la période de test, l\'écart entre le CA réel et le CA prévu est '
            'décomposé en effets : <b>prix</b> (vente plus ou moins chère que prévu), <b>volume</b> '
            '(plus ou moins d\'unités), <b>mix</b> (déformation du panier). Le graphe en cascade part '
            'du CA prévu et arrive au CA réel. Sous le graphe, l\'écart de volume est reventilé par '
            'cause : promotions, effet calendaire (jours fériés…) et le reste (tendance, météo, '
            'aléa).</div>', unsafe_allow_html=True)
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

# --------------------------------------------------------------------------- #
# 7. Explicabilité — ce qui pilote la prévision
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Ce qui pilote la prévision</div>', unsafe_allow_html=True)
_imp_path = config.RESULTS_DIR / "explain" / "feature_importance.csv"
_fam_path = config.RESULTS_DIR / "explain" / "by_family.csv"
if not _imp_path.exists():
    st.info("Explicabilité non calculée. Lancer :  `python -m src.explain --scenario 2`")
else:
    imp = pd.read_csv(_imp_path)
    fam = pd.read_csv(_fam_path)
    st.markdown('<div class="expl">Poids de chaque variable dans les décisions du modèle '
                '(part du « gain » : la réduction d\'erreur qu\'elle apporte). À gauche, les '
                '10 variables les plus influentes ; à droite, le total par famille. Lecture '
                'attendue pour ce type de modèle : l\'historique récent des ventes et la '
                'saisonnalité dominent, les promotions et la météo apportent un complément '
                'ciblé.</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1.5, 1])
    with g1:
        top = imp.head(10).iloc[::-1]
        ifig = go.Figure(go.Bar(x=top.importance_pct, y=top.libelle, orientation="h",
                                marker_color=NAVY, text=top.importance_pct.round(1),
                                textposition="outside"))
        ifig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_title="part du gain (%)")
        st.plotly_chart(ifig, use_container_width=True)
    with g2:
        ff = fam.iloc[::-1]
        ffig = go.Figure(go.Bar(x=ff.importance_pct, y=ff.famille, orientation="h",
                                marker_color=ORANGE, text=ff.importance_pct.round(0),
                                textposition="outside"))
        ffig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_title="part du gain (%)")
        st.plotly_chart(ffig, use_container_width=True)
    st.caption("Diagnostic calculé sur le scénario 2 (météo incluse), sur les 5 ans d'historique.")

st.markdown(f"""<br><div style="color:{INK_SOFT};font-size:12px">
<b>Méthode.</b> Backtest « origine glissante » : {config.BACKTEST_N_FOLDS} plis de
{config.BACKTEST_HORIZON_DAYS} jours, métrique WAPE + biais, comparés à une prévision naïve
(valeur de la semaine précédente). Hypothèses de travail 🟡 encore à caler avec le métier :
définitions EB/PA, normes de productivité ETP, projection du panier article, écart d'inflation.
Données illustratives (détail dans le README).</div>""",
            unsafe_allow_html=True)
