"""
Dashboard de simulation — POC Prévision des ventes (Bloc B).

Application Streamlit MULTI-PAGES (st.navigation). Profil unique CDG Ventes
(cadrage §4). Toutes les données affichées sont SYNTHÉTIQUES (cf. data/README.md).

Barre latérale (persistante, s'applique à toutes les pages) : scénario + magasin.
Pages :
  Guide                      — à quoi sert l'outil, sur quoi reposent les données
  Combien va-t-on vendre ?   — prévision & chiffre d'affaires (+ hypothèses / promo)
  Comment finira l'année ?   — atterrissage (rolling forecast)
  Besoin en personnel        — ETP par magasin
  Écarts prévu / réel        — décomposition price/volume/mix/promo/calendaire
  Ce qui pilote la prévision — explicabilité (importance des variables)

Lancement :  streamlit run dashboard_simulation.py
Pré-requis : python main.py  (backtest + prévision + écarts + explicabilité)
"""
# --------------------------------------------------------------------------- #
# STABILITÉ — NE PAS RETIRER, et garder AVANT l'import de pandas/pyarrow.
#
# Symptôme : le serveur Streamlit mourait sans aucune trace dès qu'on changeait
# de page ; seule la 1ʳᵉ page (Guide) restait affichable. Ce n'était pas un
# plantage applicatif mais un SEGFAULT du processus.
#
# Diagnostic (`python -X faulthandler`) : toutes les piles pointaient pyarrow
# (lecture parquet, tableaux de chaînes). Cause : l'allocateur mémoire par
# défaut de pyarrow (jemalloc) est instable dans cet environnement (WSL + venv
# avec libgomp préchargé en RTLD_GLOBAL). Bascule sur l'allocateur système :
# 0 segfault sur toutes les pages, testé en boucle.
#
# La variable est lue par pyarrow À SON IMPORT : elle doit donc être posée
# avant. setdefault() laisse la main à un réglage externe éventuel.
import os

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.cascade_ca import cascade as _cascade
from src.cascade_ca import cascade_monthly as _cascade_monthly
from src.cascade_ca import default_hypotheses
from src.etp import DEFAULT_PARAMS as ETP_DEFAULTS
from src.etp import compute_etp as _compute_etp
from src import prediction_intervals as pi
from src.simulation import apply_promo_to_forecast, propose_uplift, typology_uplift_reference

# --------------------------------------------------------------------------- #
# Charte graphique
# --------------------------------------------------------------------------- #
NAVY, ORANGE, OK, WARN = "#211948", "#E84E24", "#809C30", "#F99500"
INK, INK_SOFT, MUTED, LINE, BG = "#1A1A1A", "#33404F", "#4B5563", "#E5E5E5", "#F2F2F7"
SHADOW = "0 2px 8px rgba(0,0,0,0.30)"
SB_TEXT = "#FFFFFF"          # texte sur fond navy (sidebar)
SB_SOFT = "#C7C9DB"          # texte secondaire sidebar — lisible sur navy (contraste > 7)
# Variantes assombries POUR LE TEXTE : les couleurs de marque sont trop claires
# pour du petit texte sur fond clair (vert 3,1 et orange 3,4 -> sous le seuil
# WCAG AA de 4,5, mesuré au navigateur). Les teintes vives restent utilisées
# pour les aplats de graphiques, où le contraste ne se juge pas de la même façon.
OK_TEXT = "#5F7524"          # vert lisible : 5,2 sur blanc
ORANGE_TEXT = "#C0391A"      # orange lisible : 4,9 sur gris clair · 5,5 sur blanc


# --------------------------------------------------------------------------- #
# Formatage FRANÇAIS des nombres
# --------------------------------------------------------------------------- #
# Python formate en anglais ("193,934.5") : dans une app française, "193,934 €"
# se lit "193 virgule 934" -> contresens complet. On impose donc partout :
# espace fine insécable pour les milliers, virgule pour les décimales, et une
# unité TOUJOURS explicite (€ / k€ / M€).
NBSP = " "   # espace fine insécable (séparateur de milliers)


def fr(x, dec=0) -> str:
    """Nombre au format français : 193 934 · 5,33 · 1 234,5"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{x:,.{dec}f}"                      # 193,934.50 (anglais)
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", NBSP)
    return s


def eur(x, dec=0) -> str:
    """Montant en euros, unité explicite : 934 € · 12,3 k€ · 5,33 M€"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    a = abs(x)
    if a >= 1e6:
        return f"{fr(x / 1e6, 2)}{NBSP}M€"
    if a >= 1e3:
        return f"{fr(x / 1e3, 1)}{NBSP}k€"
    return f"{fr(x, dec)}{NBSP}€"


def pct(x, dec=1, signe=False) -> str:
    """Pourcentage français : 25,9 % · +2,7 %"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{fr(x * 100, dec)}{NBSP}%"
    return f"+{s}" if signe and x >= 0 else s


# Axes Plotly : mêmes conventions (séparateur milliers = espace fine, virgule décimale)
PLOTLY_FR = dict(separators=", ")


MOIS_FR = {1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
           7: "juil.", 8: "août", 9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."}


def mois_fr(periode) -> str:
    """'2026-07' -> 'juil. 2026'. Sans ça, Plotly interprète la chaîne comme une
    date et affiche des mois ANGLAIS ('Jul 2026') dans une interface française."""
    p = pd.Period(str(periode), freq="M")
    return f"{MOIS_FR[p.month]} {p.year}"

st.set_page_config(page_title="Prévision des ventes", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: Arial, sans-serif; }}
  .stApp {{ background: {BG}; }}
  [data-testid="stToolbar"] {{ display: none; }}
  /* Bouton pour RÉOUVRIR la sidebar quand elle est repliée (stExpandSidebarButton) :
     son icône est en gris clair (fadedText60) posée sur notre fond clair -> quasi
     invisible, la sidebar semblait impossible à récupérer. On en fait une pastille
     navy bien visible, épinglée en haut à gauche. */
  [data-testid="stExpandSidebarButton"] {{
      display: flex !important; visibility: visible !important; opacity: 1 !important;
      z-index: 1000 !important; background: {NAVY} !important; border-radius: 8px !important;
      box-shadow: {SHADOW} !important; }}
  [data-testid="stExpandSidebarButton"]:hover {{ background: {INK_SOFT} !important; }}
  [data-testid="stExpandSidebarButton"] *,
  [data-testid="stExpandSidebarButton"] svg {{ color: #fff !important; fill: #fff !important; }}
  /* Filet de sécurité : Streamlit mémorise l'état "replié" dans le localStorage du
     navigateur -> la sidebar (qui porte TOUTE la navigation) pouvait rester
     introuvable au rechargement, bloquant l'utilisateur sur la 1re page. On la
     force visible : la navigation ne peut plus être perdue. */
  [data-testid="stSidebar"] {{
      display: block !important; visibility: visible !important;
      transform: none !important; margin-left: 0 !important;
      min-width: 244px !important; width: 244px !important; }}
  [data-testid="stSidebar"][aria-expanded="false"] {{
      display: block !important; transform: none !important; }}
  [data-testid="stSidebarContent"] {{ visibility: visible !important; opacity: 1 !important; }}
  [data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 700 !important; font-size: 13px !important; }}
  h1,h2,h3 {{ color: {INK}; font-weight: 700; }}
  [data-testid="stSidebar"] {{ background: {NAVY}; }}
  /* --- Texte de la sidebar ---------------------------------------------------
     ATTENTION : ne JAMAIS remettre une règle « [data-testid="stSidebar"] * ».
     C'était la cause racine du bug « blanc sur blanc » : le sélecteur * gagne
     sur l'héritage et repeignait en blanc le texte des champs (dont le fond est
     blanc), rendant la valeur choisie invisible. On passe par l'HÉRITAGE (le
     conteneur donne la couleur, les enfants peuvent la redéfinir librement). */
  [data-testid="stSidebar"] {{ color: {SB_TEXT}; }}
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ color: {SB_TEXT}; }}
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: {SB_TEXT} !important; }}
  /* Captions de la sidebar : la règle globale les mettait en INK_SOFT (#33404F)
     sur navy -> contraste 1,53, illisible. Ici : gris clair lisible (> 7). */
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] em,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong {{
      color: {SB_SOFT} !important; }}
  /* Item de navigation actif : fond plus contrasté que le rgba(…,.25) par défaut */
  [data-testid="stSidebarNavLink"][aria-current="page"],
  [data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.16) !important; }}
  .ptitle {{ font-size:22px; font-weight:700; color:{INK}; margin:2px 0 2px; }}
  .psub {{ font-size:13px; color:{INK_SOFT}; margin-bottom:4px; }}
  .rule {{ height:3px; background:{ORANGE}; border-radius:2px; width:60px; margin:6px 0 16px; }}
  .hero {{ background:{NAVY}; border-radius:14px; padding:20px 26px; box-shadow:{SHADOW};
           margin-bottom:14px; }}
  .hero-t {{ color:#fff; font-size:23px; font-weight:700; }}
  .hero-s {{ color:rgba(255,255,255,.82); font-size:13px; margin-top:4px; }}
  .sec {{ font-size:11px; font-weight:700; color:{MUTED}; letter-spacing:.1em; text-transform:uppercase;
          margin:22px 0 12px; padding-bottom:8px; border-bottom:2px solid {ORANGE}; }}
  .kpi {{ background:#fff; border-radius:12px; padding:14px 18px; box-shadow:{SHADOW}; }}
  .kpi-l {{ font-size:11.5px; color:{MUTED}; font-weight:700; text-transform:uppercase; }}
  .kpi-v {{ font-size:24px; color:{INK}; font-weight:700; margin-top:2px; }}
  .kpi-d {{ font-size:12px; margin-top:2px; }}
  .expl {{ background:#fff; border-radius:12px; padding:14px 18px; box-shadow:{SHADOW};
           color:{INK_SOFT}; font-size:13px; line-height:1.6; margin:2px 0 16px; }}
  .expl b {{ color:{INK}; }}
  .intro {{ background:#fff; border-radius:12px; padding:16px 20px; box-shadow:{SHADOW};
            color:{INK_SOFT}; font-size:13px; line-height:1.65; margin-bottom:12px; }}
  .intro b {{ color:{INK}; }}
  .intro ol {{ margin:8px 0 10px; padding-left:22px; }}
  .intro li {{ margin-bottom:7px; }}
  .intro .lead {{ font-size:14px; color:{INK}; font-weight:700; margin-bottom:6px; }}
  .intro .fine {{ font-size:12px; color:{MUTED}; margin-top:8px;
                  border-top:1px solid {LINE}; padding-top:8px; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
      color:{INK_SOFT} !important; font-size:12.5px !important; }}
  /* --- Champs de saisie -------------------------------------------------------
     Structure réelle (vérifiée en inspectant le DOM ; cette version de Streamlit
     n'expose AUCUN data-baseweb, les anciennes règles ne matchaient rien) :
       [data-testid="stSelectbox"]
         ├── <label data-testid="stWidgetLabel">   <- le LIBELLÉ (fond navy)
         └── .react-aria-ComboBox > [role=group] > input[role=combobox]  <- le CHAMP
     On ne peint QUE le champ en blanc : peindre tout le conteneur mettait aussi
     le libellé sur fond blanc (donc blanc sur blanc). */
  [data-testid="stSelectbox"] [role="group"],
  [data-testid="stSelectbox"] input[role="combobox"],
  [data-testid="stMultiSelect"] [role="group"],
  [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
  [data-testid="stDateInput"] input {{ background-color: #fff !important; }}
  [data-testid="stSelectbox"] [role="group"],
  [data-testid="stMultiSelect"] [role="group"],
  [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
  [data-testid="stDateInput"] input {{ border: 1px solid {LINE} !important; }}
  /* Texte du champ : TOUJOURS sombre (le fond est blanc), y compris dans la
     sidebar navy -> c'est LE correctif du « magasin invisible ». */
  [data-testid="stSelectbox"] input[role="combobox"],
  [data-testid="stMultiSelect"] input,
  [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
  [data-testid="stDateInput"] input {{ color: {INK} !important; }}
  /* Chevron du selectbox (svg fill="currentColor") */
  [data-testid="stSelectbox"] button[aria-haspopup="listbox"],
  [data-testid="stMultiSelect"] button {{ color: {INK} !important; background: transparent !important; }}
  /* Menu déroulant : rendu dans un portail HORS de la sidebar -> règles globales */
  [role="listbox"], [role="listbox"] *, [role="option"], [role="option"] * {{
      color: {INK} !important; }}
  [role="listbox"] {{ background-color: #fff !important; }}
  [role="option"][aria-selected="true"], [role="option"]:hover {{
      background-color: {BG} !important; }}
  /* Icône « replier la sidebar » : sur navy -> blanche (elle héritait du sombre
     depuis la suppression de la règle globale). */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapseButton"] * {{ color: {SB_TEXT} !important; }}
  /* Sliders : la valeur courante s'affiche dans la couleur primaire (orange)
     JUSTE AU-DESSUS du curseur, lui aussi orange -> texte invisible (contraste
     mesuré : 1,0). On la passe en sombre, et les bornes min/max en gris lisible. */
  /* La valeur se superpose au curseur (orange) : on lui donne sa propre pastille
     blanche -> texte sombre lisible quel que soit ce qu'il y a dessous. */
  [data-testid="stSliderThumbValue"] {{
      color: {INK} !important; font-weight: 700 !important;
      background: #fff !important; padding: 0 5px !important; border-radius: 4px !important;
      border: 1px solid {LINE} !important; }}
  [data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] * {{
      color: {MUTED} !important; }}
  [data-testid="stForm"] {{
      background: #fff; border-radius: 12px; padding: 18px 20px 6px; box-shadow: {SHADOW}; }}
  [data-testid="stExpander"] {{
      background: #fff; border-radius: 12px; box-shadow: {SHADOW}; border: none; }}
  [data-testid="stExpander"] summary {{ border-radius: 12px; }}
  [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]) {{
      background: #fff; border-radius: 12px; padding: 4px; box-shadow: {SHADOW}; }}
  [data-testid="stFormSubmitButton"] button {{
      background-color: {MUTED} !important; color: #fff !important; border: none !important; }}
  [data-testid="stFormSubmitButton"] button:hover {{ background-color: {INK_SOFT} !important; }}
  /* --- Narration pédagogique (page Guide) --- */
  .story {{ background:#fff; border-radius:12px; padding:18px 24px; box-shadow:{SHADOW};
            color:{INK_SOFT}; font-size:13.5px; line-height:1.7; margin-bottom:14px; }}
  .story b {{ color:{INK}; }}
  .story .lead {{ font-size:15px; color:{INK}; font-weight:700; margin:2px 0 8px; }}
  .story p {{ margin:0 0 10px; }}
  .journey {{ display:flex; flex-direction:column; gap:10px; margin:8px 0 4px; }}
  .qstep {{ display:flex; gap:14px; align-items:flex-start; background:{BG}; border-radius:10px;
            padding:13px 16px; border-left:4px solid {ORANGE}; }}
  .qstep .num {{ flex:0 0 auto; width:27px; height:27px; border-radius:50%; background:{NAVY};
                 color:#fff; font-weight:700; font-size:13px; display:flex; align-items:center;
                 justify-content:center; margin-top:1px; }}
  .qstep .body {{ font-size:13px; color:{INK_SOFT}; line-height:1.6; }}
  .qstep .q {{ font-weight:700; color:{INK}; font-size:14.5px; }}
  .qstep .to {{ color:{ORANGE_TEXT}; font-weight:700; }}
  .method {{ display:flex; gap:12px; margin:8px 0 2px; flex-wrap:wrap; }}
  .mcard {{ flex:1 1 200px; background:{BG}; border-radius:10px; padding:14px 16px; }}
  .mcard .mn {{ color:{ORANGE_TEXT}; font-weight:700; font-size:11px; letter-spacing:.08em;
                text-transform:uppercase; }}
  .mcard .mt {{ font-weight:700; color:{INK}; font-size:13.5px; margin:3px 0 4px; }}
  .mcard .md {{ font-size:12.5px; color:{INK_SOFT}; line-height:1.6; }}
  /* --- Bandeau narratif compact (pages analytiques) --- */
  .narr {{ background:#fff; border-radius:12px; padding:4px 20px; box-shadow:{SHADOW};
           margin:2px 0 16px; border-left:4px solid {ORANGE}; }}
  .narr-row {{ display:flex; gap:14px; padding:11px 0; border-bottom:1px solid {LINE}; }}
  .narr-row:last-child {{ border-bottom:none; }}
  .narr-k {{ flex:0 0 118px; font-size:10.5px; font-weight:700; color:{MUTED};
             text-transform:uppercase; letter-spacing:.07em; padding-top:2px; }}
  .narr-v {{ flex:1; font-size:13px; color:{INK_SOFT}; line-height:1.6; }}
  .narr-v b {{ color:{INK}; }}
  .narr-good {{ flex:0 0 118px; font-size:10.5px; font-weight:700; color:{OK_TEXT};
               text-transform:uppercase; letter-spacing:.07em; padding-top:2px; }}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
YEAR = pd.Timestamp(config.HIST_END).year


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
# @st.cache_resource et NON @st.cache_data pour toutes les données lourdes :
# cache_data sérialise/désérialise les DataFrame via Arrow à chaque rerun. Avec
# pandas 3 + pyarrow 25, les objets restitués provoquaient des SEGFAULT (crash
# du serveur, sans trace, dès qu'on changeait de page). cache_resource garde
# l'objet Python tel quel — aucune conversion Arrow. Ces tables sont lues et
# jamais modifiées, ce qui est exactement l'usage prévu de cache_resource.
@st.cache_resource
def load_static():
    from src.dataset import load_tables
    t = load_tables()
    ref = typology_uplift_reference(t["sales"], t["promos"], t["scope"])
    return t, ref


@st.cache_resource
def load_pi_factors(scenario: int):
    """Facteurs de fourchette P10/P90 calibrés sur le backtest (ou None)."""
    return pi.load_factors(scenario)


@st.cache_resource
def load_results(scenario: int):
    r = {}
    bt = config.RESULTS_DIR / "backtest" / f"scenario{scenario}"
    fc = config.RESULTS_DIR / "forecast" / f"forecast_daily_scenario{scenario}.parquet"
    r["summary"] = pd.read_csv(bt / "summary.csv") if (bt / "summary.csv").exists() else None
    r["forecast"] = pd.read_parquet(fc) if fc.exists() else None
    if r["forecast"] is not None:
        r["forecast"]["date"] = pd.to_datetime(r["forecast"]["date"])
    ec = config.RESULTS_DIR / "ecarts" / f"ecarts_scenario{scenario}.csv"
    r["ecarts"] = pd.read_csv(ec) if ec.exists() else None
    return r


# Cache manuel léger (clé = id() des objets stables + signature bon marché des
# hypothèses) : évite de recalculer la cascade (143 k lignes) et l'ETP (jointure
# sur 130 k lignes) à chaque interaction, sans hacher tout le contenu comme le
# ferait st.cache_data.
def _hyp_signature(hyp_df):
    return None if hyp_df is None else tuple(map(tuple, hyp_df.itertuples(index=False)))


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


TABLES, UPLIFT_REF = load_static()
STORES = TABLES["stores"]
# Simple dict (et non STORES.set_index(...).store_name) : avec pandas 3 les
# colonnes texte sont adossées à Arrow, et un set_index sur un objet restitué
# par le cache Streamlit provoquait un SEGFAULT de pyarrow — le serveur mourait
# à chaque changement de page (seul le Guide, qui n'y touche pas, s'affichait).
# .tolist() ramène des objets Python natifs : plus aucune opération Arrow ici.
STORE_NAME = dict(zip(STORES["store_id"].tolist(), STORES["store_name"].tolist()))


# --------------------------------------------------------------------------- #
# Contexte partagé entre pages (scénario + magasin + hypothèses + promo simulée)
# --------------------------------------------------------------------------- #
def store_label(s):
    return "Réseau entier" if s == "Tous" else f"{s} — {STORE_NAME.get(s, s)}"


def get_context(require_forecast=True):
    """Charge les résultats du scénario courant et calcule la cascade CA en
    tenant compte des hypothèses et de la campagne promo éventuellement saisies
    (page « Combien va-t-on vendre ? »). Renvoie un dict, ou None si la prévision
    n'a pas encore été générée."""
    scenario = st.session_state.get("scenario", 1)
    sel_store = st.session_state.get("sel_store", "Tous")
    RES = load_results(scenario)
    if require_forecast and RES["forecast"] is None:
        return {"scenario": scenario, "sel_store": sel_store, "RES": RES, "missing": True}

    ctx = {"scenario": scenario, "sel_store": sel_store, "RES": RES, "missing": False}
    if RES["forecast"] is None:
        return ctx

    months = sorted(RES["forecast"]["date"].dt.to_period("M").astype(str).unique())
    if "hyp" not in st.session_state or list(st.session_state.hyp.month) != months:
        st.session_state.hyp = default_hypotheses(months)
    hyp = st.session_state.hyp

    # campagne promo : utilisée seulement si elle a été construite pour CE scénario
    fc = RES["forecast"]
    if st.session_state.get("fc_sim") is not None and st.session_state.get("fc_sim_scenario") == scenario:
        fc = st.session_state.fc_sim

    casc = cascade_cached(fc, TABLES["sales"], TABLES["traffic"], hyp)
    casc_base = cascade_cached(RES["forecast"], TABLES["sales"], TABLES["traffic"], None)
    if sel_store != "Tous":
        casc_view = casc[casc.store_id == sel_store]
        casc_base_view = casc_base[casc_base.store_id == sel_store]
    else:
        casc_view, casc_base_view = casc, casc_base

    ctx.update(hyp=hyp, casc=casc, casc_view=casc_view, casc_base_view=casc_base_view,
               months=months)
    return ctx


def page_header(title, subtitle):
    st.markdown(f'<div class="ptitle">{title}</div>'
                f'<div class="psub">{subtitle}</div><div class="rule"></div>',
                unsafe_allow_html=True)


def story_block(question, method, read, good):
    """Bandeau narratif homogène en tête de page analytique : rappelle la question
    métier, la méthode, la clé de lecture et ce qu'un bon résultat montre — pour
    qu'un lecteur non initié sache toujours quoi regarder."""
    st.markdown(
        f'<div class="narr">'
        f'<div class="narr-row"><span class="narr-k">La question</span>'
        f'<span class="narr-v">{question}</span></div>'
        f'<div class="narr-row"><span class="narr-k">Notre méthode</span>'
        f'<span class="narr-v">{method}</span></div>'
        f'<div class="narr-row"><span class="narr-k">Comment lire</span>'
        f'<span class="narr-v">{read}</span></div>'
        f'<div class="narr-row"><span class="narr-good">Bon signe</span>'
        f'<span class="narr-v">{good}</span></div>'
        f'</div>', unsafe_allow_html=True)


def forecast_missing_stop(ctx):
    st.error(f"La prévision du scénario {ctx['scenario']} n'a pas encore été générée. "
             f"Lancer dans un terminal :  `python -m src.forecast --scenario {ctx['scenario']}` "
             f"(ou `python main.py` pour tout produire).")
    st.stop()


def ca_band_by_month(casc_df, factors):
    """Fourchette P10/P90 du CA mensuel, calibrée sur le backtest (ou None si
    les facteurs ne sont pas disponibles)."""
    if factors is None or casc_df is None or len(casc_df) == 0:
        return None
    sd = pi.store_day_bounds(casc_df, factors, value="ca_net")
    rows = [(m, *pi.aggregate_band(g)) for m, g in sd.groupby("month")]
    return pd.DataFrame(rows, columns=["month", "point", "lo", "hi"])


def ca_band_total(casc_df, factors):
    """Fourchette P10/P90 du CA total prévu (ou None)."""
    if factors is None or casc_df is None or len(casc_df) == 0:
        return None
    return pi.aggregate_band(pi.store_day_bounds(casc_df, factors, value="ca_net"))


# --------------------------------------------------------------------------- #
# PAGE — Guide
# --------------------------------------------------------------------------- #
def page_guide():
    st.markdown(f"""
    <div class="hero">
      <div class="hero-t">Prévision des ventes — le fil de l'analyse</div>
      <div class="hero-s">Démonstrateur d'aide au pilotage commercial · profil Contrôleur de
      gestion (CDG) Ventes · prévision par magasin et par produit, au jour · atterrissage fin {YEAR}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 1. Le contexte : de quoi part-on ? -------------------------------- #
    st.markdown(f"""
    <div class="story">
    <div class="lead">Mettez-vous à la place du contrôleur de gestion</div>
    <p>Nous sommes à la mi-{YEAR}. Le premier semestre est <b>connu</b> : on sait exactement ce que
    chaque magasin a vendu. Mais la direction attend une réponse à une question simple et difficile&nbsp;:
    <b>où va-t-on atterrir en fin d'année, et avec quels moyens&nbsp;?</b></p>
    <p>Y répondre «&nbsp;au feeling&nbsp;» ne tient pas&nbsp;: il y a <b>12 magasins</b>, <b>60 produits</b>,
    des saisons, des promotions, des jours fériés, une météo capricieuse. Trop de combinaisons pour une
    intuition. L'idée de cet outil&nbsp;: <b>apprendre des 5 dernières années</b> pour prolonger l'histoire
    de façon crédible — puis dérouler, une question après l'autre, tout ce qu'un pilote de l'activité a
    besoin de savoir.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 2. Le fil rouge : les 5 questions, dans l'ordre ------------------- #
    st.markdown('<div class="sec">Le fil rouge — 5 questions, dans l\'ordre</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="story" style="padding-top:14px">
    <p style="margin-bottom:2px">Chaque page du menu de gauche répond à <b>une</b> question. Elles se
    lisent comme un raisonnement continu&nbsp;: on prévoit, on projette l'année, on en tire des moyens,
    puis on se contrôle.</p>
    <div class="journey">
      <div class="qstep"><div class="num">1</div><div class="body">
        <span class="q">Combien va-t-on vendre&nbsp;?</span><br>
        On prolonge l'historique sur les 6 prochains mois, magasin par magasin et produit par produit,
        puis on traduit ces volumes en <b>chiffre d'affaires</b> (le montant encaissé).
        <br><span class="to">→ page «&nbsp;Combien va-t-on vendre&nbsp;?&nbsp;»</span></div></div>
      <div class="qstep"><div class="num">2</div><div class="body">
        <span class="q">Comment va finir l'année&nbsp;?</span><br>
        On colle le <b>réel déjà encaissé</b> (janv.–juin) devant le <b>prévu</b> (juil.–déc.)&nbsp;:
        leur somme est l'«&nbsp;atterrissage&nbsp;», la meilleure estimation du résultat annuel.
        <br><span class="to">→ page «&nbsp;Comment finira l'année&nbsp;?&nbsp;»</span></div></div>
      <div class="qstep"><div class="num">3</div><div class="body">
        <span class="q">De combien de vendeurs a-t-on besoin&nbsp;?</span><br>
        Des ventes prévues, on déduit la <b>charge de travail</b>&nbsp;: combien de personnes il faut
        par magasin, heure par heure, pour absorber l'activité attendue.
        <br><span class="to">→ page «&nbsp;Besoin en personnel&nbsp;»</span></div></div>
      <div class="qstep"><div class="num">4</div><div class="body">
        <span class="q">S'est-on trompé, et pourquoi&nbsp;?</span><br>
        Sur les mois déjà passés, on confronte le prévu au réel et on <b>décortique l'écart</b>&nbsp;:
        vient-il du prix, du volume, du mix de produits, des promotions&nbsp;?
        <br><span class="to">→ page «&nbsp;Écarts prévu / réel&nbsp;»</span></div></div>
      <div class="qstep"><div class="num">5</div><div class="body">
        <span class="q">Sur quoi le modèle s'appuie-t-il&nbsp;?</span><br>
        On ouvre la boîte noire&nbsp;: quelles informations pèsent le plus dans ses prévisions&nbsp;?
        De quoi vérifier que sa logique est <b>plausible pour un métier retail</b>.
        <br><span class="to">→ page «&nbsp;Ce qui pilote la prévision&nbsp;»</span></div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 3. Comment l'outil devine l'avenir ------------------------------- #
    st.markdown('<div class="sec">Comment l\'outil devine l\'avenir</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="story">
    <p style="margin-bottom:4px">Pas de boule de cristal&nbsp;: rien que du passé, lu méthodiquement.
    Trois idées suffisent à comprendre la mécanique.</p>
    <div class="method">
      <div class="mcard"><div class="mn">1 · Apprendre</div><div class="mt">Digérer 5 ans de ventes</div>
        <div class="md">L'outil lit chaque journée de vente des 5 dernières années — près d'un million
        et demi de lignes magasin&nbsp;× produit&nbsp;× jour.</div></div>
      <div class="mcard"><div class="mn">2 · Repérer</div><div class="mt">Retrouver les régularités</div>
        <div class="md">Il isole seul les rythmes cachés&nbsp;: pic du samedi, saisons, effet des promos,
        des jours fériés, de la météo — sans qu'on les lui souffle.</div></div>
      <div class="mcard"><div class="mn">3 · Prolonger</div><div class="mt">Rejouer vers l'avenir</div>
        <div class="md">Il applique ces régularités aux 6 mois à venir, jour par jour, puis on additionne
        pour obtenir des chiffres à l'échelle du magasin, du mois, de l'enseigne.</div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 4. Peut-on lui faire confiance ? --------------------------------- #
    st.markdown('<div class="sec">Peut-on lui faire confiance&nbsp;?</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="story">
    <p>Une prévision ne vaut que si on sait <b>de combien elle se trompe</b>. On le mesure sans tricher&nbsp;:
    on cache à l'outil une période qu'on connaît déjà, on lui demande de la prévoir, puis on compare à la
    réalité. On répète l'exercice sur plusieurs périodes glissantes ({config.BACKTEST_N_FOLDS} passages de
    {config.BACKTEST_HORIZON_DAYS} jours) — c'est le «&nbsp;backtest à origine glissante&nbsp;».</p>
    <p style="margin-bottom:4px">L'erreur se lit en <b>WAPE</b> : l'écart moyen entre prévu et réel, en&nbsp;%
    (plus c'est bas, mieux c'est). Et la barre à battre est une prévision «&nbsp;bête&nbsp;» — <b>répéter la
    semaine précédente</b>&nbsp;: un modèle qui ne fait pas mieux ne sert à rien.</p>
    <div class="method">
      <div class="mcard"><div class="mn">Résultat</div><div class="mt">WAPE ≈ 0,69 vs 0,94</div>
        <div class="md">L'outil se trompe nettement moins que la prévision naïve, sur <b>chacun</b> des
        12&nbsp;magasins et des 8&nbsp;familles.</div></div>
      <div class="mcard"><div class="mn">À l'échelle pilotage</div><div class="mt">WAPE ≈ 0,16</div>
        <div class="md">Ramené au niveau magasin&nbsp;× jour — celui du contrôleur — l'erreur devient
        faible&nbsp;: la prévision est exploitable pour décider.</div></div>
      <div class="mcard"><div class="mn">Apport météo</div><div class="mt">+5&nbsp;% (jusqu'à +16&nbsp;%)</div>
        <div class="md">Le scénario&nbsp;2 (météo) affine surtout les jours atypiques&nbsp;: gain net les
        jours de pluie ou de forte anomalie de température.</div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 5. Comment s'en servir ------------------------------------------- #
    st.markdown(f"""
    <div class="intro">
    <div class="lead">Pour commencer</div>
    En haut de la barre de gauche, choisissez un <b>scénario</b> et un <b>magasin</b> — ces deux choix
    s'appliquent à toutes les pages. Puis suivez le fil, de la page&nbsp;1 à la page&nbsp;5. Le
    <b>scénario&nbsp;1</b> s'appuie sur l'historique, le calendrier et les promotions&nbsp;; le
    <b>scénario&nbsp;2</b> ajoute l'effet de la météo.
    <div class="fine">
    <b>Périmètre&nbsp;:</b> 12 magasins (villes françaises) + 1 canal Online · 60 produits en 8 familles ·
    5 ans d'historique quotidien.
    <br><b>Abréviations&nbsp;:</b> <b>CA</b> = chiffre d'affaires · <b>PV</b> = prix de vente ·
    <b>PA</b> = panier article (nombre d'articles par ticket) · <b>PM</b> = panier moyen (PV × PA) ·
    <b>ETP</b> = équivalent temps plein (1 = un salarié à temps complet) ·
    <b>WAPE</b> = erreur moyenne de la prévision, en&nbsp;% (plus c'est bas, plus la prévision est fiable).
    </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Sur quoi reposent ces données ? — hypothèses de la simulation "
                     "(cliquer pour déplier)"):
        st.markdown("""
#### Pourquoi des données simulées&nbsp;?
Ce démonstrateur n'a pas encore accès aux vraies données de l'enseigne. **Toutes les données
affichées sont fabriquées par ordinateur**, uniquement pour montrer comment l'outil fonctionne.
**Attention : les montants n'ont aucune valeur réelle** : ils ne doivent pas être lus comme de vrais
chiffres d'affaires.

#### Ce que représentent les données
- **12 magasins = 12 vraies villes françaises.** La taille de la ville fixe la taille du magasin
  (Paris = grand, Brive = petit).
- **1 magasin « Online »** pour le canal e-commerce.
- **60 produits d'animalerie** en 8 familles (chien, chat, oiseau, aquariophilie, rongeur,
  reptile, hygiène & soins, accessoires & jouets), dont 4 « lancés » en cours de période.
- **5 ans de ventes quotidiennes** (mi-2021 à mi-2026), pour chaque magasin et chaque produit.

#### Ce qu'on a volontairement caché dans les données (et que l'outil doit retrouver seul)
C'est le vrai test : on injecte des comportements réalistes, puis on vérifie que le moteur les
**redécouvre tout seul**.
- **Rythmes saisonniers** : pic du samedi, ventes qui suivent les saisons (antiparasitaires l'été,
  oiseaux l'hiver, jouets à Noël).
- **Jours fériés et fêtes**, **tendances de fond** (l'Online grandit ~+14 %/an), **inflation** des
  prix calquée sur 2021-2023.
- **6 types de promotions**, chacun avec son effet propre (une remise crée un pic puis un creux&nbsp;;
  une opération influenceur a un effet décalé d'environ un mois&nbsp;; etc.).
- **Effet météo**, **ruptures de stock** (~1 % des cas) et **produits qui ne se vendent pas tous les
  jours** (~45 % de journées sans vente).

#### La seule donnée réelle&nbsp;: la météo
On a utilisé la **vraie météo historique** des 12 villes (source Open-Meteo), figée dans le projet,
pour que le scénario 2 ait un vrai signal à apprendre. *(Son effet a été volontairement accentué
pour être détectable&nbsp;; son poids réel devra être mesuré sur de vraies données.)*

#### Garde-fou
À chaque génération, **13 vérifications automatiques** s'assurent que les données restent cohérentes.
""")

    st.markdown("""<div class="expl" style="margin-top:14px">
    <b>À garder en tête.</b> Certaines règles restent des <b>hypothèses de travail</b>, à caler avec le
    métier&nbsp;: définitions EB/PA, normes de productivité ETP, projection du panier article, écart
    d'inflation. Et surtout&nbsp;: <b>toutes les données sont synthétiques</b> — les montants illustrent
    la mécanique, ils n'ont aucune valeur métier (détail dans le README).
    </div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE — Combien va-t-on vendre ? (prévision & CA + hypothèses/promo)
# --------------------------------------------------------------------------- #
def page_ca():
    ctx = get_context()
    page_header("Combien va-t-on vendre&nbsp;?",
                f"Chiffre d'affaires attendu sur le 2ᵉ semestre {YEAR} — "
                f"{store_label(ctx['sel_store'])} · scénario {ctx['scenario']}")
    if ctx["missing"]:
        forecast_missing_stop(ctx)

    story_block(
        question="Combien chaque magasin va-t-il vendre sur le 2ᵉ semestre, et combien cela "
                 "représente-t-il en chiffre d'affaires&nbsp;?",
        method="On prolonge l'historique produit par produit, puis on remonte la cascade "
               "<b>volume → panier moyen → CA net</b>. Vous pouvez tester vos propres hypothèses ou "
               "une campagne promo (encart repliable plus bas)&nbsp;: tout se recalcule.",
        read="Les <b>barres</b> = le CA net prévu mois par mois&nbsp;; la <b>zone ombrée</b> = la "
             "fourchette P10–P90 (dans 8 cas sur 10, le réel tombe dedans). Le KPI "
             "<b>Erreur moyenne (WAPE)</b> est un taux d'<b>erreur</b>&nbsp;: plus il est bas, "
             "meilleure est la prévision. Si vous testez des ajustements (encart en bas), une "
             "<b>ligne orange</b> apparaît pour montrer la prévision d'origine.",
        good="Une fourchette resserrée autour de la prévision (les aléas se compensent au cumul), une "
             "erreur nettement sous celle de la prévision naïve, et un profil mensuel cohérent avec "
             "la saison.")

    cv, cbv = ctx["casc_view"], ctx["casc_base_view"]
    ca_scn, ca_ref = cv.ca_net.sum(), cbv.ca_net.sum()
    delta = ca_scn / ca_ref - 1 if ca_ref else 0
    ajuste = abs(delta) > 1e-9        # l'utilisateur a-t-il modifié quelque chose ?
    err_main, err_sub = "—", ""
    if ctx["RES"]["summary"] is not None:
        s = ctx["RES"]["summary"]
        wh = s.loc[s.modele == "hybride", "WAPE"].iloc[0]
        wn = s.loc[s.modele == "naive_saiso", "WAPE"].iloc[0]
        err_main = pct(wh)
        err_sub = f"vs {pct(wn)} pour une prévision naïve"

    k1, k2, k3, k4 = st.columns(4)
    # KPI 1 — le chiffre principal
    k1.markdown(f'<div class="kpi"><div class="kpi-l">CA net prévu — S2 {YEAR}</div>'
                f'<div class="kpi-v">{eur(ca_scn)}</div>'
                f'<div class="kpi-d" style="color:{OK_TEXT if delta >= 0 else ORANGE_TEXT}">'
                + (f'{pct(delta, 1, signe=True)} vs sans ajustement' if ajuste
                   else f'<span style="color:{MUTED}">aucun ajustement appliqué</span>')
                + '</div></div>', unsafe_allow_html=True)
    # KPI 2 — n'a de sens QUE si l'utilisateur a ajusté ; sinon c'est le même
    # chiffre affiché deux fois (source de confusion signalée en démo).
    if ajuste:
        k2.markdown(f'<div class="kpi"><div class="kpi-l">Sans vos ajustements</div>'
                    f'<div class="kpi-v">{eur(ca_ref)}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">prévision brute de l\'outil</div></div>',
                    unsafe_allow_html=True)
    else:
        tr = cv.transactions.sum()
        k2.markdown(f'<div class="kpi"><div class="kpi-l">Transactions prévues</div>'
                    f'<div class="kpi-v">{fr(tr / 1e3, 0)}{NBSP}k</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">panier moyen '
                    f'{eur(ca_scn / max(tr, 1), 2)}</div></div>', unsafe_allow_html=True)
    # KPI 3 — ATTENTION : le WAPE est une ERREUR. L'ancien libellé « Fiabilité :
    # 69,4 % » se lisait « fiable à 69 % » alors que c'est l'inverse.
    k3.markdown(f'<div class="kpi"><div class="kpi-l">Erreur moyenne (WAPE)</div>'
                f'<div class="kpi-v">{err_main}</div>'
                f'<div class="kpi-d" style="color:{MUTED}">plus c\'est bas, mieux c\'est · '
                f'{err_sub}</div></div>', unsafe_allow_html=True)
    factors = load_pi_factors(ctx["scenario"])
    band = ca_band_by_month(cv, factors)
    total_band = ca_band_total(cv, factors)
    if total_band is not None:
        _, tlo, thi = total_band
        k4.markdown(f'<div class="kpi"><div class="kpi-l">Fourchette S2 (P10–P90)</div>'
                    f'<div class="kpi-v">{eur(tlo)} – {eur(thi)}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">8 chances sur 10 d\'être '
                    f'dans cette plage</div></div>', unsafe_allow_html=True)
    else:
        k4.markdown(f'<div class="kpi"><div class="kpi-l">Fourchette S2 (P10–P90)</div>'
                    f'<div class="kpi-v">—</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">non calibrée</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec">Chiffre d\'affaires net prévu, mois par mois</div>',
                unsafe_allow_html=True)
    mm = cascade_monthly_cached(cv)
    mm_base = cascade_monthly_cached(cbv)
    ser = mm.groupby("month").ca_net.sum()
    # Axe en MILLIONS d'euros : un axe en euros bruts affichait « 1200000 »,
    # illisible. On trace donc en M€ et on remet le montant exact au survol.
    xs = [mois_fr(m) for m in ser.index]
    mfig = go.Figure()
    if band is not None:
        bx = [mois_fr(m) for m in band.month]
        mfig.add_scatter(x=bx, y=band.hi / 1e6, mode="lines", line=dict(width=0),
                         hoverinfo="skip", showlegend=False)
        mfig.add_scatter(x=bx, y=band.lo / 1e6, mode="lines", line=dict(width=0),
                         fill="tonexty", fillcolor="rgba(33,25,72,0.13)",
                         name="Fourchette P10–P90 (8 chances sur 10)", hoverinfo="skip")
    mfig.add_bar(x=xs, y=ser.values / 1e6, name="CA net prévu", marker_color=NAVY,
                 customdata=[eur(v) for v in ser.values],
                 hovertemplate="<b>%{x}</b><br>CA net prévu : %{customdata}<extra></extra>")
    # La courbe « sans ajustement » n'est tracée QUE si l'utilisateur a ajusté
    # quelque chose : sinon elle se superpose exactement aux barres (deux séries
    # identiques = confusion inutile).
    if ajuste:
        base = mm_base.groupby("month").ca_net.sum()
        mfig.add_scatter(x=[mois_fr(m) for m in base.index], y=base.values / 1e6,
                         name="Sans vos ajustements", mode="lines+markers",
                         line=dict(color=ORANGE, width=3),
                         customdata=[eur(v) for v in base.values],
                         hovertemplate="<b>%{x}</b><br>Sans ajustement : %{customdata}"
                                       "<extra></extra>")
    mfig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=-0.18), **PLOTLY_FR)
    mfig.update_yaxes(title_text="CA net du mois (millions d'euros)", tickformat=",.1f",
                      ticksuffix=" M€", gridcolor=LINE, zeroline=False)
    mfig.update_xaxes(title_text="", type="category")
    st.plotly_chart(mfig, use_container_width=True)
    if band is not None:
        st.caption("Lecture — **barres** : le CA net prévu chaque mois. **Zone ombrée** : la fourchette "
                   "P10–P90, calibrée sur les erreurs passées ; dans 8 cas sur 10 le réel devrait y "
                   "tomber. Elle s'élargit avec l'horizon : décembre est plus incertain que juillet.")

    # ------ contrôles facultatifs, repliés par défaut pour garder la page nette ------
    with st.expander("**Simuler : « et si… ? » — tester des hypothèses ou une promotion "
                     "(facultatif)**"):
        st.markdown('<div class="expl" style="box-shadow:none;padding:0 0 10px">'
                    'Ces deux encadrés servent à <b>tester un scénario</b>. Tant que vous n\'y touchez '
                    'pas, l\'outil affiche sa prévision brute. Dès que vous modifiez quelque chose, '
                    '<b>tous les chiffres et graphiques de l\'application se recalculent</b> (y compris '
                    'l\'atterrissage), et une <b>ligne orange</b> apparaît sur le graphique pour '
                    'rappeler la prévision d\'origine.</div>', unsafe_allow_html=True)
        hcol, pcol = st.columns([1.15, 1])
        with hcol:
            st.markdown("**A · Ajuster les hypothèses, mois par mois**")
            st.caption("Chaque valeur est un **multiplicateur** : 1,00 = inchangé · 1,02 = +2 % · "
                       "0,95 = −5 %. **PV** = prix de vente, **PA** = articles par ticket. "
                       "« Δ inflation » = points d'inflation ajoutés ou retirés par rapport à la "
                       "trajectoire déjà intégrée aux prix (0 = on ne touche à rien).")
            hyp = st.data_editor(
                st.session_state.hyp, hide_index=True, use_container_width=True,
                column_config={
                    "month": st.column_config.TextColumn("Mois", disabled=True),
                    "coef_pv": st.column_config.NumberColumn(
                        "PV ×", min_value=0.5, max_value=1.5, step=0.01,
                        help="Multiplicateur du prix de vente moyen."),
                    "coef_pa": st.column_config.NumberColumn(
                        "PA ×", min_value=0.5, max_value=1.5, step=0.01,
                        help="Multiplicateur du nombre d'articles par ticket."),
                    "coef_transactions": st.column_config.NumberColumn(
                        "Transactions ×", min_value=0.5, max_value=1.5, step=0.01,
                        help="Multiplicateur du nombre de tickets (la fréquentation)."),
                    "inflation_delta_pct": st.column_config.NumberColumn(
                        "Δ inflation %", min_value=-5.0, max_value=5.0, step=0.1,
                        help="Écart d'inflation en points, en plus ou en moins."),
                })
            st.session_state.hyp = hyp
        with pcol:
            st.markdown("**B · Simuler une campagne promotionnelle**")
            st.caption("Décrivez la campagne : l'outil propose une hausse des ventes déduite de "
                       "l'historique des promos similaires, que vous pouvez corriger. "
                       "Rien n'est appliqué tant que vous n'avez pas cliqué sur le bouton.")
            with st.form("promo_form"):
                f1, f2 = st.columns(2)
                p_name = f1.text_input("Nom de la campagne", "Ma campagne test",
                                       help="Sert uniquement à vous repérer.")
                p_type = f2.selectbox("Type de promotion", config.PROMO_TYPES,
                                      help="Chaque type a un effet différent : une remise crée un pic "
                                           "puis un creux, une opération influenceur agit avec ~1 mois "
                                           "de décalage, etc.")
                f3, f4, f5 = st.columns(3)
                p_start = f3.date_input("Début", pd.Timestamp(f"{YEAR}-10-01"),
                                        min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                                        max_value=pd.Timestamp(config.FORECAST_END),
                                        help="Premier jour de la campagne.")
                p_end = f4.date_input("Fin", pd.Timestamp(f"{YEAR}-10-14"),
                                      min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                                      max_value=pd.Timestamp(config.FORECAST_END),
                                      help="Dernier jour de la campagne.")
                p_perim = f5.selectbox("Périmètre", ["omnicanal", "magasin", "online"],
                                       help="Où la campagne s'applique : partout (omnicanal), "
                                            "en magasin seulement, ou sur le site seulement.")
                # Sliders en POURCENTAGES ENTIERS : en flottants, Streamlit
                # affichait « 0.20 » / « 1.47 » (décimales anglaises, et valeur
                # illisible car écrite en orange sur le curseur orange). Un entier
                # « 20 % » est à la fois lisible et plus parlant qu'un multiplicateur.
                p_disc_pct = st.slider(
                    "Niveau de remise (%)", 0, 50, 20, 5,
                    help="Utilisé uniquement par le type « produits » : 20 = 20 % de remise.")
                p_disc = p_disc_pct / 100
                p_cat = st.multiselect("Familles de produits ciblées",
                                       sorted(TABLES["products"].commodity_group.unique()),
                                       default=["Chien"],
                                       help="La campagne ne touchera que ces familles.")
                proposed = propose_uplift(UPLIFT_REF, p_type, p_disc)
                prop_pct = int(round((float(proposed) - 1) * 100))
                p_uplift_pct = st.slider(
                    "Hausse des ventes attendue (%)", -20, 200, prop_pct, 5,
                    help="Combien de ventes en plus la campagne apporte sur le périmètre ciblé. "
                         "50 = +50 % de ventes. L'outil pré-remplit cette valeur d'après "
                         "l'historique des promos du même type ; corrigez-la si besoin.")
                p_uplift = 1 + p_uplift_pct / 100
                st.caption(f"Proposition de l'outil pour ce type de promo : "
                           f"**{'+' if prop_pct >= 0 else ''}{prop_pct}{NBSP}%** de ventes "
                           f"(médiane observée dans l'historique).")
                p_apply = st.form_submit_button("Appliquer la campagne à la prévision")
        if p_apply:
            skus = TABLES["products"].loc[
                TABLES["products"].commodity_group.isin(p_cat), "sku_id"].tolist()
            st.session_state.fc_sim = apply_promo_to_forecast(
                ctx["RES"]["forecast"], promo_type=p_type, date_start=p_start, date_end=p_end,
                perimeter=p_perim, sku_ids=skus, uplift=p_uplift,
                discount_rate=p_disc if p_type == "produits" else 0.0)
            st.session_state.fc_sim_scenario = ctx["scenario"]
            st.session_state.promo_name = p_name
            st.rerun()
        if st.session_state.get("fc_sim") is not None and \
                st.session_state.get("fc_sim_scenario") == ctx["scenario"]:
            c1b, c2b = st.columns([3, 1])
            c1b.success(f"Campagne « {st.session_state.get('promo_name', '')} » appliquée.")
            if c2b.button("Retirer la campagne"):
                st.session_state.fc_sim = None
                st.rerun()


# --------------------------------------------------------------------------- #
# PAGE — Comment finira l'année ? (atterrissage)
# --------------------------------------------------------------------------- #
def page_atterrissage():
    ctx = get_context()
    page_header(f"Comment va finir l'année&nbsp;? — atterrissage {YEAR}",
                f"{store_label(ctx['sel_store'])} · scénario {ctx['scenario']}")
    if ctx["missing"]:
        forecast_missing_stop(ctx)

    story_block(
        question="Si l'on prolonge la tendance, où l'année entière va-t-elle atterrir en chiffre "
                 "d'affaires&nbsp;?",
        method="On colle bout à bout le <b>réel déjà encaissé</b> (janv.–juin) et le <b>prévu</b> "
               "(juil.–déc.), puis on additionne. C'est un « rolling forecast »&nbsp;: l'estimation "
               "se précise chaque mois, à mesure que du réel remplace du prévisionnel.",
        read="Barres <b>vertes</b> = réel observé (acquis)&nbsp;; barres <b>bleues</b> = prévision, "
             "avec leur <b>fourchette P10–P90</b> (les moustaches). Le grand chiffre en haut est "
             "l'atterrissage central&nbsp;; la fourchette ne porte que sur la partie prévue "
             "(le réalisé est acquis).",
        good="Une fourchette d'atterrissage étroite au regard des enjeux&nbsp;: l'incertitude sur le "
             "semestre à venir reste maîtrisée une fois le 1ᵉʳ semestre acquis.")

    mm = cascade_monthly_cached(ctx["casc_view"])
    sales = TABLES["sales"]
    reel = sales[sales.date.dt.year == YEAR]
    if ctx["sel_store"] != "Tous":
        reel = reel[reel.store_id == ctx["sel_store"]]
    reel_m = reel.assign(month=reel.date.dt.to_period("M").astype(str)).groupby("month").revenue.sum()
    prev_m = mm.groupby("month").ca_net.sum()

    total = reel_m.sum() + prev_m.sum()
    # fourchette : seule la partie prévue (juil–déc) est incertaine ; le réel est acquis.
    factors = load_pi_factors(ctx["scenario"])
    band_total = ca_band_total(ctx["casc_view"], factors)
    k1, k2, k3 = st.columns(3)
    if band_total is not None:
        _, plo, phi = band_total
        k1.markdown(f'<div class="kpi"><div class="kpi-l">Atterrissage {YEAR}</div>'
                    f'<div class="kpi-v">{eur(total)}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">fourchette&nbsp;: '
                    f'{eur(reel_m.sum() + plo)} – {eur(reel_m.sum() + phi)}</div></div>',
                    unsafe_allow_html=True)
    else:
        k1.markdown(f'<div class="kpi"><div class="kpi-l">Atterrissage {YEAR}</div>'
                    f'<div class="kpi-v">{eur(total)}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">année entière estimée</div></div>',
                    unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-l">Déjà réalisé (janv–juin)</div>'
                f'<div class="kpi-v">{eur(reel_m.sum())}</div>'
                f'<div class="kpi-d" style="color:{OK_TEXT}">observé — chiffre acquis</div></div>',
                unsafe_allow_html=True)
    if band_total is not None:
        _, plo, phi = band_total
        k3.markdown(f'<div class="kpi"><div class="kpi-l">Reste à faire (juil–déc)</div>'
                    f'<div class="kpi-v">{eur(prev_m.sum())}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">prévu · fourchette '
                    f'{eur(plo)} – {eur(phi)}</div></div>', unsafe_allow_html=True)
    else:
        k3.markdown(f'<div class="kpi"><div class="kpi-l">Reste à faire (juil–déc)</div>'
                    f'<div class="kpi-v">{eur(prev_m.sum())}</div>'
                    f'<div class="kpi-d" style="color:{MUTED}">prévu</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec">Mois par mois — ce qui est acquis, puis ce qui est prévu</div>',
                unsafe_allow_html=True)
    afig = go.Figure()
    afig.add_bar(x=[mois_fr(m) for m in reel_m.index], y=reel_m.values / 1e6,
                 name="Réel encaissé (janv–juin)", marker_color=OK,
                 customdata=[eur(v) for v in reel_m.values],
                 hovertemplate="<b>%{x}</b><br>Réel encaissé : %{customdata}<extra></extra>")
    afig.add_bar(x=[mois_fr(m) for m in prev_m.index], y=prev_m.values / 1e6,
                 name="Prévision (juil–déc)", marker_color=NAVY,
                 customdata=[eur(v) for v in prev_m.values],
                 hovertemplate="<b>%{x}</b><br>Prévu : %{customdata}<extra></extra>")
    # Moustaches d'incertitude sur les mois prévus. Le marqueur porteur est
    # invisible (size=1, opacity=0) : on le sort donc de la légende, où il
    # apparaissait comme une entrée vide et incompréhensible.
    band = ca_band_by_month(ctx["casc_view"], factors)
    if band is not None:
        bm = band.set_index("month").reindex(prev_m.index)
        afig.add_scatter(x=[mois_fr(m) for m in prev_m.index], y=prev_m.values / 1e6,
                         mode="markers", marker=dict(color=NAVY, size=1, opacity=0),
                         error_y=dict(type="data", symmetric=False,
                                      array=((bm.hi - bm.point) / 1e6).values,
                                      arrayminus=((bm.point - bm.lo) / 1e6).values,
                                      color=INK_SOFT, thickness=1.5, width=5),
                         showlegend=False, hoverinfo="skip")
    afig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", y=-0.18), **PLOTLY_FR)
    afig.update_yaxes(title_text="CA net du mois (millions d'euros)", tickformat=",.1f",
                      ticksuffix=" M€", gridcolor=LINE, zeroline=False)
    afig.update_xaxes(type="category")
    st.plotly_chart(afig, use_container_width=True)
    st.caption("Lecture — **barres vertes** : ce qui est déjà encaissé, donc certain. "
               "**Barres bleues** : la prévision, avec sa fourchette P10–P90 (les moustaches "
               "verticales). Les moustaches s'allongent vers décembre : plus c'est loin, plus "
               "c'est incertain.")


# --------------------------------------------------------------------------- #
# PAGE — Besoin en personnel (ETP)
# --------------------------------------------------------------------------- #
def page_etp():
    ctx = get_context()
    page_header("De combien de vendeurs a-t-on besoin&nbsp;?",
                f"Effectif estimé par magasin (2ᵉ semestre {YEAR}) · scénario {ctx['scenario']}")
    if ctx["missing"]:
        forecast_missing_stop(ctx)

    story_block(
        question="Combien de vendeurs faut-il, magasin par magasin, pour absorber l'activité "
                 "prévue&nbsp;?",
        method="On convertit le CA et la fréquentation attendus, heure par heure, en charge de "
               "travail, à l'aide de <b>normes de productivité</b> (CA/heure, tickets/heure) et des "
               "horaires d'ouverture. Résultat en <b>ETP</b> (1 = un temps complet). Les normes "
               "ci-dessous sont ajustables.",
        read="Le graphique du haut compare l'effectif moyen entre magasins&nbsp;; celui du bas montre "
             "la répartition sur une journée type, pour dimensionner les plannings.",
        good="Un classement des magasins cohérent avec leur taille, et une courbe journalière qui "
             "suit les heures d'affluence (le canal Online est exclu — pas de magasin physique).")

    st.markdown('<div class="sec">Les règles de calcul — modifiez-les pour voir l\'impact</div>',
                unsafe_allow_html=True)
    st.caption("Ces 4 valeurs sont des **hypothèses de travail** 🟡, à caler avec le métier. "
               "Chaque modification recalcule immédiatement les deux graphiques ci-dessous.")
    e1, e2, e3, e4 = st.columns(4)
    p_ca = e1.number_input(
        "CA par heure-vendeur (€)", 100.0, 500.0,
        float(ETP_DEFAULTS["prod_ca_per_hour"]), 10.0, format="%.0f",
        help="Combien d'euros de chiffre d'affaires un vendeur peut absorber en une heure. "
             "Plus la valeur est haute, moins il faut de vendeurs.")
    p_tk = e2.number_input(
        "Tickets par heure-vendeur", 5.0, 30.0,
        float(ETP_DEFAULTS["tickets_per_hour"]), 1.0, format="%.0f",
        help="Combien de passages en caisse un vendeur traite en une heure. "
             "C'est la 2ᵉ contrainte : on retient le besoin le plus élevé des deux.")
    p_min = e3.number_input(
        "Effectif minimum par heure", 1.0, 5.0,
        float(ETP_DEFAULTS["min_staff"]), 0.5, format="%.1f",
        help="Plancher de sécurité : nombre de personnes présentes même quand le magasin est vide.")
    p_hm = e4.number_input(
        "Heures par mois pour 1 ETP", 100.0, 200.0,
        float(ETP_DEFAULTS["hours_per_month"]), 1.0, format="%.2f",
        help="Durée mensuelle d'un temps plein (35 h/semaine = 151,67 h/mois). "
             "Sert à convertir des heures de travail en nombre de personnes.")

    params = (("prod_ca_per_hour", p_ca), ("tickets_per_hour", p_tk),
              ("min_staff", p_min), ("hours_per_month", p_hm))
    etp_hourly, _, etp_monthly = compute_etp_cached(ctx["casc"], TABLES["hourly"], STORES, params)
    etp_avg = etp_monthly.groupby("store_id").etp.mean().sort_values(ascending=False)

    st.markdown('<div class="sec">Effectif moyen nécessaire, par magasin</div>', unsafe_allow_html=True)
    efig = go.Figure(go.Bar(
        x=[STORE_NAME.get(s, s) for s in etp_avg.index], y=etp_avg.values,
        marker_color=NAVY, text=[fr(v, 1) for v in etp_avg.values], textposition="outside",
        customdata=[[s, fr(v, 1)] for s, v in zip(etp_avg.index, etp_avg.values)],
        hovertemplate="<b>%{x}</b> (%{customdata[0]})<br>"
                      "%{customdata[1]} personnes à temps plein<extra></extra>"))
    # marge gauche explicite : le titre d'axe était tronqué (« … par m »)
    efig.update_layout(height=360, margin=dict(l=70, r=10, t=30, b=80), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", **PLOTLY_FR)
    efig.update_yaxes(title_text="ETP moyen par mois", gridcolor=LINE, zeroline=False)
    efig.update_xaxes(tickangle=-40)
    st.plotly_chart(efig, use_container_width=True)
    st.caption(f"Lecture — un **ETP** = un salarié à temps plein. « {fr(etp_avg.iloc[0], 1)} » pour "
               f"le 1ᵉʳ magasin signifie qu'il faut l'équivalent de {fr(etp_avg.iloc[0], 1)} personnes "
               "à temps plein en moyenne sur le semestre. Le canal Online est absent : pas de "
               "magasin physique.")

    etp_store = ctx["sel_store"] if ctx["sel_store"] != "Tous" else etp_avg.index[0]
    auto = ctx["sel_store"] == "Tous"
    st.markdown(f'<div class="sec">Une journée type (samedi) — {store_label(etp_store)}</div>',
                unsafe_allow_html=True)
    st.caption(("Aucun magasin n'étant sélectionné, on montre ici **le plus gros du réseau**. "
                "Choisissez un magasin dans le menu de gauche pour voir le sien. " if auto else "")
               + "Le samedi est le jour le plus chargé : c'est lui qui dimensionne le planning.")
    day_prof = (etp_hourly[etp_hourly.store_id == etp_store]
                .assign(dow=lambda d: pd.to_datetime(d.date).dt.dayofweek)
                .query("dow == 5").groupby("hour").staff.mean())
    pfig = go.Figure(go.Scatter(
        x=[f"{h}h" for h in day_prof.index], y=day_prof.values, fill="tozeroy",
        line=dict(color=ORANGE, width=3), mode="lines+markers",
        customdata=[fr(v, 1) for v in day_prof.values],
        hovertemplate="<b>%{x}</b><br>%{customdata} personnes en rayon<extra></extra>"))
    pfig.update_layout(height=280, margin=dict(l=60, r=10, t=20, b=40), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", **PLOTLY_FR)
    pfig.update_xaxes(title_text="Heure de la journée", type="category")
    pfig.update_yaxes(title_text="Personnes en rayon", gridcolor=LINE, zeroline=False)
    st.plotly_chart(pfig, use_container_width=True)
    st.caption("Lecture — hauteur = nombre de personnes à avoir en rayon à cette heure-là. "
               "Le creux du midi et le pic de l'après-midi sortent des données de fréquentation.")


# --------------------------------------------------------------------------- #
# PAGE — Écarts prévu / réel
# --------------------------------------------------------------------------- #
def page_ecarts():
    ctx = get_context(require_forecast=False)
    page_header("S'est-on trompé, et pourquoi&nbsp;?",
                f"Écart entre le prévu et le réellement vendu · "
                f"{store_label(ctx['sel_store'])} · scénario {ctx['scenario']}")

    story_block(
        question="Quand le réel s'écarte du prévu, d'où vient la différence — et est-ce inquiétant&nbsp;?",
        method="Sur les mois passés, on décompose l'écart de CA en trois causes indépendantes&nbsp;: "
               "<b>le prix</b> (vendu plus ou moins cher), <b>le volume</b> (plus ou moins d'unités), "
               "<b>le mix</b> (produits plus ou moins chers que d'habitude). L'écart de volume est "
               "ensuite reventilé entre promotions, calendrier et reste.",
        read="La page se lit en 3 temps&nbsp;: <b>1</b> le constat (prévu, réel, écart), "
             "<b>2</b> les 3 causes de l'écart (vert = a fait gagner du CA, orange = a fait perdre), "
             "<b>3</b> le détail de l'effet volume.",
        good="Des écarts qui s'expliquent par des causes identifiables (une promo, un jour férié) "
             "plutôt que par un gros « autre » inexpliqué.")

    if ctx["RES"]["ecarts"] is None:
        st.info(f"Décomposition non calculée. Lancer :  `python -m src.ecarts --scenario {ctx['scenario']}`")
        return
    ec = ctx["RES"]["ecarts"]
    if ctx["sel_store"] != "Tous":
        ec = ec[ec.store_id == ctx["sel_store"]]
    if ec.empty:
        st.info("Aucune donnée d'écart pour ce périmètre.")
        return
    agg = ec[["ca_prev", "effet_prix", "effet_volume", "effet_mix"]].sum()
    drivers = ec[["dont_promo", "dont_calendaire", "dont_autre"]].sum()
    ca_prev = float(agg.ca_prev)
    ecart = float(agg.effet_prix + agg.effet_volume + agg.effet_mix)
    ca_reel = ca_prev + ecart

    # --- 1. Les TOTAUX en KPI ------------------------------------------------- #
    # Ils étaient auparavant dans le même graphique que les écarts : des barres de
    # ~3 M€ à côté d'effets de ~40 k€ écrasaient totalement ces derniers (invisibles).
    # On sépare : les totaux ici, les écarts dans leur propre graphique ci-dessous.
    st.markdown('<div class="sec">1 · Le constat — prévu contre réel</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.markdown(f'<div class="kpi"><div class="kpi-l">CA prévu</div>'
                f'<div class="kpi-v">{eur(ca_prev)}</div>'
                f'<div class="kpi-d" style="color:{MUTED}">ce que l\'outil annonçait</div></div>',
                unsafe_allow_html=True)
    t2.markdown(f'<div class="kpi"><div class="kpi-l">CA réel</div>'
                f'<div class="kpi-v">{eur(ca_reel)}</div>'
                f'<div class="kpi-d" style="color:{OK_TEXT}">ce qui a été encaissé</div></div>',
                unsafe_allow_html=True)
    t3.markdown(f'<div class="kpi"><div class="kpi-l">Écart total</div>'
                f'<div class="kpi-v" style="color:{OK_TEXT if ecart >= 0 else ORANGE_TEXT}">'
                f'{"+" if ecart >= 0 else "−"}{eur(abs(ecart))}</div>'
                f'<div class="kpi-d" style="color:{MUTED}">soit '
                f'{pct(ecart / ca_prev if ca_prev else 0, 1, signe=True)} du CA prévu</div></div>',
                unsafe_allow_html=True)

    # --- 2. D'où vient l'écart : graphique dédié, à l'échelle des écarts ------ #
    st.markdown('<div class="sec">2 · D\'où vient cet écart&nbsp;?</div>', unsafe_allow_html=True)
    causes = [("Prix", float(agg.effet_prix), "Vendu plus ou moins cher que prévu"),
              ("Volume", float(agg.effet_volume), "Plus ou moins d'unités vendues"),
              ("Mix", float(agg.effet_mix), "Produits plus ou moins chers que d'habitude")]
    cfig = go.Figure(go.Bar(
        x=[c[0] for c in causes], y=[c[1] / 1e3 for c in causes],
        marker_color=[OK if c[1] >= 0 else ORANGE for c in causes],
        text=[f"{'+' if c[1] >= 0 else '−'}{eur(abs(c[1]))}" for c in causes],
        textposition="outside",
        customdata=[[c[2], f"{'+' if c[1] >= 0 else '−'}{eur(abs(c[1]))}"] for c in causes],
        hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    cfig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", **PLOTLY_FR)
    cfig.update_yaxes(title_text="Contribution à l'écart (milliers d'euros)", tickformat=",.0f",
                      ticksuffix=" k€", gridcolor=LINE, zeroline=True,
                      zerolinecolor=MUTED, zerolinewidth=1)
    st.plotly_chart(cfig, use_container_width=True)
    st.caption("Ces trois causes s'additionnent **exactement** pour reconstituer l'écart total "
               "ci-dessus. Barre **verte** = la cause a fait gagner du CA ; **orange** = elle en a "
               "fait perdre. Échelle propre aux écarts : ils sont ~100 fois plus petits que le CA "
               "total, les mélanger sur un même axe les rendait invisibles.")

    # --- 3. Le détail de l'effet volume -------------------------------------- #
    st.markdown('<div class="sec">3 · Et dans le volume, qu\'est-ce qui joue&nbsp;?</div>',
                unsafe_allow_html=True)
    st.caption(f"L'effet volume ({'+' if agg.effet_volume >= 0 else '−'}"
               f"{eur(abs(float(agg.effet_volume)))}) se répartit entre ces trois postes.")
    d1, d2, d3 = st.columns(3)
    for col, (lab, v, aide) in zip(
            (d1, d2, d3),
            [("Effet promotions", float(drivers.dont_promo), "opérations commerciales"),
             ("Effet calendaire", float(drivers.dont_calendaire), "jours fériés, week-ends, saison"),
             ("Autre", float(drivers.dont_autre), "tendance de fond, météo, aléa")]):
        col.markdown(f'<div class="kpi"><div class="kpi-l">{lab}</div>'
                     f'<div class="kpi-v" style="color:{OK_TEXT if v >= 0 else ORANGE_TEXT}">'
                     f'{"+" if v >= 0 else "−"}{eur(abs(v))}</div>'
                     f'<div class="kpi-d" style="color:{MUTED}">{aide}</div>'
                     f'</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE — Ce qui pilote la prévision (explicabilité)
# --------------------------------------------------------------------------- #
def page_explain():
    page_header("Ce qui pilote la prévision",
                "Poids de chaque information dans les décisions du modèle")

    story_block(
        question="Sur quelles informations le modèle s'appuie-t-il vraiment pour prévoir&nbsp;?",
        method="On mesure, pour chaque information fournie au modèle, sa contribution à ses décisions, "
               "puis on regroupe par grande famille (historique, calendrier, prix, météo…).",
        read="À gauche, les 10 informations les plus influentes&nbsp;; à droite, le total par famille. "
             "Plus la barre est longue, plus l'information pèse.",
        good="Une hiérarchie plausible pour du retail&nbsp;: l'historique récent et la saisonnalité "
             "dominent, prix et météo apportent un complément ciblé — pas de variable absurde en tête.")

    imp_path = config.RESULTS_DIR / "explain" / "feature_importance.csv"
    fam_path = config.RESULTS_DIR / "explain" / "by_family.csv"
    if not imp_path.exists():
        st.info("Explicabilité non calculée. Lancer :  `python -m src.explain --scenario 2`")
        return
    imp = pd.read_csv(imp_path)
    fam = pd.read_csv(fam_path)
    g1, g2 = st.columns([1.5, 1])
    with g1:
        st.markdown('<div class="sec">Les 10 informations les plus influentes</div>', unsafe_allow_html=True)
        top = imp.head(10).iloc[::-1]
        ifig = go.Figure(go.Bar(x=top.importance_pct, y=top.libelle, orientation="h",
                                marker_color=NAVY,
                                text=[f"{fr(v, 1)}{NBSP}%" for v in top.importance_pct],
                                textposition="outside",
                                hovertemplate="<b>%{y}</b><br>%{x:.1f} % du poids<extra></extra>"))
        ifig.update_layout(height=380, margin=dict(l=10, r=40, t=20, b=40), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", **PLOTLY_FR)
        ifig.update_xaxes(title_text="Part du poids dans les décisions (%)", gridcolor=LINE)
        st.plotly_chart(ifig, use_container_width=True)
    with g2:
        st.markdown('<div class="sec">Par famille</div>', unsafe_allow_html=True)
        ff = fam.iloc[::-1]
        # 1 décimale : arrondir à l'entier affichait « Météo : 0 » (pour 0,4 %),
        # ce qui contredisait le texte disant que la météo apporte un complément.
        ffig = go.Figure(go.Bar(x=ff.importance_pct, y=ff.famille, orientation="h",
                                marker_color=ORANGE,
                                text=[f"{fr(v, 1)}{NBSP}%" for v in ff.importance_pct],
                                textposition="outside",
                                hovertemplate="<b>%{y}</b><br>%{x:.1f} % du poids<extra></extra>"))
        ffig.update_layout(height=380, margin=dict(l=10, r=45, t=20, b=40), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", **PLOTLY_FR)
        ffig.update_xaxes(title_text="Part du poids (%)", gridcolor=LINE, range=[0, 92])
        st.plotly_chart(ffig, use_container_width=True)
    st.caption("Lecture — ces poids disent **sur quoi le modèle s'appuie**, pas ce qui « cause » les "
               "ventes. Diagnostic calculé sur le scénario 2 (météo incluse), sur les 5 ans "
               "d'historique.")


# --------------------------------------------------------------------------- #
# Barre latérale (persistante) + navigation
# --------------------------------------------------------------------------- #
pages = st.navigation([
    st.Page(page_guide, title="Guide", default=True),
    st.Page(page_ca, title="Combien va-t-on vendre ?"),
    st.Page(page_atterrissage, title="Comment finira l'année ?"),
    st.Page(page_etp, title="Besoin en personnel"),
    st.Page(page_ecarts, title="Écarts prévu / réel"),
    st.Page(page_explain, title="Ce qui pilote la prévision"),
])

with st.sidebar:
    st.markdown("### Filtres")
    st.caption("Ces deux réglages s'appliquent à **toutes les pages**. "
               "Changez-les : tous les chiffres et graphiques se recalculent.")
    st.radio(
        "Scénario de prévision", [1, 2], key="scenario",
        format_func=lambda s: ("1 — Base (calendrier + promos)" if s == 1
                               else "2 — Base + météo"),
        help="Quelles informations le modèle a le droit d'utiliser. "
             "Scénario 1 : historique, calendrier et promotions. "
             "Scénario 2 : idem + la météo. Comparez les deux pour voir l'apport de la météo.")
    st.selectbox(
        "Magasin analysé", ["Tous"] + STORES.store_id.tolist(), key="sel_store",
        format_func=store_label,
        help="« Réseau entier » additionne les 12 magasins et le canal Online. "
             "Choisissez un magasin pour ne voir que le sien.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("**Données synthétiques** — chiffres fabriqués pour la démonstration, "
               "sans valeur métier.")

pages.run()
