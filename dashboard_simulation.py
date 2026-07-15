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
from src.simulation import apply_promo_to_forecast, propose_uplift, typology_uplift_reference

# --------------------------------------------------------------------------- #
# Charte graphique
# --------------------------------------------------------------------------- #
NAVY, ORANGE, OK, WARN = "#211948", "#E84E24", "#809C30", "#F99500"
INK, INK_SOFT, MUTED, LINE, BG = "#1A1A1A", "#33404F", "#4B5563", "#E5E5E5", "#F2F2F7"
SHADOW = "0 2px 8px rgba(0,0,0,0.30)"

st.set_page_config(page_title="Prévision des ventes", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: Arial, sans-serif; }}
  .stApp {{ background: {BG}; }}
  [data-testid="stToolbar"] {{ display: none; }}
  [data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 700 !important; font-size: 13px !important; }}
  h1,h2,h3 {{ color: {INK}; font-weight: 700; }}
  [data-testid="stSidebar"] {{ background: {NAVY}; }}
  [data-testid="stSidebar"] * {{ color: #fff; }}
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: rgba(255,255,255,.85) !important; }}
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
  div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
  div[data-testid="stDateInput"] input, div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div:has(> input),
  [data-testid="stMultiSelect"] div:has(> input) {{
      background-color: #fff !important; border: 1px solid {LINE} !important; }}
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
  .qstep .to {{ color:{ORANGE}; font-weight:700; }}
  .method {{ display:flex; gap:12px; margin:8px 0 2px; flex-wrap:wrap; }}
  .mcard {{ flex:1 1 200px; background:{BG}; border-radius:10px; padding:14px 16px; }}
  .mcard .mn {{ color:{ORANGE}; font-weight:700; font-size:11px; letter-spacing:.08em;
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
  .narr-good {{ flex:0 0 118px; font-size:10.5px; font-weight:700; color:{OK};
               text-transform:uppercase; letter-spacing:.07em; padding-top:2px; }}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
YEAR = pd.Timestamp(config.HIST_END).year


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
STORE_NAME = STORES.set_index("store_id").store_name


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
        read="Les <b>barres</b> = le CA net simulé mois par mois&nbsp;; la <b>ligne orange</b> = la "
             "proposition de l'outil, sans ajustement, pour comparer. La <b>fiabilité (WAPE)</b> "
             "rappelle l'erreur mesurée sur le passé — plus le % est bas, mieux c'est.",
        good="Un WAPE nettement sous celui de la prévision naïve, et un profil mensuel cohérent avec "
             "la saison (creux d'été, montée vers Noël).")

    cv, cbv = ctx["casc_view"], ctx["casc_base_view"]
    ca_scn, ca_ref = cv.ca_net.sum(), cbv.ca_net.sum()
    delta = ca_scn / ca_ref - 1 if ca_ref else 0
    wape_txt, wape_main = "—", "—"
    if ctx["RES"]["summary"] is not None:
        s = ctx["RES"]["summary"]
        wh = s.loc[s.modele == "hybride", "WAPE"].iloc[0]
        wn = s.loc[s.modele == "naive_saiso", "WAPE"].iloc[0]
        wape_main, wape_txt = f"{wh:.1%}", f"{wh:.1%} · naïve {wn:.1%}"

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi"><div class="kpi-l">CA net simulé — S2 {YEAR}</div>'
                f'<div class="kpi-v">{ca_scn/1e6:,.2f} M€</div>'
                f'<div class="kpi-d" style="color:{OK if delta>=0 else ORANGE}">'
                f'{delta:+.1%} vs proposition outil</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-l">Proposition de l\'outil</div>'
                f'<div class="kpi-v">{ca_ref/1e6:,.2f} M€</div>'
                f'<div class="kpi-d" style="color:{MUTED}">hypothèses neutres</div></div>',
                unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-l">Fiabilité (WAPE)</div>'
                f'<div class="kpi-v">{wape_main}</div>'
                f'<div class="kpi-d" style="color:{MUTED}">{wape_txt}</div></div>',
                unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-l">Transactions prévues</div>'
                f'<div class="kpi-v">{cv.transactions.sum()/1e3:,.0f} k</div>'
                f'<div class="kpi-d" style="color:{MUTED}">panier moyen '
                f'{cv.ca_net.sum()/max(cv.transactions.sum(),1):,.1f} €</div></div>',
                unsafe_allow_html=True)

    mm = cascade_monthly_cached(cv)
    mm_base = cascade_monthly_cached(cbv)
    mfig = go.Figure()
    mfig.add_bar(x=mm.groupby("month").ca_net.sum().index,
                 y=mm.groupby("month").ca_net.sum().values, name="CA net simulé", marker_color=NAVY)
    mfig.add_scatter(x=mm_base.groupby("month").ca_net.sum().index,
                     y=mm_base.groupby("month").ca_net.sum().values, name="Proposition de l'outil",
                     mode="lines+markers", line=dict(color=ORANGE, width=3))
    mfig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    st.plotly_chart(mfig, use_container_width=True)

    # ------ contrôles facultatifs, repliés par défaut pour garder la page nette ------
    with st.expander("**Tester des hypothèses ou une promotion (facultatif)**"):
        st.markdown('<div class="expl" style="box-shadow:none;padding:0 0 10px">Par défaut, l\'outil '
                    'utilise ses propres hypothèses. Ici, vous pouvez tester une variante — les '
                    'valeurs sont des <b>multiplicateurs</b> (1,00 = inchangé, 1,02 = +2 %). '
                    'Le CA et l\'atterrissage se recalculent partout.</div>', unsafe_allow_html=True)
        hcol, pcol = st.columns([1.15, 1])
        with hcol:
            st.caption("Hypothèses mois par mois. « Δ inflation » = point d'inflation ajouté/retiré "
                       "par rapport à la trajectoire déjà intégrée aux prix.")
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
            st.session_state.hyp = hyp
        with pcol:
            st.caption("Simuler une campagne promotionnelle : l'outil propose une hausse des ventes "
                       "issue de l'historique, modifiable avant application.")
            with st.form("promo_form"):
                f1, f2 = st.columns(2)
                p_name = f1.text_input("Nom de campagne", "Ma campagne test")
                p_type = f2.selectbox("Type de promotion", config.PROMO_TYPES)
                f3, f4, f5 = st.columns(3)
                p_start = f3.date_input("Début", pd.Timestamp(f"{YEAR}-10-01"),
                                        min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                                        max_value=pd.Timestamp(config.FORECAST_END))
                p_end = f4.date_input("Fin", pd.Timestamp(f"{YEAR}-10-14"),
                                      min_value=pd.Timestamp(config.HIST_END) + pd.Timedelta(days=1),
                                      max_value=pd.Timestamp(config.FORECAST_END))
                p_perim = f5.selectbox("Périmètre", ["omnicanal", "magasin", "online"])
                p_disc = st.slider("Niveau de remise (pour une promo de type 'produits')", 0.0, 0.5, 0.20, 0.05)
                p_cat = st.multiselect("Familles ciblées",
                                       sorted(TABLES["products"].commodity_group.unique()),
                                       default=["Chien"])
                proposed = propose_uplift(UPLIFT_REF, p_type, p_disc)
                p_uplift = st.slider(f"Hausse des ventes attendue (proposée par l'outil : ×{proposed})",
                                     0.8, 3.0, float(proposed), 0.05)
                p_apply = st.form_submit_button("Appliquer la campagne")
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
        read="Barres <b>vertes</b> = réel observé&nbsp;; barres <b>bleues</b> = prévision. Le grand "
             "chiffre en haut est l'atterrissage&nbsp;: la meilleure estimation actuelle du total "
             "annuel.",
        good="Une transition régulière entre réel et prévu, sans marche d'escalier suspecte au "
             "mois de bascule (juillet).")

    mm = cascade_monthly_cached(ctx["casc_view"])
    sales = TABLES["sales"]
    reel = sales[sales.date.dt.year == YEAR]
    if ctx["sel_store"] != "Tous":
        reel = reel[reel.store_id == ctx["sel_store"]]
    reel_m = reel.assign(month=reel.date.dt.to_period("M").astype(str)).groupby("month").revenue.sum()
    prev_m = mm.groupby("month").ca_net.sum()

    total = (reel_m.sum() + prev_m.sum()) / 1e6
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi"><div class="kpi-l">Atterrissage {YEAR}</div>'
                f'<div class="kpi-v">{total:,.2f} M€</div>'
                f'<div class="kpi-d" style="color:{MUTED}">année entière estimée</div></div>',
                unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-l">Déjà réalisé (janv–juin)</div>'
                f'<div class="kpi-v">{reel_m.sum()/1e6:,.2f} M€</div>'
                f'<div class="kpi-d" style="color:{OK}">observé</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-l">Reste à faire (juil–déc)</div>'
                f'<div class="kpi-v">{prev_m.sum()/1e6:,.2f} M€</div>'
                f'<div class="kpi-d" style="color:{MUTED}">prévu</div></div>', unsafe_allow_html=True)

    afig = go.Figure()
    afig.add_bar(x=reel_m.index, y=reel_m.values, name="Réel (janv–juin)", marker_color=OK)
    afig.add_bar(x=prev_m.index, y=prev_m.values, name="Prévision (juil–déc)", marker_color=NAVY)
    afig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    st.plotly_chart(afig, use_container_width=True)


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

    e1, e2, e3, e4 = st.columns(4)
    p_ca = e1.number_input("€ CA / heure-vendeur **(hypothèse)**", 100.0, 500.0, float(ETP_DEFAULTS["prod_ca_per_hour"]), 10.0)
    p_tk = e2.number_input("Tickets / heure-vendeur **(hypothèse)**", 5.0, 30.0, float(ETP_DEFAULTS["tickets_per_hour"]), 1.0)
    p_min = e3.number_input("Effectif minimum / h **(hypothèse)**", 1.0, 5.0, float(ETP_DEFAULTS["min_staff"]), 0.5)
    p_hm = e4.number_input("Heures / mois / ETP", 100.0, 200.0, float(ETP_DEFAULTS["hours_per_month"]), 0.01)

    params = (("prod_ca_per_hour", p_ca), ("tickets_per_hour", p_tk),
              ("min_staff", p_min), ("hours_per_month", p_hm))
    etp_hourly, _, etp_monthly = compute_etp_cached(ctx["casc"], TABLES["hourly"], STORES, params)
    etp_avg = etp_monthly.groupby("store_id").etp.mean().sort_values(ascending=False)

    st.markdown('<div class="sec">Effectif moyen nécessaire, par magasin</div>', unsafe_allow_html=True)
    efig = go.Figure(go.Bar(x=etp_avg.index, y=etp_avg.values, marker_color=NAVY,
                            text=etp_avg.round(1), textposition="outside"))
    efig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", yaxis_title=f"ETP moyen / mois (S2 {YEAR})")
    st.plotly_chart(efig, use_container_width=True)

    etp_store = ctx["sel_store"] if ctx["sel_store"] != "Tous" else etp_avg.index[0]
    st.markdown(f'<div class="sec">Répartition sur une journée type — samedi, '
                f'{store_label(etp_store)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="expl" style="box-shadow:none;padding:0 0 8px">Nombre de personnes '
                'nécessaires heure par heure&nbsp;: sert à dimensionner les plannings.</div>',
                unsafe_allow_html=True)
    day_prof = (etp_hourly[etp_hourly.store_id == etp_store]
                .assign(dow=lambda d: pd.to_datetime(d.date).dt.dayofweek)
                .query("dow == 5").groupby("hour").staff.mean())
    pfig = go.Figure(go.Scatter(x=day_prof.index, y=day_prof.values, fill="tozeroy",
                                line=dict(color=ORANGE, width=3)))
    pfig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)", xaxis_title="heure", yaxis_title="personnes")
    st.plotly_chart(pfig, use_container_width=True)


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
        read="Le graphique en cascade part du <b>CA prévu</b> (gauche) et arrive au <b>CA réel</b> "
             "(droite)&nbsp;: chaque marche est une cause, verte si elle ajoute, orange si elle "
             "retranche. Les trois tuiles du bas détaillent l'écart de volume.",
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

    wfig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["CA prévu", "Effet prix", "Effet volume", "Effet mix", "CA réel"],
        y=[agg.ca_prev, agg.effet_prix, agg.effet_volume, agg.effet_mix, 0],
        connector=dict(line=dict(color=LINE)),
        increasing=dict(marker=dict(color=OK)), decreasing=dict(marker=dict(color=ORANGE)),
        totals=dict(marker=dict(color=NAVY))))
    wfig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fff",
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(wfig, use_container_width=True)

    st.markdown('<div class="sec">Détail de l\'écart de volume</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for col, (lab, v) in zip((d1, d2, d3),
                             [("dont effet promotions", drivers.dont_promo),
                              ("dont effet calendaire", drivers.dont_calendaire),
                              ("dont autre (tendance, météo, aléa)", drivers.dont_autre)]):
        col.markdown(f'<div class="kpi"><div class="kpi-l">{lab}</div>'
                     f'<div class="kpi-v" style="color:{OK if v>=0 else ORANGE}">{v:+,.0f} €</div>'
                     f'<div class="kpi-d" style="color:{MUTED}">part de l\'écart de quantité</div>'
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
                                marker_color=NAVY, text=top.importance_pct.round(1),
                                textposition="outside"))
        ifig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_title="part du poids (%)")
        st.plotly_chart(ifig, use_container_width=True)
    with g2:
        st.markdown('<div class="sec">Par famille</div>', unsafe_allow_html=True)
        ff = fam.iloc[::-1]
        ffig = go.Figure(go.Bar(x=ff.importance_pct, y=ff.famille, orientation="h",
                                marker_color=ORANGE, text=ff.importance_pct.round(0),
                                textposition="outside"))
        ffig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#fff",
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_title="part du poids (%)")
        st.plotly_chart(ffig, use_container_width=True)
    st.caption("Diagnostic calculé sur le scénario 2 (météo incluse), sur les 5 ans d'historique.")


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
    st.markdown("### Réglages")
    st.caption("Ces deux choix s'appliquent à toutes les pages.")
    st.radio("Scénario de prévision", [1, 2], key="scenario",
             format_func=lambda s: "1 — Base (calendrier + promos)" if s == 1 else "2 — Base + météo")
    st.selectbox("Magasin", ["Tous"] + STORES.store_id.tolist(), key="sel_store",
                 format_func=store_label)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("**Données synthétiques** — chiffres sans valeur métier.")

pages.run()
