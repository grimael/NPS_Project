"""NPS prediction dashboard for the retention team (challenge brief 4.8).

Four pages, sidebar navigation (D32): "Vue d'ensemble" (live KPI cards +
NPS category donut + quick highlights), "Prédiction individuelle" (pick
an existing "silent" customer — one who never answered the satisfaction
survey, the actual deployment scenario, D8 — or fill in a manual form),
"Performance du modèle" (family comparison + detraction drivers), and
"Équité" (per-group Detractor recall, D27/D28). The prediction page
shows the predicted NPS category, the 3-class probabilities, and the
top-5 SHAP drivers for that specific customer. A bonus panel lets a
retention agent paste a customer verbatim; it explains plainly why
that isn't wired to a live prediction (D31) rather than faking a
result. Every number shown is computed live against the current model/
report files, cache-invalidated by file modification time (D33) — never
a hardcoded copy of a number from docs/decisions.md.

Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from features import add_engineered_features  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# --------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------
# Fixed semantic colors for NPS categories, same order everywhere (never
# cycled) — matches notebooks 01-03 and reports/figures for visual
# consistency across the whole deliverable. Every chart also prints the
# value as text, so meaning never depends on color alone.
NPS_COLORS = {"Detractor": "#C44E52", "Passive": "#DD8452", "Promoter": "#55A868"}

# Chrome (non-data) brand palette, separate from the semantic colors above
# on purpose — brand color never encodes data meaning (dataviz principle:
# color follows the entity, never decoration). Navy/magenta chosen to
# match Artefact's own brief document.
NAVY_DARK = "#0A1B33"
NAVY = "#16284A"
MAGENTA = "#E4067E"
MAGENTA_LIGHT = "#FDEAF4"
BG = "#F6F7FB"
CARD_BG = "#FFFFFF"
TEXT_MUTED = "#69728A"
BORDER = "#E7E9F2"
NEUTRAL_BAR = "#C7CCDA"

FONT_STACK = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
              '"Helvetica Neue", Arial, sans-serif')

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": NAVY_DARK,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": NAVY_DARK,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.titlecolor": NAVY_DARK,
})

RAW_NUMERIC_FIELDS = [
    "Tenure Months", "Monthly Charges", "Total Charges",
    "Number of Referrals", "Number of Dependents", "Age",
]
CATEGORICAL_OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Payment Method": ["Bank transfer (automatic)", "Credit card (automatic)",
                        "Electronic check", "Mailed check"],
    "Paperless Billing": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "Yes"],
    "Online Backup": ["No", "Yes"],
    "Device Protection": ["No", "Yes"],
    "Tech Support": ["No", "Yes"],
    "Streaming TV": ["No", "Yes"],
    "Streaming Movies": ["No", "Yes"],
    "Streaming Music": ["No", "Yes"],
    "Unlimited Data": ["No", "Yes"],
    "Referred a Friend": ["No", "Yes"],
    "Offer": ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"],
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
}
INTERNET_TYPE_OPTIONS = ["DSL", "Fiber Optic", "Cable"]
RAW_FIELDS_ALL = RAW_NUMERIC_FIELDS + list(CATEGORICAL_OPTIONS) + ["Multiple Lines", "Internet Type"]

DRIVER_TABLE = pd.DataFrame([
    {"Driver": "Monthly Charges", "Actionnable": "Oui",
     "Levier": "Révision tarifaire ciblée / remise de rétention pour les clients à charge élevée relativement à leur segment."},
    {"Driver": "Internet Type (Fibre)", "Actionnable": "Partiellement",
     "Levier": "L'accès en soi dépend de l'infrastructure disponible, mais le grief sous-jacent (qualité de service, positionnement prix) est actionnable."},
    {"Driver": "Online Security (absent)", "Actionnable": "Oui",
     "Levier": "Proposer/packager l'add-on de façon proactive plutôt qu'attendre une souscription spontanée."},
    {"Driver": "Contrat mensuel", "Actionnable": "Oui",
     "Levier": "Incitations à la montée vers un contrat annuel (driver secondaire dans plusieurs segments)."},
    {"Driver": "Ancienneté", "Actionnable": "Non directement",
     "Levier": "Ne peut pas être changée ; adapter l'onboarding des nouveaux clients à risque."},
])


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
#
# Every loader below takes an explicit, non-underscore `mtime` (or
# `*_mtime`) argument computed by `_file_signature()`. This is deliberate,
# not decorative: Streamlit's cache_data/cache_resource treat any
# argument whose name starts with "_" as excluded from the cache key
# (the documented escape hatch for passing unhashable objects like a
# fitted Pipeline or a DataFrame). If a cached loader's *only* arguments
# are underscore-prefixed — which every one of these originally was —
# the cache key never changes at all, so the function runs exactly once
# per server process and then serves that first result forever, even
# after `models/final_model.joblib` or `reports/*.csv` are regenerated
# on disk. Threading the file's modification time through as a plain
# argument makes a changed file produce a new cache key, so the app
# picks up fresh results on the next rerun without needing a server
# restart.


def _file_signature(*paths: Path) -> float:
    """Latest mtime across `paths`; 0.0 for any that don't exist yet."""
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return max(times) if times else 0.0


@st.cache_resource
def load_model(mtime: float):
    if not config.FINAL_MODEL_PATH.exists():
        return None
    return joblib.load(config.FINAL_MODEL_PATH)


@st.cache_resource
def load_text_model(mtime: float):
    if not config.FINAL_MODEL_TEXT_PATH.exists():
        return None
    return joblib.load(config.FINAL_MODEL_TEXT_PATH)


@st.cache_data
def load_silent_customers(mtime: float) -> pd.DataFrame:
    labeled = pd.read_parquet(config.LABELED_DATASET_PATH)
    split = pd.read_parquet(config.REALISTIC_SPLIT_PATH)
    merged = labeled.merge(split, on=config.ID_COL)
    return merged[merged["split"] == "silent_test"].reset_index(drop=True)


@st.cache_data
def compute_defaults(_labeled: pd.DataFrame, mtime: float) -> dict:
    defaults = {}
    for col in RAW_NUMERIC_FIELDS:
        defaults[col] = float(_labeled[col].median())
    for col in CATEGORICAL_OPTIONS:
        defaults[col] = _labeled[col].mode(dropna=True).iloc[0]
    defaults["Multiple Lines"] = _labeled["Multiple Lines"].mode(dropna=True).iloc[0]
    non_null_type = _labeled["Internet Type"].dropna()
    defaults["Internet Type"] = non_null_type.mode().iloc[0] if len(non_null_type) else "DSL"
    return defaults


@st.cache_resource(show_spinner=False)
def build_shap_explainer(_pipeline, _background_raw: pd.DataFrame, mtime: float):
    import shap
    prep = _pipeline.named_steps["prep"]
    model = _pipeline.named_steps["model"]
    feature_columns = get_feature_columns(_pipeline)
    background_engineered = add_engineered_features(_background_raw)[feature_columns]
    background_t = prep.transform(background_engineered)
    feature_names = prep.get_feature_names_out()
    explainer = shap.Explainer(model.predict_proba, background_t)
    return explainer, feature_names


@st.cache_data(show_spinner=False)
def compute_headline_metrics(mtime: float) -> dict:
    """Live-evaluated on silent_test, never hardcoded, so the KPI row can
    never silently drift from whatever model is actually loaded."""
    from evaluate import evaluate
    from features import build_feature_matrix

    artifact = load_model(mtime)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    labeled = pd.read_parquet(config.LABELED_DATASET_PATH)
    split = pd.read_parquet(config.REALISTIC_SPLIT_PATH)
    X = build_feature_matrix(labeled, include_geo=False)
    merged = X.merge(labeled[[config.ID_COL, config.TARGET_COL]], on=config.ID_COL).merge(split, on=config.ID_COL)
    test_df = merged[merged["split"] == "silent_test"]

    y_test = test_df[config.TARGET_COL].astype(str)
    y_pred = pipeline.predict(test_df[feature_columns])
    y_proba = pipeline.predict_proba(test_df[feature_columns])
    results = evaluate(y_test, y_pred, y_proba)

    return {
        "n_total": len(labeled),
        "n_silent": len(test_df),
        "n_respondents": (merged["split"] == "respondent_train").sum(),
        "n_features": len(feature_columns),
        "qwk": results["qwk"],
        "macro_f1": results["macro_f1"],
        "balanced_accuracy": results["balanced_accuracy"],
        "recall_detractor": results["recall_per_class"]["Detractor"],
    }


@st.cache_data
def load_model_comparison(mtime: float):
    path = config.REPORTS_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_fairness_data(mtime: float):
    audit_path = config.REPORTS_DIR / "fairness_audit.csv"
    gaps_path = config.REPORTS_DIR / "fairness_audit_gaps.csv"
    if not (audit_path.exists() and gaps_path.exists()):
        return None, None
    audit = pd.read_csv(audit_path)
    gaps = pd.read_csv(gaps_path)
    return audit[audit["model"] == "no_geo"], gaps


def find_logo_path() -> Path | None:
    for name in ("logo.svg", "logo.png", "logo.jpg", "logo.jpeg"):
        candidate = ASSETS_DIR / name
        if candidate.exists():
            return candidate
    return None


def prettify_feature_name(name: str) -> str:
    name = name.split("__", 1)[-1]
    return name.replace("_", ": ", 1) if "_" in name and not name.startswith(("Tenure", "Number")) else name


# --------------------------------------------------------------------------
# Feature engineering helpers
# --------------------------------------------------------------------------

def build_raw_row(values: dict) -> pd.DataFrame:
    """One raw-field dict -> a one-row DataFrame with every column
    `add_engineered_features` expects, filling structurally-dependent
    fields the same way the training pipeline's source tables do.
    """
    row = dict(values)
    row["Under 30"] = "Yes" if row["Age"] < 30 else "No"

    if row["Phone Service"] == "No":
        row["Multiple Lines"] = "No phone service"

    if row["Internet Service"] == "No":
        row["Internet Type"] = None
        for col in ["Online Security", "Online Backup", "Device Protection",
                    "Tech Support", "Streaming TV", "Streaming Movies"]:
            row[col] = "No internet service"

    if row.get("Offer") == "No Offer":
        row["Offer"] = None

    return pd.DataFrame([row])


def get_feature_columns(pipeline) -> list:
    prep = pipeline.named_steps["prep"]
    return prep.transformers[0][2] + prep.transformers[1][2]


def predict_one(pipeline, raw_row: pd.DataFrame):
    engineered = add_engineered_features(raw_row)
    feature_columns = get_feature_columns(pipeline)
    X = engineered[feature_columns]
    proba = pipeline.predict_proba(X)[0]
    pred = pipeline.predict(X)[0]
    return pred, dict(zip(pipeline.named_steps["model"].classes_, proba)), X


# --------------------------------------------------------------------------
# Chart styling helpers
# --------------------------------------------------------------------------

def style_ax(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)


def show_probability_bars(proba: dict, predicted: str) -> None:
    categories = config.NPS_CATEGORIES
    values = [proba[c] for c in categories]
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    bars = ax.barh(categories, values, color=[NPS_COLORS[c] for c in categories], height=0.62)
    for bar, cat, val in zip(bars, categories, values):
        weight = "bold" if cat == predicted else "normal"
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                va="center", fontsize=10, fontweight=weight, color=NAVY_DARK)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probabilité")
    style_ax(ax)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)


def show_shap_drivers(pipeline, X_row: pd.DataFrame, predicted: str, background_raw: pd.DataFrame) -> None:
    explainer, feature_names = build_shap_explainer(
        pipeline, background_raw, _file_signature(config.FINAL_MODEL_PATH))
    prep = pipeline.named_steps["prep"]
    X_t = pd.DataFrame(prep.transform(X_row), columns=feature_names)

    with st.spinner("Calcul des drivers SHAP..."):
        shap_values = explainer(X_t)

    class_idx = config.NPS_CATEGORIES.index(predicted)
    values = shap_values.values[0, :, class_idx]
    order = np.argsort(-np.abs(values))[:5]
    top_names = [prettify_feature_name(feature_names[i]) for i in order]
    top_values = values[order]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    colors = [NPS_COLORS[predicted] if v > 0 else NEUTRAL_BAR for v in top_values]
    bars = ax.barh(top_names[::-1], top_values[::-1], color=colors[::-1], height=0.62)
    for bar, val in zip(bars, top_values[::-1]):
        offset = 0.001 if val >= 0 else -0.001
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=9, color=NAVY_DARK)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel(f"Impact SHAP sur P({predicted})")
    style_ax(ax)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)
    st.caption(
        "Barres colorées = poussent vers la classe prédite ; barres grises = "
        "poussent contre. Noms de features tels que produits par le "
        "préprocesseur (one-hot + standardisation) — voir docs/decisions.md D22."
    )


def show_imputed_fields_note(values: dict, defaults: dict) -> None:
    unchanged = [f for f in RAW_FIELDS_ALL if f in values and f in defaults
                 and values[f] == defaults[f]]
    if unchanged:
        st.info(
            "Champs laissés à leur valeur par défaut (imputée à partir de la "
            "médiane/mode du jeu d'entraînement) : " + ", ".join(unchanged)
        )


# --------------------------------------------------------------------------
# UI components
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1180px; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* Sidebar shell — narrow, compact */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 100%);
    min-width: 215px !important;
    max-width: 215px !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
div[data-testid="stSidebarHeader"] {{ padding: 0 !important; min-height: 0 !important; height: 0.5rem !important; }}
section[data-testid="stSidebarUserContent"] {{ padding-top: 0 !important; }}
section[data-testid="stSidebar"] .block-container {{ padding: 0 0.6rem !important; }}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{ gap: 0.15rem; }}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{ margin: 0 !important; }}
section[data-testid="stSidebar"] div.element-container {{ margin: 0 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); margin: 6px 0; }}

/* Sidebar logo — no card, sits directly on the navy background, sized to
   fill most of the sidebar width so it reads clearly at a glance. */
.sidebar-logo-wrap {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 4px 0 4px;
    margin: 0;
}}
.sidebar-logo-img {{ max-width: 96%; max-height: 90px; width: 100%; height: auto; object-fit: contain; }}
.sidebar-wordmark {{
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: #FFFFFF;
}}
.sidebar-caption {{
    color: #93A0BD;
    font-size: 0.72rem;
    line-height: 1.45;
    padding: 0 4px 14px 4px;
}}

/* Nav buttons (built from st.button, one per page) */
section[data-testid="stSidebar"] .stButton {{ margin-bottom: 4px; }}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.3;
    min-height: 0;
    border: 1px solid transparent;
    transition: background 0.15s ease;
}}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    background: transparent;
    color: #C3CBE0;
}}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
    background: rgba(255,255,255,0.08);
    color: #FFFFFF;
    border-color: rgba(255,255,255,0.12);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: {MAGENTA};
    color: #FFFFFF;
    box-shadow: 0 3px 8px rgba(228, 6, 126, 0.32);
}}

/* Main content page header: project theme (eyebrow + big title, taken
   verbatim from the Artefact brief's own title), then the specific
   page's own heading + description underneath. */
.project-banner {{
    font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.1em; color: {MAGENTA}; margin-bottom: 5px;
}}
.theme-title {{
    color: {NAVY_DARK}; font-size: 1.45rem; font-weight: 800;
    line-height: 1.3; margin-bottom: 16px; max-width: 760px;
}}
.page-title {{
    font-size: 1.05rem; font-weight: 700; color: {MAGENTA}; margin: 0 0 3px 0;
    padding-top: 12px; border-top: 1px solid {BORDER};
    text-transform: uppercase; letter-spacing: 0.03em;
}}
.page-subtitle {{ color: {TEXT_MUTED}; font-size: 0.86rem; margin-bottom: 18px; }}

.highlight-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 24px 0; }}
.highlight-card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 2px 10px rgba(16, 27, 51, 0.05);
}}
.highlight-label {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: {TEXT_MUTED}; margin-bottom: 6px;
}}
.highlight-value {{ font-size: 1.05rem; font-weight: 800; color: {NAVY_DARK}; line-height: 1.3; }}
.highlight-context {{ font-size: 0.76rem; color: {TEXT_MUTED}; margin-top: 5px; line-height: 1.35; }}

.kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 20px; }}
.kpi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-top: 3px solid {MAGENTA};
    border-radius: 10px;
    padding: 10px 12px;
    box-shadow: 0 2px 8px rgba(16, 27, 51, 0.05);
    min-width: 0;
}}
.kpi-label {{
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: {TEXT_MUTED}; margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.kpi-value {{ font-size: 1.15rem; font-weight: 800; color: {NAVY_DARK}; line-height: 1.1; }}
.kpi-context {{
    font-size: 0.66rem; color: {TEXT_MUTED}; margin-top: 3px; line-height: 1.25;
}}

.section-eyebrow {{
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: {MAGENTA}; margin-bottom: 2px;
}}
.section-title {{ font-size: 1.25rem; font-weight: 700; color: {NAVY_DARK}; margin-top: 0; margin-bottom: 10px; }}

.pred-badge {{
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 1.35rem;
    font-weight: 800;
    color: {NAVY_DARK};
}}

.callout {{
    background: {MAGENTA_LIGHT};
    border-left: 4px solid {MAGENTA};
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: {NAVY_DARK};
    margin: 10px 0 18px 0;
}}

div[data-testid="stMetric"] {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 14px;
}}
</style>
""", unsafe_allow_html=True)


PAGES = {
    "Vue d'ensemble": "Indicateurs clés et vue globale du projet.",
    "Prédiction individuelle": "Client existant ou saisie manuelle, catégorie NPS prédite et drivers SHAP.",
    "Performance du modèle": "Comparaison de familles de modèles et drivers de détraction actionnables.",
    "Équité": "Rappel Detractor par groupe démographique, écarts et points d'escalade.",
}
THEME_TITLE = "Customer NPS Prediction for a Telecom Operator"


def render_sidebar_logo() -> None:
    logo_path = find_logo_path()
    if logo_path is not None:
        import base64
        encoded = base64.b64encode(logo_path.read_bytes()).decode()
        mime = "image/svg+xml" if logo_path.suffix == ".svg" else f"image/{logo_path.suffix.lstrip('.')}"
        inner = f'<img class="sidebar-logo-img" src="data:{mime};base64,{encoded}" />'
    else:
        inner = '<div class="sidebar-wordmark">ARTEFACT</div>'
    st.sidebar.markdown(f'<div class="sidebar-logo-wrap">{inner}</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    render_sidebar_logo()

    if "page" not in st.session_state:
        st.session_state.page = next(iter(PAGES))

    for page_name in PAGES:
        is_active = st.session_state.page == page_name
        if st.sidebar.button(page_name, key=f"nav_{page_name}", width="stretch",
                              type="primary" if is_active else "secondary"):
            st.session_state.page = page_name
            st.rerun()

    return st.session_state.page


def render_page_header(page_name: str) -> None:
    st.markdown(
        '<div class="project-banner">Challenge Artefact</div>'
        f'<div class="theme-title">{THEME_TITLE}</div>'
        f'<div class="page-title">{page_name}</div>'
        f'<div class="page-subtitle">{PAGES[page_name]}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_cards(metrics: dict) -> None:
    cards = [
        ("Clients", f"{metrics['n_total']:,}".replace(",", " "),
         f"{metrics['n_silent']:,}".replace(",", " ") + " silencieux (85%)"),
        ("QWK", f"{metrics['qwk']:.3f}", "Quadratic Weighted Kappa"),
        ("Rappel Detractor", f"{metrics['recall_detractor']:.0%}", "Population silencieuse"),
        ("Balanced acc.", f"{metrics['balanced_accuracy']:.0%}", f"Macro-F1 {metrics['macro_f1']:.2f}"),
        ("Features", f"{metrics['n_features']}", "mord.LogisticAT (D21)"),
    ]
    html = '<div class="kpi-row">'
    for label, value, context in cards:
        html += (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div>'
                 f'<div class="kpi-context">{context}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(eyebrow: str, title: str) -> None:
    st.markdown(f'<div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div>',
                unsafe_allow_html=True)


def render_nps_distribution(labeled_df: pd.DataFrame) -> None:
    """Donut chart: NPS category mix, with the total client count as a
    center stat — a compact, direct-labeled alternative to a legend-only
    pie, per the project's dataviz conventions (fixed category order/
    colors, values always printed, never color-only)."""
    counts = labeled_df[config.TARGET_COL].value_counts().reindex(config.NPS_CATEGORIES)
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    wedges, _ = ax.pie(
        counts.values,
        colors=[NPS_COLORS[c] for c in config.NPS_CATEGORIES],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.40, edgecolor="#FFFFFF", linewidth=1.3),
        radius=1.0,
    )

    for wedge, cat in zip(wedges, config.NPS_CATEGORIES):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        share = counts[cat] / total
        label_r = 1.30
        ax.annotate(
            f"{cat}\n{int(counts[cat]):,}".replace(",", " ") + f"  ({share:.0%})",
            xy=(x * 0.80, y * 0.80), xytext=(x * label_r, y * label_r),
            ha="center", va="center", fontsize=5.2, fontweight="bold", color=NAVY_DARK,
            arrowprops=dict(arrowstyle="-", color=BORDER, lw=0.7,
                             shrinkA=0, shrinkB=0,
                             connectionstyle="arc3,rad=0"),
        )

    ax.text(0, 0.06, f"{total:,}".replace(",", " "), ha="center", va="center",
            fontsize=10, fontweight="800", color=NAVY_DARK)
    ax.text(0, -0.16, "clients", ha="center", va="center", fontsize=5.2, color=TEXT_MUTED)

    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.55, 1.55)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    st.pyplot(fig, transparent=True, width="content")
    plt.close(fig)


def render_highlight_cards(metrics: dict, gaps: pd.DataFrame | None) -> None:
    dep_gap_text = "voir onglet Équité"
    if gaps is not None:
        dep = gaps[gaps["group"] == "Dependents"]
        if len(dep):
            dep_gap_text = f"{dep['point'].iloc[0]:+.0%} (résiduel, D28)"

    cards = [
        ("Meilleur modèle", "mord.LogisticAT",
         f"QWK {metrics['qwk']:.3f} — seul modèle stable entre les deux splits (D14)"),
        ("Driver n°1 de détraction", "Monthly Charges",
         "Dominant dans presque tous les segments testés (D22/D23)"),
        ("Écart d'équité", dep_gap_text,
         "Rappel Detractor, groupe Dependents — point d'escalade (D27/D28)"),
    ]
    html = '<div class="highlight-row">'
    for label, value, context in cards:
        html += (f'<div class="highlight-card"><div class="highlight-label">{label}</div>'
                 f'<div class="highlight-value">{value}</div>'
                 f'<div class="highlight-context">{context}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================================
# App
# ==========================================================================

st.set_page_config(
    page_title="NPS Prediction — Artefact",
    page_icon=str(find_logo_path()) if find_logo_path() else "◆",
    layout="wide",
)
inject_css()
page = render_sidebar()
render_page_header(page)

model_mtime = _file_signature(config.FINAL_MODEL_PATH)
pipeline_artifact = load_model(model_mtime)
if pipeline_artifact is None:
    st.error(
        "**Aucun modèle trouvé** (`models/final_model.joblib` est absent).\n\n"
        "Entraînez-le d'abord depuis la racine du projet :\n"
        "```\npython -c \"import sys; sys.path.insert(0, 'src'); "
        "from train import main; main()\"\n```\n"
        "(ne pas lancer `python src/train.py` directement — voir "
        "docs/decisions.md pour le bug de sérialisation que ça provoque)."
    )
    st.stop()

pipeline = pipeline_artifact["pipeline"]
labeled_df = pd.read_parquet(config.LABELED_DATASET_PATH)
defaults = compute_defaults(labeled_df, _file_signature(config.LABELED_DATASET_PATH))
background_raw = labeled_df.sample(n=min(50, len(labeled_df)), random_state=config.SEED)

headline_metrics = compute_headline_metrics(model_mtime)

# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------
if page == "Vue d'ensemble":
    render_kpi_cards(headline_metrics)

    section_header("Portrait du dataset", "Répartition des clients par catégorie NPS")
    render_nps_distribution(labeled_df)
    st.caption(
        f"{len(labeled_df):,}".replace(",", " ") + " clients (dataset IBM Telco Customer Churn "
        "v11.1.3+) — répartition dérivée du Satisfaction Score, voir docs/decisions.md D6."
    )

    st.divider()
    section_header("Points clés", "À retenir en un coup d'œil")
    _, overview_gaps = load_fairness_data(_file_signature(
        config.REPORTS_DIR / "fairness_audit.csv", config.REPORTS_DIR / "fairness_audit_gaps.csv"))
    render_highlight_cards(headline_metrics, overview_gaps)
    st.caption(
        "Détail complet : onglets Performance du modèle et Équité. Journal de décisions "
        "complet dans docs/decisions.md."
    )

# --------------------------------------------------------------------------
# Page: Prediction
# --------------------------------------------------------------------------
elif page == "Prédiction individuelle":
    tab_existing, tab_manual = st.tabs(["Client existant", "Saisie manuelle"])

    selected_raw_row = None
    selected_values = None

    with tab_existing:
        silent_customers = load_silent_customers(
            _file_signature(config.LABELED_DATASET_PATH, config.REALISTIC_SPLIT_PATH))
        customer_id = st.selectbox(
            "Client (population silencieuse — n'a jamais répondu à l'enquête)",
            options=silent_customers[config.ID_COL].tolist(),
        )
        customer_row = silent_customers[silent_customers[config.ID_COL] == customer_id].iloc[[0]]

        with st.expander("Profil du client", expanded=True):
            preview_cols = ["Tenure Months", "Contract", "Monthly Charges", "Internet Service",
                             "Payment Method", "Senior Citizen", "Partner", "Dependents"]
            st.dataframe(customer_row[preview_cols], hide_index=True, width="stretch")
            st.caption(
                f"Valeur réelle de Satisfaction Score dans ce jeu de démonstration : "
                f"{customer_row['Satisfaction Score'].iloc[0]} -> "
                f"**{customer_row[config.TARGET_COL].iloc[0]}** (connue uniquement ici "
                f"car IBM simule des clients silencieux ; un vrai client silencieux "
                f"n'aurait pas répondu à l'enquête — jamais utilisée comme feature, "
                f"affichée à titre de validation uniquement)."
            )

        if st.button("Prédire", key="predict_existing", type="primary"):
            selected_raw_row = customer_row
            selected_values = "existing"

    with tab_manual:
        with st.form("manual_form"):
            st.markdown("**Profil client**")
            c1, c2, c3, c4 = st.columns(4)
            gender = c1.selectbox("Genre", CATEGORICAL_OPTIONS["Gender"],
                                   index=CATEGORICAL_OPTIONS["Gender"].index(defaults["Gender"]))
            age = c2.number_input("Âge", min_value=18, max_value=100, value=int(defaults["Age"]))
            senior = c3.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"],
                                   index=CATEGORICAL_OPTIONS["Senior Citizen"].index(defaults["Senior Citizen"]))
            partner = c4.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"],
                                    index=CATEGORICAL_OPTIONS["Partner"].index(defaults["Partner"]))
            c5, c6, c7 = st.columns(3)
            dependents = c5.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"],
                                       index=CATEGORICAL_OPTIONS["Dependents"].index(defaults["Dependents"]))
            n_dependents = c6.number_input("Nombre de personnes à charge", min_value=0, max_value=15,
                                            value=int(defaults["Number of Dependents"]))
            referred = c7.selectbox("A parrainé un ami", CATEGORICAL_OPTIONS["Referred a Friend"],
                                     index=CATEGORICAL_OPTIONS["Referred a Friend"].index(defaults["Referred a Friend"]))
            n_referrals = st.number_input("Nombre de parrainages", min_value=0, max_value=20,
                                           value=int(defaults["Number of Referrals"]))

            st.markdown("**Contrat & facturation**")
            c1, c2, c3 = st.columns(3)
            tenure = c1.number_input("Ancienneté (mois)", min_value=0, max_value=100,
                                      value=int(defaults["Tenure Months"]))
            contract = c2.selectbox("Contrat", CATEGORICAL_OPTIONS["Contract"],
                                     index=CATEGORICAL_OPTIONS["Contract"].index(defaults["Contract"]))
            payment = c3.selectbox("Moyen de paiement", CATEGORICAL_OPTIONS["Payment Method"],
                                    index=CATEGORICAL_OPTIONS["Payment Method"].index(defaults["Payment Method"]))
            c4, c5, c6 = st.columns(3)
            paperless = c4.selectbox("Facturation dématérialisée", CATEGORICAL_OPTIONS["Paperless Billing"],
                                      index=CATEGORICAL_OPTIONS["Paperless Billing"].index(defaults["Paperless Billing"]))
            monthly_charges = c5.number_input("Charges mensuelles ($)", min_value=0.0, max_value=200.0,
                                               value=float(defaults["Monthly Charges"]), step=1.0)
            total_charges = c6.number_input("Charges totales ($)", min_value=0.0, max_value=10000.0,
                                             value=float(defaults["Total Charges"]), step=10.0,
                                             help=f"Estimation typique : ancienneté × charges mensuelles ≈ "
                                                  f"{tenure * monthly_charges:,.0f} $")
            offer = st.selectbox("Offre promotionnelle", CATEGORICAL_OPTIONS["Offer"],
                                  index=CATEGORICAL_OPTIONS["Offer"].index(defaults["Offer"])
                                  if defaults["Offer"] in CATEGORICAL_OPTIONS["Offer"] else 0)

            st.markdown("**Services**")
            c1, c2 = st.columns(2)
            phone_service = c1.selectbox("Téléphone", CATEGORICAL_OPTIONS["Phone Service"],
                                          index=CATEGORICAL_OPTIONS["Phone Service"].index(defaults["Phone Service"]))
            multiple_lines = "No"
            if phone_service == "Yes":
                multiple_lines = c2.selectbox("Lignes multiples", ["No", "Yes"],
                                               index=["No", "Yes"].index(defaults["Multiple Lines"])
                                               if defaults["Multiple Lines"] in ["No", "Yes"] else 0)
            else:
                c2.text_input("Lignes multiples", value="No phone service", disabled=True)

            internet_service = st.selectbox("Internet", CATEGORICAL_OPTIONS["Internet Service"],
                                             index=CATEGORICAL_OPTIONS["Internet Service"].index(defaults["Internet Service"]))
            internet_type = None
            online_security = online_backup = device_protection = tech_support = "No internet service"
            streaming_tv = streaming_movies = "No internet service"
            if internet_service != "No":
                c1, c2 = st.columns(2)
                internet_type = c1.selectbox("Type de connexion", INTERNET_TYPE_OPTIONS,
                                              index=INTERNET_TYPE_OPTIONS.index(defaults["Internet Type"])
                                              if defaults["Internet Type"] in INTERNET_TYPE_OPTIONS else 0)
                c1, c2, c3, c4 = st.columns(4)
                online_security = c1.selectbox("Sécurité en ligne", ["No", "Yes"],
                                                index=["No", "Yes"].index(defaults["Online Security"]))
                online_backup = c2.selectbox("Sauvegarde en ligne", ["No", "Yes"],
                                              index=["No", "Yes"].index(defaults["Online Backup"]))
                device_protection = c3.selectbox("Protection appareil", ["No", "Yes"],
                                                  index=["No", "Yes"].index(defaults["Device Protection"]))
                tech_support = c4.selectbox("Support technique", ["No", "Yes"],
                                             index=["No", "Yes"].index(defaults["Tech Support"]))
                c1, c2 = st.columns(2)
                streaming_tv = c1.selectbox("Streaming TV", ["No", "Yes"],
                                             index=["No", "Yes"].index(defaults["Streaming TV"]))
                streaming_movies = c2.selectbox("Streaming Films", ["No", "Yes"],
                                                 index=["No", "Yes"].index(defaults["Streaming Movies"]))

            c1, c2 = st.columns(2)
            streaming_music = c1.selectbox("Streaming Musique", ["No", "Yes"],
                                            index=["No", "Yes"].index(defaults["Streaming Music"]))
            unlimited_data = c2.selectbox("Données illimitées", ["No", "Yes"],
                                           index=["No", "Yes"].index(defaults["Unlimited Data"]))

            submitted = st.form_submit_button("Prédire", type="primary")

        if submitted:
            manual_values = {
                "Gender": gender, "Age": age, "Senior Citizen": senior, "Partner": partner,
                "Dependents": dependents, "Number of Dependents": n_dependents,
                "Referred a Friend": referred, "Number of Referrals": n_referrals,
                "Tenure Months": tenure, "Contract": contract, "Payment Method": payment,
                "Paperless Billing": paperless, "Monthly Charges": monthly_charges,
                "Total Charges": total_charges, "Offer": offer,
                "Phone Service": phone_service, "Multiple Lines": multiple_lines,
                "Internet Service": internet_service, "Internet Type": internet_type,
                "Online Security": online_security, "Online Backup": online_backup,
                "Device Protection": device_protection, "Tech Support": tech_support,
                "Streaming TV": streaming_tv, "Streaming Movies": streaming_movies,
                "Streaming Music": streaming_music, "Unlimited Data": unlimited_data,
            }
            selected_raw_row = build_raw_row(manual_values)
            selected_values = manual_values

    if selected_raw_row is not None:
        st.divider()
        pred, proba, X_row = predict_one(pipeline, selected_raw_row)

        if isinstance(selected_values, dict):
            show_imputed_fields_note(selected_values, defaults)

        col1, col2 = st.columns([1, 1])
        with col1:
            section_header("Résultat", "Prédiction")
            badge_color = NPS_COLORS[pred]
            st.markdown(
                f"<div class='pred-badge' style='background-color:{badge_color}18; "
                f"border-left:6px solid {badge_color};'>{pred}</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            show_probability_bars(proba, pred)

        with col2:
            section_header("Interprétabilité", "Top 5 drivers (SHAP)")
            show_shap_drivers(pipeline, X_row, pred, background_raw)

        st.divider()
        section_header("Bonus", "Impact d'un verbatim client")
        verbatim_text = st.text_area(
            "Coller un verbatim (note d'interaction, avis, extrait de chat)",
            placeholder="Ex: \"J'ai eu des soucis de coupures internet régulières...\"",
        )
        if st.button("Recalculer avec ce verbatim"):
            text_model = load_text_model(_file_signature(config.FINAL_MODEL_TEXT_PATH))
            if text_model is None:
                st.info(
                    "Le modèle enrichi de texte n'a pas été produit — voir "
                    "`docs/decisions.md` D31."
                )
            else:
                st.markdown(
                    "<div class='callout'><b>Fonctionnalité désactivée délibérément.</b> "
                    "Le modèle texte+tabulaire existe (<code>models/final_model_text.joblib</code>) "
                    "mais son gain apparent (QWK 0.273 → 0.875) s'est révélé très majoritairement "
                    "un artefact : les verbatims sont générés à partir du vrai label, donc le "
                    "modèle re-détecte en grande partie le ton qu'on lui a demandé d'écrire, pas "
                    "un signal indépendant. Le brancher ici tromperait quiconque colle un vrai "
                    "verbatim. Détail complet et test de robustesse (sous-ensemble discordant) "
                    "dans <code>docs/decisions.md</code> D31.</div>",
                    unsafe_allow_html=True,
                )

# --------------------------------------------------------------------------
# Page: Model performance
# --------------------------------------------------------------------------
elif page == "Performance du modèle":
    section_header("Sélection de modèle", "Comparaison de familles de modèles (split réaliste)")
    comparison = load_model_comparison(_file_signature(config.REPORTS_DIR / "model_comparison.csv"))
    if comparison is not None:
        realistic = comparison[comparison["split"] == "realistic"].sort_values("qwk", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        colors = [MAGENTA if "mord" in m.lower() or "ordinal" in m.lower() else NEUTRAL_BAR
                  for m in realistic["model"]]
        bars = ax.barh(realistic["model"], realistic["qwk"], color=colors, height=0.6)
        for bar, val in zip(bars, realistic["qwk"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                    va="center", fontsize=9, color=NAVY_DARK, fontweight="bold")
        ax.set_xlabel("Quadratic Weighted Kappa (QWK)")
        style_ax(ax)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        st.caption(
            "Split réaliste (15% répondants biaisés / 85% silencieux), le protocole "
            "principal d'évaluation — voir docs/decisions.md D8/D13/D14. `mord.LogisticAT` "
            "est le seul modèle qui ne se dégrade pas entre split aléatoire et réaliste."
        )
    else:
        st.info("`reports/model_comparison.csv` introuvable — lancer `notebooks/02_modeling.ipynb`.")

    section_header("Interprétation", "Drivers de détraction : actionnables vs non")
    st.dataframe(DRIVER_TABLE, hide_index=True, width="stretch")
    st.caption(
        "Même trio de drivers dominant dans presque tous les segments testés "
        "(ancienneté, type de contrat, type d'accès internet) — voir "
        "docs/decisions.md D22/D23 et notebooks/03_interpretation.ipynb."
    )

# --------------------------------------------------------------------------
# Page: Fairness
# --------------------------------------------------------------------------
elif page == "Équité":
    section_header("Audit d'équité", "Rappel Detractor par groupe démographique")
    audit, gaps = load_fairness_data(_file_signature(
        config.REPORTS_DIR / "fairness_audit.csv", config.REPORTS_DIR / "fairness_audit_gaps.csv"))
    if audit is not None:
        audit = audit.copy()
        audit["label"] = audit["group"] + " = " + audit["value"].astype(str)
        audit = audit.sort_values("recall_Detractor")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        bars = ax.barh(audit["label"], audit["recall_Detractor"], color=NPS_COLORS["Detractor"], height=0.6)
        for bar, val, n in zip(bars, audit["recall_Detractor"], audit["n"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.0%} (n={n})",
                    va="center", fontsize=9, color=NAVY_DARK)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Rappel Detractor")
        style_ax(ax)
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        gap_dependents = gaps[gaps["group"] == "Dependents"]
        gap_gender = gaps[gaps["group"] == "Gender"]
        if len(gap_dependents) and len(gap_gender) and len(gap_dependents) and len(gap_gender):
            dep_point = gap_dependents["point"].iloc[0]
            dep_lower = gap_dependents["lower"].iloc[0]
            dep_upper = gap_dependents["upper"].iloc[0]
            st.markdown(
                f"<div class='callout'><b>Point d'escalade non résolu.</b> Écart de rappel "
                f"Detractor sur <code>Dependents</code> : {dep_point:+.1%} "
                f"(IC 95% [{dep_lower:+.1%}, {dep_upper:+.1%}]) — réduit de 19 à 12 points après "
                f"retrait de trois features corrélées (D28, sans coût de QWK), mais pas "
                f"entièrement résolu. Signalé pour escalade Customer Experience / Legal avant "
                f"toute mise en production réelle — voir docs/decisions.md D27/D28.</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "silent_test (n=5987), modèle sans features géographiques (D11), intervalles de "
            "confiance bootstrap à 95% — voir `src/fairness_audit.py` et docs/decisions.md D27."
        )
    else:
        st.info("`reports/fairness_audit.csv` introuvable — lancer `python src/fairness_audit.py`.")
