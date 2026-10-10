"""TelcoPulse AI - Enterprise Customer Retention & NPS Intelligence Platform.

Main entry point for the Streamlit dashboard application.
Modularized architecture:
- app/styles.py: design system, modern CSS tokens, responsive typography
- app/icons.py: minimalist SVG vector icons
- app/utils.py: cached loaders, prediction service, SHAP explainer
- app/views/: overview, prediction, simulator, campaigns, monitoring
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from app.icons import icon_svg
from app.styles import get_custom_css, render_page_hero
from app.utils import (
    get_file_signature,
    load_datasets,
    load_trained_model,
)
from app.views.campaigns import render_campaigns_view
from app.views.monitoring import render_monitoring_view
from app.views.overview import render_overview_view
from app.views.prediction import render_prediction_view
from app.views.simulator import render_simulator_view
from src.config import FINAL_MODEL_PATH

# 1. Page Configuration
st.set_page_config(
    page_title="TelcoPulse AI — Plateforme d'Intelligence Client & Rétention",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Modern CSS Styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# 3. Model & Data Loaders
mtime = get_file_signature(FINAL_MODEL_PATH)
model_artifact = load_trained_model(mtime)
labeled_df, splits_df = load_datasets(mtime)

if model_artifact is None or labeled_df is None or splits_df is None:
    st.error(
        "**Initialisation du pipeline en cours...** "
        "Si ce message persiste, exécutez `python src/train.py`."
    )
    st.stop()

pipeline = model_artifact["pipeline"]

# 4. Navigation Menu Definitions (No tacky emojis)
NAVIGATION_PAGES = {
    "overview": (
        "Vue d'ensemble",
        "Indicateurs clés du parc, sentiment NPS et volume de revenu exposé au churn",
        "Synthèse Exécutive",
    ),
    "prediction": (
        "Diagnostic & Scoring",
        "Analyse probabiliste individuelle et drivers d'influence locaux (SHAP)",
        "Scoring Client",
    ),
    "simulator": (
        "Simulateur Prescriptif",
        "Laboratoire de test d'impact immédiat des offres de fidélisation (What-If)",
        "Aide à la Décision",
    ),
    "campaigns": (
        "Campagnes & ROI",
        "Segmentation de cohorte, calculateur de gain financier ($ MRR) et export de listes",
        "Pilotage Business",
    ),
    "monitoring": (
        "Équité & MLOps",
        "Audit d'équité démographique et suivi de la dérive de distribution (PSI)",
        "Gouvernance & MLOps",
    ),
}

# 5. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div class="brand-title-wrap">
            <div class="brand-logo-icon">
                {icon_svg("pulse", size=18, color="#FFFFFF")}
            </div>
            <div>
                <div class="brand-name">TelcoPulse AI</div>
                <div class="brand-tag">Retention Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "overview"

    for page_key, (page_label, _, _) in NAVIGATION_PAGES.items():
        is_selected = (st.session_state.current_page == page_key)
        if st.sidebar.button(page_label, key=f"nav_{page_key}", type="primary" if is_selected else "secondary"):
            st.session_state.current_page = page_key
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color:#94A3B8; font-size: 0.72rem; line-height: 1.5; padding: 0.2rem 0.5rem;">
        <span style="color:#FFFFFF; font-weight:700;">Système en Production v2.0</span><br>
        • Modèle : Ordinal Logistic (mord)<br>
        • Parc Silencieux : 5 987 clients<br>
        • Validation : Split 3-voies étanche<br>
        • Reproductibilité : Seed 42
    </div>
    """, unsafe_allow_html=True)

# 6. Page Hero Header & View Routing
current_view = st.session_state.current_page
page_label, page_sub, badge = NAVIGATION_PAGES[current_view]

render_page_hero(title=page_label, subtitle=page_sub, badge_text=badge)

if current_view == "overview":
    render_overview_view(labeled_df, splits_df)
elif current_view == "prediction":
    render_prediction_view(pipeline, labeled_df, splits_df)
elif current_view == "simulator":
    render_simulator_view(pipeline, labeled_df, splits_df)
elif current_view == "campaigns":
    render_campaigns_view(pipeline, labeled_df, splits_df)
elif current_view == "monitoring":
    render_monitoring_view(pipeline, labeled_df, splits_df)
