"""PulseRetention AI - Enterprise Customer Retention & NPS Intelligence Platform.

Main entry point for the Streamlit dashboard application.
Modularized architecture:
- app/styles.py: design system, modern CSS tokens, responsive typography
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

from app.styles import get_custom_css
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
    page_title="PulseRetention AI — Plateforme d'Intelligence Client & Rétention",
    page_icon="⚡",
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
        "⚠️ **Artefacts de modèle ou données introuvables.**\n\n"
        "Veuillez exécuter le pipeline d'entraînement depuis la racine du projet :\n"
        "```powershell\n"
        "python src/make_dataset.py\n"
        "python src/build_target.py\n"
        "python src/make_splits.py\n"
        "python src/train.py\n"
        "```"
    )
    st.stop()

pipeline = model_artifact["pipeline"]

# 4. Navigation Menu Definitions
NAVIGATION_PAGES = {
    "overview": ("📊 Vue d'ensemble", "Indicateurs clés du parc, répartition NPS et volume à risque"),
    "prediction": ("🔍 Scoring & Diagnostic", "Analyse individuelle par client et facteurs d'influence SHAP"),
    "simulator": ("🛠️ Simulateur Prescriptif", "Test d'impact immédiat des offres de fidélisation (What-If)"),
    "campaigns": ("🎯 Campagnes & ROI", "Ciblage de cohorte, calculateur de gain financier ($ MRR) et export"),
    "monitoring": ("⚖️ Équité & MLOps", "Audit d'équité démographique et suivi de la dérive de distribution (PSI)"),
}

# 5. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-title">⚡ PulseRetention AI</div>
        <div class="brand-subtitle">Retention & Customer NPS Platform</div>
    </div>
    """, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "overview"

    for page_key, (page_label, _) in NAVIGATION_PAGES.items():
        is_selected = (st.session_state.current_page == page_key)
        if st.button(page_label, key=f"nav_{page_key}", type="primary" if is_selected else "secondary"):
            st.session_state.current_page = page_key
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#64748B; font-size: 0.72rem; line-height: 1.4; padding: 0.2rem 0.5rem;">
        <b>Version 2.0 (Production)</b><br>
        Modèle : Régression Ordinale (mord)<br>
        Base scorée : 5 987 clients silencieux<br>
        Reproductibilité : Seed 42
    </div>
    """, unsafe_allow_html=True)

# 6. Page Routing
current_view = st.session_state.current_page

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
