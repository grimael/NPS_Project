"""Overview Page for TelcoPulse AI: Executive KPIs, NPS mix, and revenue at risk."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.styles import (
    COLOR_DETRACTOR,
    COLOR_PASSIVE,
    COLOR_PROMOTER,
    NPS_COLOR_MAP,
)
from app.utils import compute_executive_kpis, get_file_signature
from src.config import FINAL_MODEL_PATH, NPS_CATEGORIES, TARGET_COL


def render_overview_view(labeled_df, splits_df):
    mtime = get_file_signature(FINAL_MODEL_PATH)
    kpis = compute_executive_kpis(mtime)

    # 1. Executive KPI Cards (Anti-Overflow, Clean Typography)
    if kpis:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Parc Clients Global</div>
                <div class="kpi-value">{kpis['total_customers']:,}</div>
                <div class="kpi-subtext">{kpis['silent_customers']:,} silencieux ({kpis['silent_pct']:.0%})</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Détracteurs Silencieux</div>
                <div class="kpi-value">{kpis['predicted_detractors_silent']:,}</div>
                <div class="kpi-subtext">Part estimée : {kpis['predicted_detractor_rate']:.1%} de la cohorte</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">MRR Exposé au Churn</div>
                <div class="kpi-value">{kpis['mrr_at_risk']:,.0f} $</div>
                <div class="kpi-subtext">Facturation mensuelle à risque</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Précision Test Scellé</div>
                <div class="kpi-value">{kpis['test_qwk']:.3f} QWK</div>
                <div class="kpi-subtext">Rappel détracteur : {kpis['test_detractor_recall']:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # 2. Charts Row: NPS Mix (Donut) & Tenure Breakdown (Interactive Plotly)
    col_chart1, col_chart2 = st.columns([1, 1.2])

    with col_chart1:
        st.markdown('<div class="section-eyebrow">Diagnostic Global</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Répartition du Sentiment NPS</div>', unsafe_allow_html=True)

        counts = labeled_df[TARGET_COL].value_counts().reindex(NPS_CATEGORIES)
        fig_donut = go.Figure(data=[go.Pie(
            labels=NPS_CATEGORIES,
            values=counts.values,
            hole=0.62,
            marker_colors=[COLOR_DETRACTOR, COLOR_PASSIVE, COLOR_PROMOTER],
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            textposition="outside",
        )])
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"<b>{len(labeled_df):,}</b><br><span style='font-size:12px;color:#64748B;'>Clients</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False, align="center"
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption("Distribution dérivée du Satisfaction Score sur le parc client (IBM Telco v11.1.3+).")

    with col_chart2:
        st.markdown('<div class="section-eyebrow">Exposition par Segment</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Détracteurs par Type de Contrat</div>', unsafe_allow_html=True)

        contract_df = labeled_df.groupby(["Contract", TARGET_COL], observed=True).size().reset_index(name="count")
        fig_bar = px.bar(
            contract_df,
            x="Contract",
            y="count",
            color=TARGET_COL,
            color_discrete_map=NPS_COLOR_MAP,
            category_orders={TARGET_COL: NPS_CATEGORIES},
            barmode="stack",
            labels={"count": "Nombre de clients", "Contract": "Type de contrat"},
        )
        fig_bar.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Forte concentration des détracteurs sur les contrats sans engagement (Month-to-month).")

    st.divider()

    # 3. Strategic Highlights Cards (Clean Responsive Box)
    st.markdown('<div class="section-eyebrow">Synthèse Stratégique</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Points d\'Action Prioritaires pour l\'Équipe Rétention</div>', unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="kpi-card" style="min-height: 140px;">
            <div class="kpi-label">Enjeu des Silencieux</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">85 % du parc sans enquête</div>
            <div class="kpi-subtext">Sans IA, l'équipe rétention ignore l'insatisfaction de 5 987 clients. TelcoPulse détecte leurs signaux avec 51% de rappel.</div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown("""
        <div class="kpi-card" style="min-height: 140px;">
            <div class="kpi-label">Levier n°1 de Détraction</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">Facture Mensuelle Élevée</div>
            <div class="kpi-subtext">Les charges mensuelles et l'absence d'options de sécurité sont les premiers facteurs poussant les clients vers la détraction (SHAP).</div>
        </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown("""
        <div class="kpi-card" style="min-height: 140px;">
            <div class="kpi-label">Équité & Conformité</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">Audit Anti-Biais Actif</div>
            <div class="kpi-subtext">Le modèle a éliminé les proxys géographiques et familiaux directs pour garantir un traitement équitable de l'ensemble des segments.</div>
        </div>
        """, unsafe_allow_html=True)
