"""Campaign Manager & Business ROI Calculator for NPS Intelligence."""
import io
import plotly.express as px
import streamlit as st
import pandas as pd

from app.styles import NPS_COLOR_MAP, get_plotly_layout_template
from src.config import ID_COL, TARGET_COL
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix


def render_campaigns_view(pipeline, labeled_df, splits_df):
    # Prepare silent cohort predictions
    merged = labeled_df.merge(splits_df, on=ID_COL)
    silent_cohort = merged[merged["split"].isin(["silent_val", "silent_test"])].copy().reset_index(drop=True)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_matrix = build_feature_matrix(silent_cohort, include_geo=False)[feature_cols]

    with st.spinner("Scoring de la cohorte silencieuse en cours..."):
        preds = pipeline.predict(X_matrix)
        probas = pipeline.predict_proba(X_matrix)

    silent_cohort["Predicted_NPS"] = preds
    silent_cohort["Detractor_Probability"] = probas[:, 0]
    silent_cohort["Promoter_Probability"] = probas[:, 2]

    st.markdown('<div class="campaign-filters-header"><div class="campaign-filter-label">Critères de ciblage de la campagne</div></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        target_nps = st.multiselect(
            "Statut NPS Prédit",
            options=["Detractor", "Passive", "Promoter"],
            default=["Detractor"],
        )
    with f2:
        contract_filter = st.multiselect(
            "Type de contrat",
            options=["Month-to-month", "One year", "Two year"],
            default=["Month-to-month"],
        )
    with f3:
        min_detractor_prob = st.slider(
            "Probabilité de détraction min.",
            min_value=0.30,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )
    with f4:
        min_charges = st.number_input(
            "Facture mensuelle min. ($)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=10.0,
        )

    # Apply filters
    filtered_df = silent_cohort[
        (silent_cohort["Predicted_NPS"].isin(target_nps)) &
        (silent_cohort["Contract"].isin(contract_filter)) &
        (silent_cohort["Detractor_Probability"] >= min_detractor_prob) &
        (silent_cohort["Monthly Charges"] >= min_charges)
    ].copy()

    st.write("")

    target_count = len(filtered_df)
    total_mrr_targeted = filtered_df["Monthly Charges"].sum() if target_count > 0 else 0.0
    avg_charge = (total_mrr_targeted / target_count) if target_count > 0 else 0.0

    st.markdown('<div class="campaign-kpi-grid">', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="campaign-kpi-card">
            <div class="campaign-kpi-label">Clients ciblés</div>
            <div class="campaign-kpi-value">{target_count:,}</div>
            <div class="campaign-kpi-subtext">Clients éligibles pour l'action de rétention</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="campaign-kpi-card">
            <div class="campaign-kpi-label">MRR exposé au churn</div>
            <div class="campaign-kpi-value">{total_mrr_targeted:,.0f} $</div>
            <div class="campaign-kpi-subtext">Facturation mensuelle directe à risque</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="campaign-kpi-card">
            <div class="campaign-kpi-label">Facture mensuelle moyenne</div>
            <div class="campaign-kpi-value">{avg_charge:.2f} $</div>
            <div class="campaign-kpi-subtext">Panier moyen par client ciblé</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.divider()

    # ROI Business Simulator
    st.markdown('<div class="section-eyebrow">Modélisation financière</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rentabilité prévisionnelle de la campagne d\'appels sortants</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        conversion_rate = st.slider(
            "Taux de succès de l'action de rétention",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.01,
            format="%.0f%%",
            help="Part des clients contactés acceptant l'offre de fidélisation.",
        )
    with r2:
        cost_per_contact = st.number_input(
            "Coût de contact par client ($)",
            min_value=1.0,
            max_value=25.0,
            value=5.0,
            step=1.0,
            help="Coût opérationnel d'un appel sortant qualifié.",
        )
    with r3:
        discount_per_month = st.number_input(
            "Remise mensuelle accordée ($)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=1.0,
            help="Remise consentie sur 12 mois pour sécuriser le client.",
        )

    # Financial calculations
    saved_customers = int(target_count * conversion_rate)
    campaign_contact_cost = target_count * cost_per_contact
    net_saved_mrr = saved_customers * max(0.0, avg_charge - discount_per_month)
    annual_net_value = (net_saved_mrr * 12) - campaign_contact_cost
    roi_multiple = (annual_net_value / campaign_contact_cost) if campaign_contact_cost > 0 else 0.0

    st.write("")
    c_fin1, c_fin2, c_fin3, c_fin4 = st.columns(4)
    with c_fin1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Clients Sauvegardés</div>
            <div class="kpi-value">{saved_customers:,}</div>
            <div class="kpi-subtext">Rétention active réussie</div>
        </div>
        """, unsafe_allow_html=True)

    with c_fin2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">MRR Net Sauvegardé</div>
            <div class="kpi-value">{net_saved_mrr:,.0f} $</div>
            <div class="kpi-subtext">Revenu mensuel récurrent préservé</div>
        </div>
        """, unsafe_allow_html=True)

    with c_fin3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Gain Net Annuel (ARR)</div>
            <div class="kpi-value">{annual_net_value:,.0f} $</div>
            <div class="kpi-subtext">Net après coûts opérationnels</div>
        </div>
        """, unsafe_allow_html=True)

    with c_fin4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Retour sur Investissement</div>
            <div class="kpi-value">{roi_multiple:.1f}x</div>
            <div class="kpi-subtext">Multiple ROI de l'opération</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Export Section
    st.markdown('<div class="section-eyebrow">Export Opérationnel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Liste de Travail pour l\'Équipe Commerciale & Télévente</div>', unsafe_allow_html=True)

    if target_count > 0:
        export_cols = [
            ID_COL, "Predicted_NPS", "Detractor_Probability", "Contract",
            "Monthly Charges", "Total Charges", "Internet Service", "Tenure Months", "Phone Service"
        ]
        csv_buffer = io.StringIO()
        filtered_df[export_cols].to_csv(csv_buffer, index=False)

        st.dataframe(
            filtered_df[export_cols].head(8),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            label=f"Exporter la liste de ciblage ({target_count:,} clients) au format CSV",
            data=csv_buffer.getvalue(),
            file_name="campagne_retention_cible.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.warning("Aucun client ne correspond aux critères de filtre sélectionnés.")
