"""Campaign Manager & Business ROI Calculator for TelcoPulse AI."""
import io
import plotly.express as px
import streamlit as st
import pandas as pd

from app.styles import NPS_COLOR_MAP
from src.config import ID_COL, TARGET_COL
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix


def render_campaigns_view(pipeline, labeled_df, splits_df):
    st.markdown("""
    Ciblez des cohortes de clients à risque au sein de la base silencieuse, simulez le retour sur investissement d'une campagne
    d'appels sortants et exportez la liste de travail pour l'équipe commerciale.
    """)

    # Prepare silent cohort predictions
    merged = labeled_df.merge(splits_df, on=ID_COL)
    silent_cohort = merged[merged["split"].isin(["silent_val", "silent_test"])].copy().reset_index(drop=True)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_matrix = build_feature_matrix(silent_cohort, include_geo=False)[feature_cols]

    with st.spinner("Scoring de la base silencieuse en cours..."):
        preds = pipeline.predict(X_matrix)
        probas = pipeline.predict_proba(X_matrix)

    silent_cohort["Predicted_NPS"] = preds
    silent_cohort["Detractor_Probability"] = probas[:, 0]
    silent_cohort["Promoter_Probability"] = probas[:, 2]

    # Filters Section
    st.markdown("### Critères de Ciblage de la Cohorte")
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

    # Campaign KPIs (Concise, zero overflow)
    target_count = len(filtered_df)
    total_mrr_targeted = filtered_df["Monthly Charges"].sum() if target_count > 0 else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Clients Ciblés", f"{target_count:,}")
    with k2:
        st.metric("MRR Exposé au Churn", f"{total_mrr_targeted:,.0f} $")
    with k3:
        avg_charge = (total_mrr_targeted / target_count) if target_count > 0 else 0.0
        st.metric("Facture Moyenne", f"{avg_charge:.2f} $")

    st.divider()

    # ROI Business Simulator
    st.markdown('<div class="section-eyebrow">Simulation Financière</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Modélisation de la Rentabilité de la Campagne</div>', unsafe_allow_html=True)

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

    c_fin1, c_fin2, c_fin3, c_fin4 = st.columns(4)
    with c_fin1:
        st.metric("Clients Sauvegardés", f"{saved_customers:,}")
    with c_fin2:
        st.metric("MRR Net Sauvegardé", f"{net_saved_mrr:,.0f} $")
    with c_fin3:
        st.metric("Gain Annuel Net", f"{annual_net_value:,.0f} $")
    with c_fin4:
        st.metric("ROI Multiple", f"{roi_multiple:.1f}x")

    st.write("")

    # Export Section
    st.markdown("### Export de la Liste de Travail Commerciale")
    if target_count > 0:
        export_cols = [
            ID_COL, "Predicted_NPS", "Detractor_Probability", "Contract",
            "Monthly Charges", "Total Charges", "Internet Service", "Tenure Months", "Phone Service"
        ]
        csv_buffer = io.StringIO()
        filtered_df[export_cols].to_csv(csv_buffer, index=False)

        st.dataframe(
            filtered_df[export_cols].head(10),
            hide_index=True,
            use_container_width=True,
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
