"""Individual Customer Prediction & SHAP Diagnostics View for TelcoPulse AI."""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.styles import (
    COLOR_DETRACTOR,
    COLOR_PASSIVE,
    COLOR_PROMOTER,
    NPS_COLOR_MAP,
    get_plotly_layout_template,
)
from app.utils import (
    CATEGORICAL_OPTIONS,
    INTERNET_TYPE_OPTIONS,
    RAW_NUMERIC_FIELDS,
    build_raw_dataframe,
    explain_prediction,
    get_default_customer_values,
    predict_single_customer,
)
from src.config import ID_COL, NPS_CATEGORIES, TARGET_COL


def render_prediction_view(pipeline, labeled_df, splits_df):
    tab_existing, tab_manual = st.tabs(["Client existant (Base silencieuse)", "Saisie manuelle d'un profil"])

    selected_raw_row = None
    defaults = get_default_customer_values(labeled_df)
    background_sample = labeled_df.sample(n=min(60, len(labeled_df)), random_state=42)

    with tab_existing:
        merged = labeled_df.merge(splits_df, on=ID_COL)
        silent_pool = merged[merged["split"].isin(["silent_val", "silent_test"])].reset_index(drop=True)

        col_search, col_filter = st.columns([2, 1])
        with col_filter:
            split_choice = st.selectbox("Partition de la base silencieuse", ["Tous les silencieux", "Validation (silent_val)", "Test (silent_test)"])
            if split_choice == "Validation (silent_val)":
                filtered_pool = silent_pool[silent_pool["split"] == "silent_val"]
            elif split_choice == "Test (silent_test)":
                filtered_pool = silent_pool[silent_pool["split"] == "silent_test"]
            else:
                filtered_pool = silent_pool

        with col_search:
            customer_id = st.selectbox(
                f"Identifiant Client ({len(filtered_pool):,} clients silencieux éligibles)",
                options=filtered_pool[ID_COL].tolist()[:500],
                help="Sélectionnez un client n'ayant jamais répondu à l'enquête NPS."
            )

        customer_row = filtered_pool[filtered_pool[ID_COL] == customer_id].iloc[[0]]

        with st.expander("Fiche signalétique du client", expanded=True):
            preview_cols = [
                "Tenure Months", "Contract", "Monthly Charges", "Total Charges",
                "Internet Service", "Payment Method", "Tech Support", "Online Security"
            ]
            st.dataframe(customer_row[preview_cols], hide_index=True, width="stretch")

        if st.button("Lancer le diagnostic client", key="btn_score_existing", type="primary"):
            selected_raw_row = customer_row

    with tab_manual:
        with st.form("manual_customer_form"):
            st.markdown("**Profil Démographique & Foyer**")
            c1, c2, c3, c4 = st.columns(4)
            gender = c1.selectbox("Genre", CATEGORICAL_OPTIONS["Gender"], index=0)
            age = c2.number_input("Âge", min_value=18, max_value=95, value=int(defaults["Age"]))
            senior = c3.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"], index=0)
            partner = c4.selectbox("Partenaire", CATEGORICAL_OPTIONS["Partner"], index=0)

            c5, c6, c7 = st.columns(3)
            dependents = c5.selectbox("Personnes à charge", CATEGORICAL_OPTIONS["Dependents"], index=0)
            n_dependents = c6.number_input("Nombre de personnes à charge", min_value=0, max_value=10, value=0)
            referred = c7.selectbox("A parrainé un client", CATEGORICAL_OPTIONS["Referred a Friend"], index=0)

            st.markdown("**Contrat, Facturation & Ancienneté**")
            c1, c2, c3 = st.columns(3)
            tenure = c1.number_input("Ancienneté (mois)", min_value=0, max_value=80, value=int(defaults["Tenure Months"]))
            contract = c2.selectbox("Type de contrat", CATEGORICAL_OPTIONS["Contract"], index=0)
            payment = c3.selectbox("Moyen de paiement", CATEGORICAL_OPTIONS["Payment Method"], index=0)

            c4, c5, c6 = st.columns(3)
            paperless = c4.selectbox("Facture électronique", CATEGORICAL_OPTIONS["Paperless Billing"], index=1)
            monthly_charges = c5.number_input("Charges mensuelles ($)", min_value=10.0, max_value=150.0, value=float(defaults["Monthly Charges"]))
            total_charges = c6.number_input("Charges cumulées ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges))

            st.markdown("**Services Télécom & Add-ons**")
            c1, c2 = st.columns(2)
            phone_service = c1.selectbox("Téléphonie fixe", CATEGORICAL_OPTIONS["Phone Service"], index=1)
            multiple_lines = c2.selectbox("Lignes multiples", ["No", "Yes"], index=0) if phone_service == "Yes" else "No phone service"

            c3, c4 = st.columns(2)
            internet_service = c3.selectbox("Accès Internet", CATEGORICAL_OPTIONS["Internet Service"], index=1)
            internet_type = c4.selectbox("Technologie Internet", INTERNET_TYPE_OPTIONS, index=1) if internet_service != "No" else None

            c5, c6, c7, c8 = st.columns(4)
            online_sec = c5.selectbox("Sécurité en ligne", ["No", "Yes"], index=0) if internet_service != "No" else "No internet service"
            tech_sup = c6.selectbox("Support technique", ["No", "Yes"], index=0) if internet_service != "No" else "No internet service"
            streaming_tv = c7.selectbox("Streaming TV", ["No", "Yes"], index=0) if internet_service != "No" else "No internet service"
            unlimited_data = c8.selectbox("Data illimitée", ["No", "Yes"], index=1)

            submitted = st.form_submit_button("Calculer la prédiction NPS", type="primary")

        if submitted:
            manual_data = {
                "Gender": gender, "Age": age, "Senior Citizen": senior, "Partner": partner,
                "Dependents": dependents, "Number of Dependents": n_dependents,
                "Referred a Friend": referred, "Number of Referrals": 0,
                "Tenure Months": tenure, "Contract": contract, "Payment Method": payment,
                "Paperless Billing": paperless, "Monthly Charges": monthly_charges,
                "Total Charges": total_charges, "Offer": "No Offer",
                "Phone Service": phone_service, "Multiple Lines": multiple_lines,
                "Internet Service": internet_service, "Internet Type": internet_type,
                "Online Security": online_sec, "Online Backup": "No",
                "Device Protection": "No", "Tech Support": tech_sup,
                "Streaming TV": streaming_tv, "Streaming Movies": "No",
                "Streaming Music": "No", "Unlimited Data": unlimited_data,
            }
            selected_raw_row = build_raw_dataframe(manual_data)

    # 3. Output Diagnosis & Explanations
    if selected_raw_row is not None:
        st.divider()
        pred, proba, X_row = predict_single_customer(pipeline, selected_raw_row)

        c_res1, c_res2 = st.columns([1, 1.2])

        with c_res1:
            st.markdown('<div class="section-eyebrow">Résultat du Scoring</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Catégorie Prédite</div>', unsafe_allow_html=True)

            if pred == "Detractor":
                st.markdown('<div class="badge-detractor">DÉFRACTEUR • RISQUE ÉLEVÉ</div>', unsafe_allow_html=True)
            elif pred == "Passive":
                st.markdown('<div class="badge-passive">PASSIF • VULNÉRABLE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-promoter">PROMOTEUR • AMBASSADEUR</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown("**Probabilités estimées :**")

            # Plotly Horizontal Probability Bars
            fig_p = go.Figure()
            for cat in NPS_CATEGORIES:
                val = proba[cat]
                fig_p.add_trace(go.Bar(
                    y=[cat],
                    x=[val],
                    orientation="h",
                    name=cat,
                    marker=dict(color=NPS_COLOR_MAP[cat], line=dict(color="#FFFFFF", width=1)),
                    text=[f"<b>{val:.1%}</b>"],
                    textposition="inside",
                    hovertemplate=f"{cat}: %{{x:.1%}}<extra></extra>",
                ))
            fig_p.update_layout(**get_plotly_layout_template())
            fig_p.update_layout(
                barmode="group",
                showlegend=False,
                height=180,
                margin=dict(t=10, b=10, l=10, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#E2E8F0"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with c_res2:
            st.markdown('<div class="section-eyebrow">Explicabilité IA (SHAP)</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Facteurs Clés Influant la Prédiction</div>', unsafe_allow_html=True)

            with st.spinner("Extraction des valeurs SHAP locales..."):
                top_names, top_values = explain_prediction(pipeline, X_row, pred, background_sample)

            colors = [NPS_COLOR_MAP[pred] if v > 0 else "#AEB8C8" for v in top_values]
            fig_shap = go.Figure(go.Bar(
                y=top_names[::-1],
                x=top_values[::-1],
                orientation="h",
                marker=dict(color=colors[::-1]),
                text=[f"{v:+.3f}" for v in top_values[::-1]],
                textposition="auto",
                marker_line_color="#FFFFFF",
                marker_line_width=1,
                hovertemplate="%{y}<br>Impact SHAP: %{x:+.3f}<extra></extra>",
            ))
            fig_shap.update_layout(**get_plotly_layout_template())
            fig_shap.update_layout(
                height=240,
                margin=dict(t=10, b=10, l=10, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title=f"Impact SHAP sur P({pred})", gridcolor="#E2E8F0"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Barres colorées = poussent vers la classe prédite. Barres grises = réduisent le risque.")

        # Actionable Agent Recommendations Box
        st.markdown("""
        <div class="callout-box">
            <b>Plan d'Action Conseiller Clientèle :</b><br>
            • <b>Si Détracteur :</b> Contact prioritaire sous 48h. Proposer la bascule vers un contrat annuel avec remise immédiate ou ajout de l'add-on Sécurité / Support technique.<br>
            • <b>Si Passif :</b> Proposer un audit de facture ou une offre de fidélité pour éviter la défection vers un concurrent.<br>
            • <b>Si Promoteur :</b> Inviter au programme de parrainage et solliciter un avis public.
        </div>
        """, unsafe_allow_html=True)
