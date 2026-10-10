"""What-If Prescriptive Action Simulator for TelcoPulse AI.

Enables retention teams to simulate operational interventions (contract upgrades,
discounting, security bundles, automatic payment) and measure their immediate
probabilistic impact on customer detraction risk.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.styles import (
    COLOR_DETRACTOR,
    COLOR_PASSIVE,
    COLOR_PROMOTER,
    NPS_COLOR_MAP,
)
from app.utils import predict_single_customer
from src.config import ID_COL, NPS_CATEGORIES


def render_simulator_view(pipeline, labeled_df, splits_df):
    st.markdown("""
    Testez en temps réel l'impact de mesures commerciales et contractuelles sur le risque de détraction d'un client.
    Ce simulateur permet aux équipes Customer Experience d'arbitrer l'efficacité d'une offre avant de contacter le client.
    """)

    merged = labeled_df.merge(splits_df, on=ID_COL)
    silent_pool = merged[merged["split"].isin(["silent_val", "silent_test"])].reset_index(drop=True)

    # Let user pick a customer
    c_pick1, c_pick2 = st.columns([2, 1])
    with c_pick1:
        at_risk_pool = silent_pool[silent_pool["Contract"] == "Month-to-month"].reset_index(drop=True)
        selected_id = st.selectbox(
            "Sélectionner un client à risque (Contrat mensuel)",
            options=at_risk_pool[ID_COL].tolist()[:300],
            help="Sélection parmi les clients silencieux les plus exposés."
        )

    base_customer = at_risk_pool[at_risk_pool[ID_COL] == selected_id].iloc[[0]].copy()

    # Base prediction
    base_pred, base_proba, _ = predict_single_customer(pipeline, base_customer)

    with st.expander("Situation contractuelle actuelle du client", expanded=True):
        st.write(
            f"**Contrat :** `{base_customer['Contract'].iloc[0]}` | "
            f"**Facture mensuelle :** `{base_customer['Monthly Charges'].iloc[0]:.2f} $` | "
            f"**Internet :** `{base_customer['Internet Service'].iloc[0]} ({base_customer['Internet Type'].iloc[0]})` | "
            f"**Support technique :** `{base_customer['Tech Support'].iloc[0]}` | "
            f"**Sécurité en ligne :** `{base_customer['Online Security'].iloc[0]}`"
        )
        st.markdown(
            f"Statut actuel : **{base_pred}** — Risque de détraction initial : **{base_proba['Detractor']:.1%}**"
        )

    st.write("")
    st.markdown("### Paramétrage des Leviers de Fidélisation")

    col_l1, col_l2, col_l3, col_l4 = st.columns(4)

    with col_l1:
        new_contract = st.selectbox(
            "Engagement contractuel",
            options=["Month-to-month", "One year", "Two year"],
            index=1,
            help="Proposition de passage à un contrat avec engagement."
        )

    with col_l2:
        discount = st.slider(
            "Remise mensuelle ($)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=2.5,
            help="Remise tarifaire mensuelle accordée au client."
        )

    with col_l3:
        add_security = st.checkbox(
            "Pack Sécurité en ligne",
            value=True if base_customer["Online Security"].iloc[0] == "No" else False,
            help="Activation de la protection anti-menace."
        )

    with col_l4:
        add_support = st.checkbox(
            "Support Technique 24/7",
            value=True if base_customer["Tech Support"].iloc[0] == "No" else False,
            help="Assistance technique dédiée prioritaire."
        )

    # Build simulated row
    sim_row = base_customer.copy()
    sim_row["Contract"] = new_contract
    sim_row["Monthly Charges"] = max(10.0, float(base_customer["Monthly Charges"].iloc[0] - discount))

    if base_customer["Internet Service"].iloc[0] != "No":
        if add_security:
            sim_row["Online Security"] = "Yes"
        if add_support:
            sim_row["Tech Support"] = "Yes"

    # Score simulated profile
    sim_pred, sim_proba, _ = predict_single_customer(pipeline, sim_row)

    st.divider()

    # Visual Comparison Before vs After
    st.markdown('<div class="section-eyebrow">Impact Prévisionnel de la Mesure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comparaison Avant / Après Intervention</div>', unsafe_allow_html=True)

    delta_detractor = sim_proba["Detractor"] - base_proba["Detractor"]
    delta_promoter = sim_proba["Promoter"] - base_proba["Promoter"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Risque Détracteur",
            value=f"{sim_proba['Detractor']:.1%}",
            delta=f"{delta_detractor:+.1%}",
            delta_color="inverse",
        )
    with k2:
        st.metric(
            label="Probabilité Promoteur",
            value=f"{sim_proba['Promoter']:.1%}",
            delta=f"{delta_promoter:+.1%}",
        )
    with k3:
        st.metric(
            label="Nouveau Statut",
            value=sim_pred,
        )
    with k4:
        cost_impact = discount * (12 if new_contract == "One year" else (24 if new_contract == "Two year" else 1))
        st.metric(
            label="Coût Remise Engagement",
            value=f"{cost_impact:,.0f} $",
            help="Coût total de la remise consentie sur la durée du nouvel engagement."
        )

    # Plotly Side-by-Side Comparison
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=NPS_CATEGORIES,
        y=[base_proba[c] for c in NPS_CATEGORIES],
        name="Avant Intervention",
        marker_color="#94A3B8",
        text=[f"{base_proba[c]:.1%}" for c in NPS_CATEGORIES],
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        x=NPS_CATEGORIES,
        y=[sim_proba[c] for c in NPS_CATEGORIES],
        name="Après Intervention (Simulée)",
        marker_color=[COLOR_DETRACTOR, COLOR_PASSIVE, COLOR_PROMOTER],
        text=[f"{sim_proba[c]:.1%}" for c in NPS_CATEGORIES],
        textposition="outside",
    ))

    fig_comp.update_layout(
        barmode="group",
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="#E2E8F0"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    if delta_detractor < -0.15:
        st.success(
            f"**Intervention hautement efficace :** Le risque de détraction chute de {abs(delta_detractor):.1%} points. "
            f"La sécurisation contractuelle compense largement le coût consenti de la remise."
        )
    elif delta_detractor < 0:
        st.info(
            f"**Amélioration modérée :** Risque de détraction réduit de {abs(delta_detractor):.1%} points. "
            f"Envisager un engagement sur 2 ans ou un pack de services étendu."
        )
    else:
        st.warning("L'intervention configurée n'a pas permis de réduire significativement le risque de détraction.")
