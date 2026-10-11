"""Fairness Audit & MLOps Drift Monitoring View for TelcoPulse AI."""
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd

from app.styles import ACCENT_INDIGO, ACCENT_SUN, ACCENT_TEAL, ACCENT_VIOLET
from app.styles import COLOR_DETRACTOR, COLOR_PASSIVE, COLOR_PROMOTER, get_plotly_layout_template
from src.config import ID_COL, TARGET_COL
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two numeric distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
    act_pct = np.maximum(act_counts / len(actual), 1e-4)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def render_monitoring_view(pipeline, labeled_df, splits_df):
    tab_fairness, tab_mlops = st.tabs(["Audit d'Équité Démographique", "Monitoring MLOps & Dérive (PSI)"])

    merged = labeled_df.merge(splits_df, on=ID_COL)
    test_cohort = merged[merged["split"] == "silent_test"].copy().reset_index(drop=True)

    # Predictions on test cohort
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_test = build_feature_matrix(test_cohort, include_geo=False)[feature_cols]
    test_cohort["pred"] = pipeline.predict(X_test)
    test_cohort["actual_detractor"] = (test_cohort[TARGET_COL] == "Detractor")
    test_cohort["pred_detractor"] = (test_cohort["pred"] == "Detractor")

    # 1. FAIRNESS TAB
    with tab_fairness:
        st.markdown("### Évaluation du Rappel Détracteur par Groupe Protégé")
        st.markdown("""
        Un modèle équitable doit détecter les clients insatisfaits avec un taux d'efficacité comparable,
        quel que soit leur profil démographique (genre, âge, situation familiale).
        """)

        groups = [
            ("Genre", "Gender", ["Female", "Male"]),
            ("Senior Citizen", "Senior Citizen", ["No", "Yes"]),
            ("Personnes à charge", "Dependents", ["No", "Yes"]),
            ("Partenaire", "Partner", ["No", "Yes"]),
        ]

        fairness_data = []
        for group_label, col, values in groups:
            for val in values:
                sub = test_cohort[test_cohort[col] == val]
                detractors = sub[sub["actual_detractor"]]
                recall = (detractors["pred_detractor"].sum() / len(detractors)) if len(detractors) > 0 else 0.0
                fairness_data.append({
                    "Groupe": group_label,
                    "Sous-groupe": f"{group_label} = {val}",
                    "Rappel Détracteur": recall,
                    "Effectif (Détracteurs)": len(detractors),
                })

        fairness_df = pd.DataFrame(fairness_data)

        subgroup_colors = [
            ACCENT_INDIGO, ACCENT_TEAL,
            ACCENT_VIOLET, ACCENT_SUN,
            COLOR_PROMOTER, COLOR_DETRACTOR,
            "#536A9C", COLOR_PASSIVE,
        ]
        fairness_by_subgroup = fairness_df.set_index("Sous-groupe")
        fairness_layout = get_plotly_layout_template()
        group_columns = st.columns(2)
        subgroup_display_labels = {
            "Female": "Femme",
            "Male": "Homme",
            "No": "Non",
            "Yes": "Oui",
        }

        for group_index, (group_label, _, values) in enumerate(groups):
            with group_columns[group_index % 2]:
                st.markdown(
                    f'<div class="fairness-donut-title">{group_label}</div>',
                    unsafe_allow_html=True,
                )
                fig_fair = go.Figure()
                annotations = []

                for subgroup_index, value in enumerate(values):
                    subgroup = f"{group_label} = {value}"
                    metric = fairness_by_subgroup.loc[subgroup]
                    recall = float(metric["Rappel Détracteur"])
                    detractor_count = int(metric["Effectif (Détracteurs)"])
                    display_value = subgroup_display_labels[value]
                    x0, x1 = (0.02, 0.48) if subgroup_index == 0 else (0.52, 0.98)
                    center_x = (x0 + x1) / 2
                    accent = subgroup_colors[group_index * 2 + subgroup_index]

                    fig_fair.add_trace(go.Pie(
                        labels=["Détectés", "Non détectés"],
                        values=[recall, 1 - recall] if detractor_count else [0, 1],
                        name=subgroup,
                        hole=0.72,
                        sort=False,
                        direction="clockwise",
                        rotation=270,
                        domain={"x": [x0, x1], "y": [0.08, 0.9]},
                        marker={
                            "colors": [accent, "#E8EDF4"],
                            "line": {"color": "#FFFFFF", "width": 3},
                        },
                        textinfo="none",
                        hovertemplate=(
                            f"{subgroup}<br>%{{label}}: %{{percent}}"
                            f"<br>Détracteurs: {detractor_count}<extra></extra>"
                        ),
                        showlegend=False,
                    ))
                    annotations.extend([
                        {
                            "text": display_value,
                            "x": center_x,
                            "y": 1.02,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 11, "color": "#475569"},
                        },
                        {
                            "text": f"{recall:.0%}" if detractor_count else "N/D",
                            "x": center_x,
                            "y": 0.49,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 19, "color": "#0F172A", "family": "Plus Jakarta Sans"},
                        },
                    ])

                layout = dict(fairness_layout)
                layout.update(
                    height=250,
                    margin={"t": 28, "b": 4, "l": 4, "r": 4},
                    showlegend=False,
                    annotations=annotations,
                )
                fig_fair.update_layout(**layout)
                st.plotly_chart(fig_fair, use_container_width=True)
                counts = [
                    int(fairness_by_subgroup.loc[f"{group_label} = {value}", "Effectif (Détracteurs)"])
                    for value in values
                ]
                st.caption(
                    f"Détracteurs réels dans chaque sous-groupe : "
                    f"{values[0]} = {counts[0]:,} · {values[1]} = {counts[1]:,}"
                )

        st.markdown("""
        <div class="callout-box">
            <b>Décisions d'Atténuation des Biais :</b><br>
            • <b>Suppression des proxys géographiques :</b> <code>Zip Code</code> et <code>City</code> ont été exclus dès la conception en tant que proxys socio-économiques.<br>
            • <b>Ablation Familiale :</b> Le retrait des features <code>Dependents</code> et <code>Household Size Proxy</code> a réduit l'écart de rappel de 19.2 points à 11.8 points sans altérer la performance globale (QWK 0.287).
        </div>
        """, unsafe_allow_html=True)

    # 2. MLOPS MONITORING TAB
    with tab_mlops:
        st.markdown("### Suivi de la Dérive de Population (Population Stability Index)")
        st.markdown("""
        Le modèle a été entraîné sur la cohorte des répondants (15 %) et est appliqué sur la cohorte silencieuse (85 %).
        L'indice PSI mesure la stabilité des variables clés pour détecter tout décrochage nécessitant un ré-entraînement.
        """)

        resp_cohort = merged[merged["split"] == "respondent_train"]
        silent_cohort = merged[merged["split"].isin(["silent_val", "silent_test"])]

        tracked_features = ["Monthly Charges", "Total Charges", "Tenure Months", "Age"]
        psi_records = []
        for feat in tracked_features:
            psi_score = compute_psi(
                resp_cohort[feat].to_numpy(),
                silent_cohort[feat].to_numpy(),
            )
            status = "Stable (PSI < 0.10)" if psi_score < 0.10 else ("Dérive Modérée" if psi_score < 0.25 else "Dérive Critique (> 0.25)")
            psi_records.append({
                "Variable": feat,
                "Score PSI": round(psi_score, 4),
                "Statut Dérive": status,
                "Moyenne Répondants": round(resp_cohort[feat].mean(), 1),
                "Moyenne Silencieux": round(silent_cohort[feat].mean(), 1),
            })

        psi_df = pd.DataFrame(psi_records)
        st.dataframe(psi_df, hide_index=True, width="stretch")

        st.write("")
        st.markdown("### Règles de Déclenchement de Ré-Entraînement")
        c_trig1, c_trig2, c_trig3 = st.columns(3)
        with c_trig1:
            st.markdown("""
            <div class="kpi-card" style="min-height: 140px;">
                <div class="kpi-label">Nouvelles Enquêtes</div>
                <div class="kpi-value">+1 000 Réponses</div>
                <div class="kpi-subtext">Déclencheur automatique dès que 1 000 nouveaux formulaires NPS sont collectés par le service client.</div>
            </div>
            """, unsafe_allow_html=True)
        with c_trig2:
            st.markdown("""
            <div class="kpi-card" style="min-height: 140px;">
                <div class="kpi-label">Seuil de Dérive PSI</div>
                <div class="kpi-value">PSI > 0.25</div>
                <div class="kpi-subtext">Alerte critique envoyée au Head of Data si une variable contractuelle dépasse 0.25 deux semaines d'affilée.</div>
            </div>
            """, unsafe_allow_html=True)
        with c_trig3:
            st.markdown("""
            <div class="kpi-card" style="min-height: 140px;">
                <div class="kpi-label">Calendrier Fixe</div>
                <div class="kpi-value">Semestriel (6 mois)</div>
                <div class="kpi-subtext">Audit semestriel complet avec ré-étalonnage des seuils ordinaux et revue avec l'équipe Rétention.</div>
            </div>
            """, unsafe_allow_html=True)
