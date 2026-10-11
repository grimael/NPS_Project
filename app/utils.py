"""Data loading, caching, inference and explanation services for NPS Intelligence."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.config import (
        FINAL_MODEL_PATH,
        ID_COL,
        LABELED_DATASET_PATH,
        NPS_CATEGORIES,
        REALISTIC_SPLIT_PATH,
        TARGET_COL,
    )
    from src.evaluate import evaluate
    from src.features import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        add_engineered_features,
        build_feature_matrix,
    )
except ImportError:
    from config import (
        FINAL_MODEL_PATH,
        ID_COL,
        LABELED_DATASET_PATH,
        NPS_CATEGORIES,
        REALISTIC_SPLIT_PATH,
        TARGET_COL,
    )
    from evaluate import evaluate
    from features import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        add_engineered_features,
        build_feature_matrix,
    )

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


def ensure_pipeline_artifacts():
    """Self-healing: automatically builds pipeline datasets and trained model if absent (Cloud deploy)."""
    if not (FINAL_MODEL_PATH.exists() and LABELED_DATASET_PATH.exists() and REALISTIC_SPLIT_PATH.exists()):
        try:
            from src.make_dataset import main as run_make_dataset
            from src.build_target import main as run_build_target
            from src.make_splits import main as run_make_splits
            from src.train import main as run_train

            if not LABELED_DATASET_PATH.exists():
                run_make_dataset()
                run_build_target()
            if not REALISTIC_SPLIT_PATH.exists():
                run_make_splits()
            if not FINAL_MODEL_PATH.exists():
                run_train()
        except Exception as e:
            st.error(f"Erreur d'initialisation du pipeline : {e}")


def get_file_signature(*paths: Path) -> float:
    """Latest modification timestamp for cache invalidation."""
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return max(times) if times else 0.0


@st.cache_resource
def load_trained_model(mtime: float):
    """Load model artifact, auto-building if missing."""
    ensure_pipeline_artifacts()
    if not FINAL_MODEL_PATH.exists():
        return None
    return joblib.load(FINAL_MODEL_PATH)


@st.cache_data
def load_datasets(mtime: float):
    """Load labeled dataset and splits, auto-building if missing."""
    ensure_pipeline_artifacts()
    if not (LABELED_DATASET_PATH.exists() and REALISTIC_SPLIT_PATH.exists()):
        return None, None
    labeled = pd.read_parquet(LABELED_DATASET_PATH)
    splits = pd.read_parquet(REALISTIC_SPLIT_PATH)
    return labeled, splits


@st.cache_data
def compute_executive_kpis(mtime: float):
    """Compute live portfolio KPIs on silent holdout test set."""
    artifact = load_trained_model(mtime)
    labeled, splits = load_datasets(mtime)
    if artifact is None or labeled is None or splits is None:
        return {}

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    X = build_feature_matrix(labeled, include_geo=False)
    merged = X.merge(labeled[[ID_COL, TARGET_COL]], on=ID_COL).merge(splits, on=ID_COL)

    test_df = merged[merged["split"] == "silent_test"]
    val_df = merged[merged["split"] == "silent_val"]
    all_silent = merged[merged["split"].isin(["silent_val", "silent_test"])]

    # Scores on frozen test set
    y_test = test_df[TARGET_COL].astype(str)
    y_pred = pipeline.predict(test_df[feature_columns])
    y_proba = pipeline.predict_proba(test_df[feature_columns])
    test_results = evaluate(y_test, y_pred, y_proba)

    # Estimate silent cohort predictions & revenue at risk
    all_silent_pred = pipeline.predict(all_silent[feature_columns])
    detractor_count = (all_silent_pred == "Detractor").sum()
    detractor_share = detractor_count / len(all_silent)

    # Monthly Revenue at Risk (sum of monthly charges for predicted detractors in silent base)
    silent_raw = labeled.merge(splits, on=ID_COL)
    silent_raw_cohort = silent_raw[silent_raw["split"].isin(["silent_val", "silent_test"])].copy()
    silent_raw_cohort["predicted_nps"] = all_silent_pred
    mrr_at_risk = silent_raw_cohort[silent_raw_cohort["predicted_nps"] == "Detractor"]["Monthly Charges"].sum()

    return {
        "total_customers": len(labeled),
        "silent_customers": len(all_silent),
        "silent_pct": len(all_silent) / len(labeled),
        "respondents": len(labeled) - len(all_silent),
        "test_qwk": test_results["qwk"],
        "test_balanced_acc": test_results["balanced_accuracy"],
        "test_detractor_recall": test_results["recall_per_class"]["Detractor"],
        "predicted_detractors_silent": int(detractor_count),
        "predicted_detractor_rate": detractor_share,
        "mrr_at_risk": float(mrr_at_risk),
    }


def predict_single_customer(pipeline, raw_df: pd.DataFrame):
    """Score a single raw customer profile."""
    engineered = add_engineered_features(raw_df)
    prep = pipeline.named_steps["prep"]
    feature_columns = prep.transformers[0][2] + prep.transformers[1][2]
    X = engineered[feature_columns]

    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.named_steps["model"].classes_
    proba_dict = dict(zip(classes, proba))
    return pred, proba_dict, X


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_pipeline, _background_raw: pd.DataFrame, mtime: float):
    """Build SHAP explainer on transformed features."""
    import shap
    prep = _pipeline.named_steps["prep"]
    model = _pipeline.named_steps["model"]
    feature_columns = prep.transformers[0][2] + prep.transformers[1][2]

    bg_eng = add_engineered_features(_background_raw)[feature_columns]
    bg_t = prep.transform(bg_eng)
    feature_names = prep.get_feature_names_out()
    explainer = shap.Explainer(model.predict_proba, bg_t)
    return explainer, feature_names


def explain_prediction(pipeline, X_row: pd.DataFrame, predicted_class: str, background_raw: pd.DataFrame):
    """Compute top SHAP drivers for a customer prediction."""
    explainer, feature_names = get_shap_explainer(
        pipeline, background_raw, get_file_signature(FINAL_MODEL_PATH)
    )
    prep = pipeline.named_steps["prep"]
    X_t = pd.DataFrame(prep.transform(X_row), columns=feature_names)
    shap_vals = explainer(X_t)

    class_idx = NPS_CATEGORIES.index(predicted_class)
    values = shap_vals.values[0, :, class_idx]
    order = np.argsort(-np.abs(values))[:6]

    clean_names = []
    for i in order:
        raw_name = feature_names[i].split("__", 1)[-1]
        raw_name = raw_name.replace("_", ": ", 1) if "_" in raw_name and not raw_name.startswith(("Tenure", "Number")) else raw_name
        clean_names.append(raw_name)

    return clean_names, values[order]


def get_default_customer_values(labeled_df: pd.DataFrame) -> dict:
    """Compute representative default values from data."""
    defaults = {}
    for col in RAW_NUMERIC_FIELDS:
        defaults[col] = float(labeled_df[col].median())
    for col in CATEGORICAL_OPTIONS:
        defaults[col] = labeled_df[col].mode(dropna=True).iloc[0]
    defaults["Multiple Lines"] = labeled_df["Multiple Lines"].mode(dropna=True).iloc[0]
    non_null_type = labeled_df["Internet Type"].dropna()
    defaults["Internet Type"] = non_null_type.mode().iloc[0] if len(non_null_type) else "DSL"
    return defaults


def build_raw_dataframe(values: dict) -> pd.DataFrame:
    """Build single row dataframe with consistent dependencies."""
    row = dict(values)
    row["Under 30"] = "Yes" if row["Age"] < 30 else "No"

    if row.get("Phone Service") == "No":
        row["Multiple Lines"] = "No phone service"

    if row.get("Internet Service") == "No":
        row["Internet Type"] = None
        for col in ["Online Security", "Online Backup", "Device Protection",
                    "Tech Support", "Streaming TV", "Streaming Movies"]:
            row[col] = "No internet service"

    if row.get("Offer") == "No Offer":
        row["Offer"] = None

    return pd.DataFrame([row])
