"""Model training, validation, and persistence pipeline.

Trained on `respondent_train` (the 15% survey respondent cohort).
Validated on `silent_val` (to tune parameters and validate under distribution shift
without snooping the final test set).
Held out `silent_test` strictly for final benchmark reporting.

Uses `OrdinalLogisticClassifier` from `src.models.ordinal`, ensuring full scikit-learn
compatibility, standard pipeline encapsulation, and safe pickle/joblib serialization
without __main__ namespace pollution.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.config import (
        ID_COL,
        LABELED_DATASET_PATH,
        MODELS_DIR,
        NPS_CATEGORIES,
        REALISTIC_SPLIT_PATH,
        TARGET_COL,
    )
    from src.evaluate import evaluate
    from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix
    from src.models.ordinal import OrdinalLogisticClassifier
except ImportError:
    from config import (
        ID_COL,
        LABELED_DATASET_PATH,
        MODELS_DIR,
        NPS_CATEGORIES,
        REALISTIC_SPLIT_PATH,
        TARGET_COL,
    )
    from evaluate import evaluate
    from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix
    from models.ordinal import OrdinalLogisticClassifier


def build_final_pipeline(alpha: float = 1.0, balanced: bool = True) -> Pipeline:
    """Build preprocessor and OrdinalLogisticClassifier pipeline."""
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("model", OrdinalLogisticClassifier(alpha=alpha, balanced=balanced)),
    ])


def main() -> None:
    print("=" * 70)
    print("NPS Intelligence - Training & Validation Pipeline")
    print("=" * 70)

    labeled = pd.read_parquet(LABELED_DATASET_PATH)
    realistic_split = pd.read_parquet(REALISTIC_SPLIT_PATH)

    X = build_feature_matrix(labeled, include_geo=False)
    y = labeled[[ID_COL, TARGET_COL]]
    data = X.merge(y, on=ID_COL).merge(realistic_split, on=ID_COL)

    train_df = data[data["split"] == "respondent_train"]
    val_df = data[data["split"] == "silent_val"]
    test_df = data[data["split"] == "silent_test"]

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL].astype(str)

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_COL].astype(str)

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COL].astype(str)

    # 1. Fit model on respondents
    print(f"\n[1/4] Fitting model on respondent_train (n={len(train_df)})...")
    pipeline = build_final_pipeline(alpha=1.0, balanced=True)
    pipeline.fit(X_train, y_train)

    # 2. Evaluate on silent_val (Validation under Distribution Shift)
    print(f"\n[2/4] Validating on silent_val (n={len(val_df)}) [Shift Validation]...")
    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)
    val_metrics = evaluate(y_val, val_pred, val_proba)
    print(f"  -> silent_val QWK:               {val_metrics['qwk']:.3f}")
    print(f"  -> silent_val Balanced Acc:      {val_metrics['balanced_accuracy']:.1%}")
    print(f"  -> silent_val Detractor Recall:  {val_metrics['recall_per_class']['Detractor']:.1%}")
    print(f"  -> silent_val Macro F1:          {val_metrics['macro_f1']:.3f}")

    # 3. Evaluate on silent_test (Strictly Frozen Holdout Test)
    print(f"\n[3/4] Evaluating on silent_test (n={len(test_df)}) [Frozen Test Holdout]...")
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)
    test_metrics = evaluate(y_test, test_pred, test_proba)
    print(f"  -> silent_test QWK:              {test_metrics['qwk']:.3f}")
    print(f"  -> silent_test Balanced Acc:     {test_metrics['balanced_accuracy']:.1%}")
    print(f"  -> silent_test Detractor Recall: {test_metrics['recall_per_class']['Detractor']:.1%}")
    print(f"  -> silent_test Macro F1:         {test_metrics['macro_f1']:.3f}")

    # 4. Serialize model artifact cleanly
    artifact = {
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": feature_columns,
        "nps_categories": NPS_CATEGORIES,
        "trained_on": "respondent_train (realistic 3-way split)",
        "n_train_rows": len(train_df),
        "val_metrics": {
            "qwk": val_metrics["qwk"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "recall_detractor": val_metrics["recall_per_class"]["Detractor"],
            "macro_f1": val_metrics["macro_f1"],
        },
        "test_metrics": {
            "qwk": test_metrics["qwk"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "recall_detractor": test_metrics["recall_per_class"]["Detractor"],
            "macro_f1": test_metrics["macro_f1"],
        },
        "model_description": (
            "OrdinalLogisticClassifier (mord.LogisticAT, alpha=1.0, class-balanced sample weights), "
            "uncalibrated, serialized via src.models.ordinal."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"\n[4/4] Saved artifact -> {artifact_path}")

    # Serialization sanity check
    reloaded = joblib.load(artifact_path)
    sample = X_train.head(20)
    assert (pipeline.predict(sample) == reloaded["pipeline"].predict(sample)).all(), (
        "Reloaded pipeline predictions do not match the freshly-fit pipeline."
    )
    print("  -> OK: Reload & serialization sanity check passed.")


if __name__ == "__main__":
    main()
