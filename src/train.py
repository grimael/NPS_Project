"""Model training and persistence.

`OrdinalLogisticClassifier` is a thin sklearn-compatible wrapper around
`mord.LogisticAT`, which internally expects integer-coded ordinal
labels rather than the `NPS Category` strings used everywhere else in
this project. Wrapping it gives it a standard scikit-learn classifier
contract (`classes_`, string-label `fit`/`predict`/`predict_proba`), so
it can sit inside a `Pipeline` with preprocessing, be used with
`CalibratedClassifierCV`, and be persisted as a single joblib artifact.

`main()` fits that pipeline on the realistic split's `respondent_train`
(the deployment-relevant training population, D8) and persists it —
per docs/decisions.md D21, this is `mord.LogisticAT`, uncalibrated
(D19: calibration made held-out performance catastrophically worse).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import mord
from config import (
    ID_COL,
    LABELED_DATASET_PATH,
    MODELS_DIR,
    NPS_CATEGORIES,
    NPS_CATEGORY_TO_CODE,
    REALISTIC_SPLIT_PATH,
    TARGET_COL,
)
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix

CODE_TO_CATEGORY = {v: k for k, v in NPS_CATEGORY_TO_CODE.items()}


class OrdinalLogisticClassifier(BaseEstimator, ClassifierMixin):
    """mord.LogisticAT operating directly on NPS_CATEGORIES string labels.

    Uses class-balanced sample weights by default (rule #5: class_weight
    before SMOTE) since `mord` has no native `class_weight` parameter.
    """

    def __init__(self, alpha: float = 1.0, balanced: bool = True):
        self.alpha = alpha
        self.balanced = balanced

    def fit(self, X, y, sample_weight=None):
        # Meta-estimators such as CalibratedClassifierCV re-encode y to
        # integer class positions (via their own LabelEncoder) before
        # calling the base estimator's fit — so y may arrive here either
        # as the original NPS_CATEGORIES strings (direct use) or as
        # already-ordinal-coded integers (wrapped use). Alphabetical
        # order happens to equal ordinal order for these three category
        # names (confirmed in docs/decisions.md D12), so both paths
        # agree on the same 0/1/2 mapping.
        y = np.asarray(y)
        if set(np.unique(y).astype(str)) <= set(NPS_CATEGORIES):
            y_code = np.array([NPS_CATEGORY_TO_CODE[str(label)] for label in y])
        else:
            y_code = y.astype(int)
        if sample_weight is None and self.balanced:
            sample_weight = compute_sample_weight("balanced", y_code)
        self.model_ = mord.LogisticAT(alpha=self.alpha)
        self.model_.fit(X, y_code, sample_weight=sample_weight)
        self.classes_ = np.array(NPS_CATEGORIES)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        codes = self.model_.predict(X)
        return np.array([CODE_TO_CATEGORY[c] for c in codes])


def build_final_pipeline() -> Pipeline:
    """Preprocessing + OrdinalLogisticClassifier, per docs/decisions.md D21."""
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("model", OrdinalLogisticClassifier(alpha=1.0, balanced=True)),
    ])


def main() -> None:
    labeled = pd.read_parquet(LABELED_DATASET_PATH)
    realistic_split = pd.read_parquet(REALISTIC_SPLIT_PATH)

    X = build_feature_matrix(labeled, include_geo=False)
    y = labeled[[ID_COL, TARGET_COL]]
    data = X.merge(y, on=ID_COL).merge(realistic_split, on=ID_COL)
    train_df = data[data["split"] == "respondent_train"]

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL].astype(str)

    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)
    print(f"Fit on {len(train_df)} rows (respondent_train, realistic split).")

    artifact = {
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": feature_columns,
        "nps_categories": NPS_CATEGORIES,
        "trained_on": "respondent_train (realistic split, docs/decisions.md D8)",
        "n_train_rows": len(train_df),
        "model_description": (
            "mord.LogisticAT (alpha=1.0, class-balanced sample weights), "
            "uncalibrated (docs/decisions.md D21; calibration rejected per D19)."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Saved -> {artifact_path}")

    reloaded = joblib.load(artifact_path)
    sample = X_train.head(20)
    assert (pipeline.predict(sample) == reloaded["pipeline"].predict(sample)).all(), (
        "Reloaded pipeline predictions do not match the freshly-fit pipeline."
    )
    print("OK: reload sanity check passed (predictions match).")


if __name__ == "__main__":
    main()
