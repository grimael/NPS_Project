"""Tests for model training, inference, and serialization."""
import joblib
import numpy as np
from src.config import FINAL_MODEL_PATH, NPS_CATEGORIES
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix


def test_model_artifact_exists_and_loads():
    """Verify final model artifact can be loaded cleanly."""
    assert FINAL_MODEL_PATH.exists(), f"Model artifact not found at {FINAL_MODEL_PATH}"
    artifact = joblib.load(FINAL_MODEL_PATH)
    assert "pipeline" in artifact
    assert "feature_columns" in artifact


def test_model_inference_probabilities_and_labels(labeled_data):
    """Verify that predictions match classes and probabilities sum to 1.0."""
    artifact = joblib.load(FINAL_MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = build_feature_matrix(labeled_data.head(50), include_geo=False)[feature_cols]

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    # Check predictions
    assert len(preds) == len(X)
    assert set(preds).issubset(set(NPS_CATEGORIES))

    # Check probabilities
    assert probas.shape == (len(X), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
