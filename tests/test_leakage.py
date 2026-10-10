"""Tests to guarantee absolute absence of target leakage."""
from src.config import LEAKAGE_BLACKLIST
from src.features import build_feature_matrix


def test_leakage_blacklist_strictly_excluded(labeled_data):
    """Ensure no leakage-blacklisted column ever enters the feature matrix."""
    X = build_feature_matrix(labeled_data, include_geo=True)

    leaked = [col for col in LEAKAGE_BLACKLIST if col in X.columns]
    assert not leaked, f"CRITICAL: Leakage columns found in feature matrix: {leaked}"


def test_satisfaction_score_not_in_features(labeled_data):
    """Ensure Satisfaction Score (label source) is not in feature matrix."""
    X = build_feature_matrix(labeled_data, include_geo=False)
    assert "Satisfaction Score" not in X.columns


def test_churn_outcomes_not_in_features(labeled_data):
    """Ensure future churn outcomes are not in feature matrix."""
    X = build_feature_matrix(labeled_data, include_geo=False)
    churn_cols = [c for c in X.columns if "churn" in c.lower() or "cltv" in c.lower()]
    assert not churn_cols, f"Churn outcome columns found: {churn_cols}"
