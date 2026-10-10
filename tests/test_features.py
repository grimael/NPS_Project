"""Tests for feature engineering correctness."""
import pandas as pd
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
    build_feature_matrix,
)


def test_engineered_features_structure(labeled_data):
    """Test that feature matrix has expected columns without nulls in key engineered metrics."""
    X = build_feature_matrix(labeled_data, include_geo=False)

    for col in NUMERIC_FEATURES:
        assert col in X.columns, f"Missing numeric feature: {col}"
        assert not X[col].isna().any(), f"Unexpected null values in {col}"

    for col in CATEGORICAL_FEATURES:
        assert col in X.columns, f"Missing categorical feature: {col}"


def test_edge_cases_in_features():
    """Test feature engineering with brand-new customers (tenure=0)."""
    edge_row = pd.DataFrame([{
        "Customer ID": "TEST-001",
        "Tenure Months": 0,
        "Contract": "Month-to-month",
        "Payment Method": "Electronic check",
        "Paperless Billing": "No",
        "Monthly Charges": 50.0,
        "Total Charges": 0.0,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Internet Type": "DSL",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Offer": "No Offer",
        "Gender": "Female",
        "Age": 25,
        "Under 30": "Yes",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Number of Dependents": 0,
    }])

    engineered = add_engineered_features(edge_row)
    assert engineered["Tenure Bucket"].iloc[0] == "0-12"
    assert engineered["Charges Per Service"].iloc[0] > 0
    assert not engineered["Charges Per Service"].isna().any()
