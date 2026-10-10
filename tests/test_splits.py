"""Tests for data splits protocol and distribution shift verification."""
from src.config import ID_COL


def test_split_completeness_and_uniqueness(labeled_data, realistic_splits):
    """Ensure all customers are uniquely assigned to one split partition."""
    assert len(realistic_splits) == len(labeled_data), "Split row count mismatch"
    assert realistic_splits[ID_COL].nunique() == len(labeled_data), "Duplicate customer IDs in split"

    valid_splits = {"respondent_train", "silent_val", "silent_test"}
    assert set(realistic_splits["split"].unique()) == valid_splits


def test_split_partition_sizes(realistic_splits):
    """Ensure respondent_train is ~15% and silent population is divided ~50/50."""
    total = len(realistic_splits)
    counts = realistic_splits["split"].value_counts()

    # ~15% respondents
    assert 0.14 <= counts["respondent_train"] / total <= 0.16
    # 50/50 partition of silent cohort
    assert abs(counts["silent_val"] - counts["silent_test"]) <= 5


def test_distribution_shift_properties(labeled_data, realistic_splits):
    """Verify that respondents exhibit the documented response bias (lower satisfaction, higher tenure)."""
    merged = labeled_data.merge(realistic_splits, on=ID_COL)
    resp = merged[merged["split"] == "respondent_train"]
    silent = merged[merged["split"].isin(["silent_val", "silent_test"])]

    # Respondents must have lower satisfaction than silent cohort
    assert resp["Satisfaction Score"].mean() < silent["Satisfaction Score"].mean()
    # Respondents must have higher tenure than silent cohort
    assert resp["Tenure Months"].mean() > silent["Tenure Months"].mean()
