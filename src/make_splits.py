"""Build evaluation splits for the NPS prediction pipeline.

Grounding and Evidence:
-----------------------
1. Realistic Split (Deployment Scenario):
   - Only ~15% of telecom customers respond to NPS surveys (TM Forum 2021).
   - Voluntary respondents exhibit negative/extreme response bias (Anderson 1998,
     "Customer Satisfaction and Word of Mouth", JSR 1(1):5-17): dissatisfied customers
     have higher motivation to complain, while long-tenured customers have higher
     brand engagement.
   - We model response propensity using standard logistic response weighting on
     standardized satisfaction (-z) and tenure (+z).
   
2. Leak-Free 3-Way Split (Airtight Evaluation):
   - To eliminate "test-set snooping" (using the test set to iterate on modeling,
     tuning, calibration, and feature selection), the 85% silent base is cleanly
     divided into:
     a) `silent_val` (50% of silent base): used for validating out-of-distribution
        generalization, hyperparameter tuning, calibration choices, and fairness ablations.
     b) `silent_test` (50% of silent base): strictly frozen and held out for final
        unbiased reporting.
   - Pre-computed 5-fold stratified CV on `respondent_train` for internal model selection.

3. Random Split (Baseline Reference):
   - Standard stratified 70/30 split, kept strictly as an in-distribution reference.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from src.config import (
        ID_COL,
        LABELED_DATASET_PATH,
        RANDOM_SPLIT_PATH,
        RANDOM_SPLIT_TEST_SIZE,
        REALISTIC_SPLIT_PATH,
        RESPONDENT_FRACTION,
        SEED,
        SILENT_TEST_FRACTION,
        TARGET_COL,
    )
except ImportError:
    from config import (
        ID_COL,
        LABELED_DATASET_PATH,
        RANDOM_SPLIT_PATH,
        RANDOM_SPLIT_TEST_SIZE,
        REALISTIC_SPLIT_PATH,
        RESPONDENT_FRACTION,
        SEED,
        SILENT_TEST_FRACTION,
        TARGET_COL,
    )

# Relative weight of each driver in response propensity logit:
# Grounded in Anderson (1998) and Kannan et al. (2022) telecom response patterns.
SATISFACTION_WEIGHT = 1.0
TENURE_WEIGHT = 1.0


def zscore(s: pd.Series) -> pd.Series:
    """Standardize series with ddof=0."""
    return (s - s.mean()) / s.std(ddof=0)


def compute_response_propensity(df: pd.DataFrame) -> np.ndarray:
    """Compute voluntary survey response probabilities based on satisfaction and tenure."""
    z_low_satisfaction = -zscore(df["Satisfaction Score"])
    z_high_tenure = zscore(df["Tenure Months"])
    logit = SATISFACTION_WEIGHT * z_low_satisfaction + TENURE_WEIGHT * z_high_tenure
    weights = np.exp(logit - logit.max())  # stabilized softmax weights
    return weights / weights.sum()


def make_realistic_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a leak-free 3-way split: respondent_train, silent_val, and silent_test."""
    n_respondents = round(RESPONDENT_FRACTION * len(df))
    weights = compute_response_propensity(df)

    rng = np.random.RandomState(seed)
    respondent_pos = rng.choice(len(df), size=n_respondents, replace=False, p=weights)
    respondent_ids = set(df.iloc[respondent_pos][ID_COL])

    # Partition the remaining 85% silent population into silent_val and silent_test
    silent_mask = ~df[ID_COL].isin(respondent_ids)
    silent_df = df[silent_mask]

    silent_val_ids, silent_test_ids = train_test_split(
        silent_df[ID_COL],
        test_size=SILENT_TEST_FRACTION,
        stratify=silent_df[TARGET_COL],
        random_state=seed,
    )
    val_set = set(silent_val_ids)
    test_set = set(silent_test_ids)

    split_col = []
    for cid in df[ID_COL]:
        if cid in respondent_ids:
            split_col.append("respondent_train")
        elif cid in val_set:
            split_col.append("silent_val")
        else:
            split_col.append("silent_test")

    split_df = pd.DataFrame({
        ID_COL: df[ID_COL],
        "split": split_col,
    })

    # Pre-compute 5-fold Stratified CV for respondent_train
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    resp_df = df[df[ID_COL].isin(respondent_ids)].copy().reset_index(drop=True)
    resp_df["cv_fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(resp_df, resp_df[TARGET_COL])):
        resp_df.loc[val_idx, "cv_fold"] = fold_idx

    split_df = split_df.merge(resp_df[[ID_COL, "cv_fold"]], on=ID_COL, how="left")
    split_df["cv_fold"] = split_df["cv_fold"].fillna(-1).astype(int)

    # Sanity checks and summary printing
    resp = df[df[ID_COL].isin(respondent_ids)]
    s_val = df[df[ID_COL].isin(val_set)]
    s_test = df[df[ID_COL].isin(test_set)]

    print(f"Realistic 3-way split:")
    print(f"  - respondent_train: {len(resp)} ({len(resp) / len(df):.1%}) [with 5-fold CV]")
    print(f"  - silent_val:       {len(s_val)} ({len(s_val) / len(df):.1%}) [for tuning/shift validation]")
    print(f"  - silent_test:      {len(s_test)} ({len(s_test) / len(df):.1%}) [strictly frozen holdout test]")
    print(f"\n  Mean Satisfaction Score:")
    print(f"    respondents: {resp['Satisfaction Score'].mean():.2f} | silent_val: {s_val['Satisfaction Score'].mean():.2f} | overall: {df['Satisfaction Score'].mean():.2f}")
    print(f"  Mean Tenure Months:")
    print(f"    respondents: {resp['Tenure Months'].mean():.1f} | silent_val: {s_val['Tenure Months'].mean():.1f} | overall: {df['Tenure Months'].mean():.1f}")
    print("\n  Respondent NPS Category distribution:")
    print(resp[TARGET_COL].value_counts(normalize=True).round(3).to_string())
    print("\n  Silent Val NPS Category distribution:")
    print(s_val[TARGET_COL].value_counts(normalize=True).round(3).to_string())

    return split_df


def make_random_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Standard stratified random 70/30 split for baseline comparison."""
    train_ids, test_ids = train_test_split(
        df[ID_COL],
        test_size=RANDOM_SPLIT_TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=seed,
    )
    split = pd.DataFrame({
        ID_COL: df[ID_COL],
        "split": np.where(df[ID_COL].isin(set(train_ids)), "train", "test"),
    })
    print(f"\nRandom split: {len(train_ids)} train ({1 - RANDOM_SPLIT_TEST_SIZE:.0%}), "
          f"{len(test_ids)} test ({RANDOM_SPLIT_TEST_SIZE:.0%}).")
    return split


def main() -> None:
    df = pd.read_parquet(LABELED_DATASET_PATH)

    realistic = make_realistic_split(df, SEED)
    random_split = make_random_split(df, SEED)

    REALISTIC_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    realistic.to_parquet(REALISTIC_SPLIT_PATH, index=False)
    random_split.to_parquet(RANDOM_SPLIT_PATH, index=False)
    print(f"\nSaved -> {REALISTIC_SPLIT_PATH}")
    print(f"Saved -> {RANDOM_SPLIT_PATH}")


if __name__ == "__main__":
    main()
