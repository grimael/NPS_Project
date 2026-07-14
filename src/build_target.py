"""Satisfaction Score -> NPS Category label.

Produces two target variants from the same analytical dataset:
- baseline (config.SATISFACTION_TO_NPS): deterministic mapping.
- noisy: baseline with ~10% of labels reassigned to an *adjacent* ordinal
  class (seeded), used later as a sensitivity test — how much does model
  ranking change if the label itself is a noisy proxy for true NPS?

Writes data/processed/labeled_dataset.parquet (analytical dataset + both
target variants, as category and as ordinal code).
"""
import numpy as np
import pandas as pd

from config import (
    ANALYTICAL_DATASET_PATH,
    LABELED_DATASET_PATH,
    NOISY_LABEL_REASSIGNMENT_RATE,
    NPS_CATEGORIES,
    NPS_CATEGORY_TO_CODE,
    SATISFACTION_TO_NPS,
    SEED,
    TARGET_COL,
    TARGET_COL_NOISY,
)

# Ordinal neighbors for noise injection: Detractor <-> Passive <-> Promoter.
ADJACENT_CATEGORIES = {
    "Detractor": ["Passive"],
    "Passive": ["Detractor", "Promoter"],
    "Promoter": ["Passive"],
}


def add_baseline_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = df["Satisfaction Score"].map(SATISFACTION_TO_NPS)
    if df[TARGET_COL].isna().any():
        raise ValueError("Some Satisfaction Score values did not map to a category.")
    df[TARGET_COL] = pd.Categorical(df[TARGET_COL], categories=NPS_CATEGORIES, ordered=True)
    return df


def add_noisy_target(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.RandomState(seed)
    n = len(df)
    flip_mask = rng.random_sample(n) < rate

    noisy = df[TARGET_COL].astype(str).to_numpy(copy=True)
    flip_idx = np.flatnonzero(flip_mask)
    for i in flip_idx:
        options = ADJACENT_CATEGORIES[noisy[i]]
        noisy[i] = options[0] if len(options) == 1 else options[rng.randint(len(options))]

    df[TARGET_COL_NOISY] = pd.Categorical(noisy, categories=NPS_CATEGORIES, ordered=True)
    n_changed = (df[TARGET_COL_NOISY].astype(str) != df[TARGET_COL].astype(str)).sum()
    print(f"OK: noisy target reassigned {n_changed}/{n} rows "
          f"({n_changed / n:.1%}), target rate was {rate:.0%}.")
    return df


def add_ordinal_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[f"{TARGET_COL} Code"] = df[TARGET_COL].astype(str).map(NPS_CATEGORY_TO_CODE)
    df[f"{TARGET_COL_NOISY} Code"] = df[TARGET_COL_NOISY].astype(str).map(NPS_CATEGORY_TO_CODE)
    return df


def print_distribution(df: pd.DataFrame, col: str) -> None:
    counts = df[col].value_counts().reindex(NPS_CATEGORIES)
    pct = (counts / len(df) * 100).round(1)
    print(f"\n{col} distribution:")
    for cat in NPS_CATEGORIES:
        print(f"  {cat:<10s} {counts[cat]:>5d}  ({pct[cat]:>5.1f}%)")


def main() -> None:
    df = pd.read_parquet(ANALYTICAL_DATASET_PATH)

    df = add_baseline_target(df)
    df = add_noisy_target(df, NOISY_LABEL_REASSIGNMENT_RATE, SEED)
    df = add_ordinal_codes(df)

    print_distribution(df, TARGET_COL)
    print_distribution(df, TARGET_COL_NOISY)

    LABELED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LABELED_DATASET_PATH, index=False)
    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} columns -> {LABELED_DATASET_PATH}")


if __name__ == "__main__":
    main()
