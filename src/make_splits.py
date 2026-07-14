"""Build the two evaluation splits used throughout the challenge.

(a) "realistic" split: simulates the deployment scenario. The model will
    score customers who never answered the satisfaction survey. We
    sub-sample ~15% of customers as "respondents" (used for training) with
    a response propensity that is HIGHER for customers with LOWER
    satisfaction and HIGHER tenure — the intuition being that dissatisfied
    customers are more motivated to complain, and long-tenured customers
    are more engaged with the brand. The remaining ~85% are the "silent"
    customers, held out as the test set. This response-bias mechanism is a
    modeling choice, not observed data — see docs/decisions.md.

(b) "random" split: stratified random 70/30 on the baseline NPS label.
    A comparison point, not the reference protocol (rule #4).

Both splits are computed on Customer ID and saved as index tables, so they
can be joined back onto any version of the labeled dataset.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ID_COL,
    LABELED_DATASET_PATH,
    RANDOM_SPLIT_PATH,
    RANDOM_SPLIT_TEST_SIZE,
    REALISTIC_SPLIT_PATH,
    RESPONDENT_FRACTION,
    SEED,
    TARGET_COL,
)

# Relative weight of each driver in the response-propensity logit. Equal
# weighting is a simplifying choice: both effects are documented as
# directionally important, but the challenge does not specify their
# relative strength.
SATISFACTION_WEIGHT = 1.0
TENURE_WEIGHT = 1.0


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def compute_response_propensity(df: pd.DataFrame) -> np.ndarray:
    z_low_satisfaction = -zscore(df["Satisfaction Score"])
    z_high_tenure = zscore(df["Tenure Months"])
    logit = SATISFACTION_WEIGHT * z_low_satisfaction + TENURE_WEIGHT * z_high_tenure
    weights = np.exp(logit - logit.max())  # stabilized softmax-style weights
    return weights / weights.sum()


def make_realistic_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    n_respondents = round(RESPONDENT_FRACTION * len(df))
    weights = compute_response_propensity(df)

    rng = np.random.RandomState(seed)
    respondent_pos = rng.choice(len(df), size=n_respondents, replace=False, p=weights)
    respondent_ids = set(df.iloc[respondent_pos][ID_COL])

    split = pd.DataFrame({
        ID_COL: df[ID_COL],
        "split": np.where(df[ID_COL].isin(respondent_ids), "respondent_train", "silent_test"),
    })

    # Sanity check: respondents should indeed skew low-satisfaction /
    # high-tenure relative to the full population.
    resp = df[df[ID_COL].isin(respondent_ids)]
    print(f"Realistic split: {len(resp)} respondents ({len(resp) / len(df):.1%}), "
          f"{len(df) - len(resp)} silent.")
    print(f"  Mean Satisfaction Score  -- respondents: {resp['Satisfaction Score'].mean():.2f} "
          f"| overall: {df['Satisfaction Score'].mean():.2f}")
    print(f"  Mean Tenure Months       -- respondents: {resp['Tenure Months'].mean():.1f} "
          f"| overall: {df['Tenure Months'].mean():.1f}")
    print("  Respondent NPS Category distribution:")
    print(resp[TARGET_COL].value_counts(normalize=True).round(3).to_string())
    return split


def make_random_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
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
