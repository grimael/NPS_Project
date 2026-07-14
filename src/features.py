"""Feature engineering for NPS prediction.

Turns the labeled analytical dataset into a model-ready feature matrix.
Every feature is grouped (see FEATURE_GROUPS) so the rationale for each
one is traceable back to docs/decisions.md (entry D10).

Geographic features (Latitude, Longitude, Population) are kept in their
own group, `"geo_features"`, excluded by default from
`build_feature_matrix`. Zip Code / City are proxies for socio-economic
status and are never turned into features at all (not even in the geo
group) — see docs/decisions.md D10. Latitude/Longitude/Population are a
coarser, less identifying proxy but are still flagged for the fairness
audit (`src/fairness_audit.py`, phase 3) before being enabled by default.

This module only builds and selects features; it does not one-hot encode
or scale them (that belongs to the model-specific sklearn Pipeline in
notebooks/02_modeling.ipynb and src/train.py), except for filling the
two structurally-missing categoricals (`Offer`, `Internet Type`) with an
explicit "none" sentinel, since that is a feature-engineering decision
per docs/decisions.md D5.
"""
import numpy as np
import pandas as pd

from config import ID_COL, LEAKAGE_BLACKLIST, TARGET_COL, TARGET_COL_NOISY

# --------------------------------------------------------------------------
# Engineering constants
# --------------------------------------------------------------------------

# Counted as "active services" for Number of Active Services / Charges Per
# Service. Internet Service is handled separately (it's a 3-way category,
# not Yes/No) and added on top.
SERVICE_YES_NO_COLUMNS = [
    "Phone Service", "Multiple Lines", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Streaming Music", "Unlimited Data",
]
STREAMING_COLUMNS = ["Streaming TV", "Streaming Movies", "Streaming Music"]
SECURITY_COLUMNS = ["Online Security", "Online Backup", "Device Protection",
                     "Tech Support"]

TENURE_BUCKET_BINS = [-1, 12, 24, 48, 60, 72]
TENURE_BUCKET_LABELS = ["0-12", "13-24", "25-48", "49-60", "61-72"]

CONTRACT_LENGTH_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}

GEO_FEATURE_COLUMNS = ["Latitude", "Longitude", "Population"]

# --------------------------------------------------------------------------
# Feature groups (drives get_feature_columns / documentation / notebooks)
# --------------------------------------------------------------------------
FEATURE_GROUPS = {
    "tenure": ["Tenure Months", "Tenure Bucket"],
    "contract_billing": [
        "Contract", "Contract Length Months", "Payment Method",
        "Automatic Payment", "Paperless Billing",
    ],
    "charges": ["Monthly Charges", "Total Charges", "Charges Per Service"],
    "services": [
        "Number of Active Services", "Has Phone Service",
        "Has Internet Service", "Has Multiple Lines",
        "Has Streaming Bundle", "Has Security Bundle",
        "Internet Service", "Internet Type", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Streaming Music",
        "Unlimited Data",
    ],
    "engagement": ["Referred a Friend", "Number of Referrals", "Has Offer",
                   "Offer"],
    "household": [
        "Gender", "Senior Citizen", "Partner", "Under 30", "Age",
    ],
    "geo_features": GEO_FEATURE_COLUMNS,
}

# NOTE: "Dependents", "Number of Dependents", and the derived "Household
# Size Proxy" (still computed by add_engineered_features below) are
# deliberately NOT model features -- excluded per docs/decisions.md D28.
# The fairness audit (D27) found a large, statistically robust Detractor-
# recall gap by Dependents (19.2 points); dropping Dependents and Number
# of Dependents alone barely moved it (18.9 points), because Household
# Size Proxy = 1 + (Partner==Yes) + Number of Dependents re-encodes the
# same signal under a different name. Only dropping all three together
# meaningfully shrank the gap (11.8 points), at essentially no QWK cost
# (0.269 -> 0.273, a marginal improvement, not a trade-off). Left
# computed (not deleted outright) since src/fairness_audit.py's ablation
# still references them for reproducibility.
NUMERIC_FEATURES = [
    "Tenure Months", "Contract Length Months", "Monthly Charges",
    "Total Charges", "Charges Per Service", "Number of Active Services",
    "Automatic Payment", "Has Phone Service", "Has Internet Service",
    "Has Multiple Lines", "Has Streaming Bundle", "Has Security Bundle",
    "Number of Referrals", "Has Offer", "Age",
]
CATEGORICAL_FEATURES = [
    "Tenure Bucket", "Contract", "Payment Method", "Paperless Billing",
    "Internet Service", "Internet Type", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Streaming Music", "Unlimited Data", "Referred a Friend", "Offer",
    "Gender", "Senior Citizen", "Partner", "Under 30",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived column used by FEATURE_GROUPS. Returns a copy."""
    df = df.copy()

    # -- Tenure --
    df["Tenure Bucket"] = pd.cut(
        df["Tenure Months"], bins=TENURE_BUCKET_BINS,
        labels=TENURE_BUCKET_LABELS,
    ).astype(str)

    # -- Contract / billing --
    df["Contract Length Months"] = df["Contract"].map(CONTRACT_LENGTH_MONTHS)
    df["Automatic Payment"] = (
        df["Payment Method"].str.contains("automatic", case=False)
    ).astype(int)

    # -- Services / bundles --
    is_yes = df[SERVICE_YES_NO_COLUMNS].eq("Yes")
    has_internet = (df["Internet Service"] != "No").astype(int)
    df["Number of Active Services"] = is_yes.sum(axis=1) + has_internet
    df["Has Phone Service"] = (df["Phone Service"] == "Yes").astype(int)
    df["Has Internet Service"] = has_internet
    df["Has Multiple Lines"] = (df["Multiple Lines"] == "Yes").astype(int)
    df["Has Streaming Bundle"] = df[STREAMING_COLUMNS].eq("Yes").any(axis=1).astype(int)
    df["Has Security Bundle"] = df[SECURITY_COLUMNS].eq("Yes").any(axis=1).astype(int)
    # Structural NaN (no internet service at all) -> explicit sentinel,
    # consistent with the "No internet service" convention already used
    # by Online Security / Online Backup / etc. in the raw data.
    df["Internet Type"] = df["Internet Type"].fillna("No internet service")

    # -- Charges --
    active_services_safe = df["Number of Active Services"].clip(lower=1)
    df["Charges Per Service"] = df["Monthly Charges"] / active_services_safe

    # -- Engagement --
    df["Has Offer"] = df["Offer"].notna().astype(int)
    df["Offer"] = df["Offer"].fillna("No Offer")

    # -- Household composition proxy --
    # Rough proxy, not an exact household size: 1 (the customer) + partner
    # + recorded dependents. Documented as approximate in docs/decisions.md.
    df["Household Size Proxy"] = (
        1 + (df["Partner"] == "Yes").astype(int) + df["Number of Dependents"]
    )

    return df


def assert_no_leakage(columns: list[str]) -> None:
    leaked = sorted(set(columns) & set(LEAKAGE_BLACKLIST))
    if leaked:
        raise ValueError(f"Leakage-blacklisted columns in feature set: {leaked}")


def get_feature_columns(include_geo: bool = True) -> list[str]:
    groups = dict(FEATURE_GROUPS)
    if not include_geo:
        groups.pop("geo_features")
    columns = [col for group in groups.values() for col in group]
    assert_no_leakage(columns)
    return columns


def build_feature_matrix(df: pd.DataFrame, include_geo: bool = True) -> pd.DataFrame:
    """Labeled/analytical dataset -> feature matrix (+ Customer ID).

    Does not include the target column(s); join back on `Customer ID` if
    needed for supervised training.
    """
    df = add_engineered_features(df)
    feature_columns = get_feature_columns(include_geo=include_geo)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Expected engineered columns not found: {missing}")
    return df[[ID_COL] + feature_columns].copy()


if __name__ == "__main__":
    from config import LABELED_DATASET_PATH

    labeled = pd.read_parquet(LABELED_DATASET_PATH)

    for include_geo in (True, False):
        X = build_feature_matrix(labeled, include_geo=include_geo)
        n_features = X.shape[1] - 1  # minus Customer ID
        print(f"include_geo={include_geo}: {X.shape[0]} rows x {n_features} features")
        na_counts = X.isna().sum()
        na_counts = na_counts[na_counts > 0]
        if len(na_counts):
            print("  Remaining NaNs:\n" + na_counts.to_string())
        else:
            print("  No remaining NaNs.")

    all_columns = get_feature_columns(include_geo=True)
    assert not set(all_columns) & {TARGET_COL, TARGET_COL_NOISY, ID_COL}
    print(f"\nOK: {len(all_columns)} feature columns, no leakage, "
          f"target/ID columns not included among features.")
    print("\nFeature groups:")
    for group, cols in FEATURE_GROUPS.items():
        print(f"  {group} ({len(cols)}): {cols}")
