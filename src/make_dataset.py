"""Raw IBM Telco tables -> single analytical dataset.

Loads the six IBM Telco Customer Churn v11.1.3+ tables, joins them on
Customer ID (plus Population on Zip Code), resolves column overlaps
between tables, and writes data/processed/analytical_dataset.parquet.

Deterministic: no randomness involved, safe to re-run.

Table roles (see docs/decisions.md for the full rationale):
- churn:        base table. Canonical source for tenure, monthly/total
                charges, and the core service/contract/billing columns.
                Chosen as base because it is the table for which the
                classic "Total Charges blank when tenure == 0" quirk
                needs handling.
- status:       sole source of every leakage-blacklisted column
                (Satisfaction Score, Customer Status, Churn *, CLTV), so
                that all outcome-adjacent fields come from one
                internally consistent snapshot.
- demographics: adds Age, Under 30, Number of Dependents (not in churn).
- services:     adds usage/engagement columns not in churn (referrals,
                offer, long-distance/data usage, revenue, refunds,
                Internet Type).
- population:   adds zip-level Population, joined on Zip Code.
- location:     NOT merged. Redundant with churn's own location columns
                and disagrees with them on ~3.7% of rows (different data
                pull timestamp within the same IBM release). churn's
                version is kept as canonical since churn is already the
                base table.
"""
import pandas as pd

from config import ANALYTICAL_DATASET_PATH, ID_COL, LEAKAGE_BLACKLIST, RAW_FILES


def load_raw_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file for '{name}': {path}")
        tables[name] = pd.read_excel(path)
    return tables


def check_satisfaction_score(status: pd.DataFrame) -> None:
    if "Satisfaction Score" not in status.columns:
        raise ValueError(
            "'Satisfaction Score' column not found in the status table. "
            "This dataset must be the IBM Telco v11.1.3+ release (the "
            "Kaggle 'blastchar' flat file does not have this column)."
        )
    n_missing = status["Satisfaction Score"].isna().sum()
    if n_missing:
        raise ValueError(f"Satisfaction Score has {n_missing} missing values.")
    out_of_range = ~status["Satisfaction Score"].between(1, 5)
    if out_of_range.any():
        raise ValueError("Satisfaction Score has values outside the 1-5 range.")
    print(f"OK: Satisfaction Score present, complete, in [1, 5] "
          f"({len(status)} rows).")


def build_base_table(churn: pd.DataFrame) -> pd.DataFrame:
    """churn.xlsx -> base table: rename ID, drop constant/redundant cols."""
    df = churn.rename(columns={"CustomerID": ID_COL})
    # Constant across all 7043 rows -> no discriminative power.
    df = df.drop(columns=["Count", "Country", "State"])
    # Outcome-adjacent columns: dropped here, re-added from `status` only,
    # so every blacklisted column comes from a single consistent table.
    df = df.drop(columns=[
        "Churn Label", "Churn Value", "Churn Score", "CLTV", "Churn Reason",
    ])
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Total Charges is blank (space string) for brand-new customers with
    Tenure Months == 0. Coerce to numeric and fill those with 0.
    """
    df = df.copy()
    total_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
    n_blank = total_charges.isna().sum()
    zero_tenure = df["Tenure Months"] == 0
    if n_blank != zero_tenure.sum() or not (total_charges.isna() == zero_tenure).all():
        raise ValueError(
            "Unexpected mismatch between blank Total Charges and "
            "Tenure Months == 0; investigate before proceeding."
        )
    total_charges = total_charges.fillna(0.0)
    df["Total Charges"] = total_charges
    print(f"OK: filled {n_blank} blank Total Charges (Tenure Months == 0) with 0.")
    return df


def add_status_columns(df: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    keep = [ID_COL, "Satisfaction Score", "Customer Status", "Churn Label",
            "Churn Value", "Churn Score", "CLTV", "Churn Category",
            "Churn Reason"]
    return df.merge(status[keep], on=ID_COL, how="left", validate="one_to_one")


def add_demographics_columns(df: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    # Gender / Senior Citizen / Dependents match churn's columns 100%;
    # Married is a 1:1 relabeling of churn's Partner. Only genuinely new
    # columns are added here.
    keep = [ID_COL, "Age", "Under 30", "Number of Dependents"]
    return df.merge(demo[keep], on=ID_COL, how="left", validate="one_to_one")


def add_services_columns(df: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    keep = [
        ID_COL, "Referred a Friend", "Number of Referrals", "Offer",
        "Avg Monthly Long Distance Charges", "Internet Type",
        "Avg Monthly GB Download", "Streaming Music", "Unlimited Data",
        "Total Refunds", "Total Extra Data Charges",
        "Total Long Distance Charges", "Total Revenue",
    ]
    merged = df.merge(services[keep], on=ID_COL, how="left", validate="one_to_one")

    # Known caveat: for 11 customers, churn's Tenure Months == 0 while
    # services.xlsx reports Tenure in Months == 10 with non-zero usage,
    # i.e. these engagement columns were computed under a different tenure
    # assumption than the one used as canonical here. Flag, don't silently
    # fix (see docs/decisions.md).
    mismatched = merged[ID_COL][df["Tenure Months"] == 0]
    services_tenure = services.set_index(ID_COL)["Tenure in Months"]
    n_conflict = (services_tenure.reindex(mismatched) != 0).sum()
    if n_conflict:
        print(f"WARNING: {n_conflict} customers have Tenure Months == 0 in "
              f"the base table but non-zero tenure in services.xlsx; their "
              f"services-derived columns (Total Revenue, etc.) reflect that "
              f"later snapshot. See docs/decisions.md.")
    return merged


def add_population_column(df: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    pop = population[["Zip Code", "Population"]].drop_duplicates("Zip Code")
    return df.merge(pop, on="Zip Code", how="left", validate="many_to_one")


def main() -> None:
    tables = load_raw_tables()
    check_satisfaction_score(tables["status"])

    df = build_base_table(tables["churn"])
    df = fix_total_charges(df)
    df = add_status_columns(df, tables["status"])
    df = add_demographics_columns(df, tables["demographics"])
    df = add_services_columns(df, tables["services"])
    df = add_population_column(df, tables["population"])

    assert df[ID_COL].is_unique, "Customer ID is not unique after joins."
    assert len(df) == 7043, f"Expected 7043 rows, got {len(df)}."
    for col in LEAKAGE_BLACKLIST:
        assert col in df.columns, f"Blacklisted column missing: {col}"

    ANALYTICAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ANALYTICAL_DATASET_PATH, index=False)

    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} columns -> "
          f"{ANALYTICAL_DATASET_PATH}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        flag = " [LEAKAGE-BLACKLIST]" if col in LEAKAGE_BLACKLIST else ""
        print(f"  - {col}{flag}")
    print("\nMissing values (non-zero only):")
    na_counts = df.isna().sum()
    print(na_counts[na_counts > 0].to_string())


if __name__ == "__main__":
    main()
