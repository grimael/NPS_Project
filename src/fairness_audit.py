"""Fairness audit (challenge brief section 4.7).

Two questions, both scored on `silent_test` (the realistic split's silent
85% -- the model's actual deployment target, docs/decisions.md D8), using
a pipeline trained on `respondent_train` exactly like `train.py`:

1. Does the deployed model (`docs/decisions.md` D21, no geo features) have
   a Detractor-recall gap across demographic groups? Checked directly via
   per-group recall, not inferred from feature importance -- D23 already
   found no demographic feature in the top-5 SHAP drivers of any segment,
   but explicitly flagged that finding as preliminary, not a substitute
   for this audit.
2. Does adding the geo features held back since D11 (Latitude, Longitude,
   Population -- socio-economic proxies) change overall performance or
   group disparity? This is the ablation D11 deferred until this audit
   existed. `City`/`Zip Code` are never used as features at all, in
   either arm (D10) -- not re-litigated here.


"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from evaluate import evaluate
from features import CATEGORICAL_FEATURES, GEO_FEATURE_COLUMNS, NUMERIC_FEATURES, build_feature_matrix
from train import OrdinalLogisticClassifier

FAIRNESS_GROUPS = ["Senior Citizen", "Gender", "Partner", "Dependents"]
NPS_COLORS = {"Detractor": "#C44E52", "Passive": "#DD8452", "Promoter": "#55A868"}


def load_realistic_split() -> pd.DataFrame:
    labeled = pd.read_parquet(config.LABELED_DATASET_PATH)
    split = pd.read_parquet(config.REALISTIC_SPLIT_PATH)
    return labeled.merge(split, on=config.ID_COL)


def build_pipeline(include_geo: bool, exclude_features: list[str] | None = None) -> tuple[Pipeline, list[str]]:
    """Mirrors train.py:build_final_pipeline, with geo features optionally
    added to the numeric transformer -- train.py itself never includes
    them (D11), so this is a local variant, not a change to train.py.
    `exclude_features` additionally drops named columns (numeric or
    categorical) -- used for the Dependents feature-removal ablation."""
    exclude = set(exclude_features or [])
    numeric = [c for c in NUMERIC_FEATURES + (GEO_FEATURE_COLUMNS if include_geo else []) if c not in exclude]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in exclude]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", OrdinalLogisticClassifier(alpha=1.0, balanced=True)),
    ])
    return pipeline, numeric + categorical


def fit_and_score(data: pd.DataFrame, include_geo: bool, exclude_features: list[str] | None = None):
    # FAIRNESS_GROUPS columns (Gender, Senior Citizen, Partner, Dependents)
    # may or may not already be present in X as CATEGORICAL_FEATURES,
    # depending on the model's current feature set (docs/decisions.md
    # D28 dropped Dependents from the shipped model) -- only bring in
    # whichever ones are actually missing, to avoid merge-suffix
    # collisions on the ones that are still model features.
    X = build_feature_matrix(data, include_geo=include_geo)
    missing_group_cols = [c for c in FAIRNESS_GROUPS if c not in X.columns]
    extra_cols = data[[config.ID_COL, "split", config.TARGET_COL] + missing_group_cols]
    merged = X.merge(extra_cols, on=config.ID_COL)

    train_df = merged[merged["split"] == "respondent_train"]
    test_df = merged[merged["split"] == "silent_test"]

    pipeline, feature_columns = build_pipeline(include_geo, exclude_features)
    pipeline.fit(train_df[feature_columns], train_df[config.TARGET_COL].astype(str))

    y_test = test_df[config.TARGET_COL].astype(str).to_numpy()
    y_pred = pipeline.predict(test_df[feature_columns])
    y_proba = pipeline.predict_proba(test_df[feature_columns])
    return pipeline, test_df.reset_index(drop=True), y_test, y_pred, y_proba


def per_group_table(test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray,
                     model_name: str) -> pd.DataFrame:
    rows = []
    for group_col in FAIRNESS_GROUPS:
        values = test_df[group_col].to_numpy()
        for v in sorted(pd.unique(values)):
            mask = values == v
            res = evaluate(y_test[mask], y_pred[mask])
            row = {
                "model": model_name, "group": group_col, "value": v, "n": int(mask.sum()),
                "macro_f1": res["macro_f1"], "balanced_accuracy": res["balanced_accuracy"],
                "qwk": res["qwk"],
            }
            row.update({f"recall_{c}": res["recall_per_class"][c] for c in config.NPS_CATEGORIES})
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_two_sample_diff_ci(y_true_a, y_pred_a, y_true_b, y_pred_b, metric_fn,
                                  n_boot: int = 2000, seed: int = config.SEED, ci: float = 0.95) -> dict:
    """Independent two-sample percentile bootstrap for metric(group_a) -
    metric(group_b). Unlike evaluate.bootstrap_paired_diff_ci (same rows,
    two models), group A and group B here are disjoint customer subsets,
    so each is resampled independently."""
    y_true_a, y_pred_a = np.asarray(y_true_a), np.asarray(y_pred_a)
    y_true_b, y_pred_b = np.asarray(y_true_b), np.asarray(y_pred_b)
    rng = np.random.RandomState(seed)
    n_a, n_b = len(y_true_a), len(y_true_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.randint(0, n_a, n_a)
        idx_b = rng.randint(0, n_b, n_b)
        diffs[i] = metric_fn(y_true_a[idx_a], y_pred_a[idx_a]) - metric_fn(y_true_b[idx_b], y_pred_b[idx_b])
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    point = metric_fn(y_true_a, y_pred_a) - metric_fn(y_true_b, y_pred_b)
    return {"point": point, "lower": lower, "upper": upper}


def detractor_recall(y_true, y_pred) -> float:
    mask_true = y_true == "Detractor"
    if mask_true.sum() == 0:
        return float("nan")
    return float((y_pred[mask_true] == "Detractor").mean())


def plot_group_recall(table: pd.DataFrame, model_name: str, path) -> None:
    import matplotlib.pyplot as plt
    sub = table[table["model"] == model_name].copy()
    sub["label"] = sub["group"] + " = " + sub["value"].astype(str)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(sub))
    bars = ax.barh(y_pos, sub["recall_Detractor"], color=NPS_COLORS["Detractor"])
    for bar, val, n in zip(bars, sub["recall_Detractor"], sub["n"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.1%} (n={n})",
                va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["label"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Detractor recall")
    ax.set_title(f"Detractor recall by demographic group -- {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    data = load_realistic_split()

    print("=== Fitting baseline model (no geo, matches models/final_model.joblib) ===")
    _, test_df, y_test, y_pred, y_proba = fit_and_score(data, include_geo=False)
    overall = evaluate(y_test, y_pred, y_proba)
    print(f"Overall (silent_test, no geo): QWK={overall['qwk']:.3f} "
          f"macro-F1={overall['macro_f1']:.3f} "
          f"recall_Detractor={overall['recall_per_class']['Detractor']:.3f}")

    table_baseline = per_group_table(test_df, y_test, y_pred, "no_geo")
    print("\nPer-group Detractor recall (no geo):")
    print(table_baseline[["group", "value", "n", "recall_Detractor", "macro_f1"]]
          .to_string(index=False))

    print("\n=== Fitting geo-ablation model (Latitude/Longitude/Population added) ===")
    _, test_df_geo, y_test_geo, y_pred_geo, y_proba_geo = fit_and_score(data, include_geo=True)
    overall_geo = evaluate(y_test_geo, y_pred_geo, y_proba_geo)
    print(f"Overall (silent_test, with geo): QWK={overall_geo['qwk']:.3f} "
          f"macro-F1={overall_geo['macro_f1']:.3f} "
          f"recall_Detractor={overall_geo['recall_per_class']['Detractor']:.3f}")

    table_geo = per_group_table(test_df_geo, y_test_geo, y_pred_geo, "with_geo")
    print("\nPer-group Detractor recall (with geo):")
    print(table_geo[["group", "value", "n", "recall_Detractor", "macro_f1"]]
          .to_string(index=False))

    print("\n=== Largest within-group Detractor-recall gap, bootstrap 95% CI ===")
    gap_rows = []
    for group_col in FAIRNESS_GROUPS:
        values = test_df[group_col].to_numpy()
        uniques = sorted(pd.unique(values))
        assert len(uniques) == 2, f"{group_col} has {len(uniques)} categories, expected 2"
        v_a, v_b = uniques
        mask_a, mask_b = values == v_a, values == v_b
        diff = bootstrap_two_sample_diff_ci(
            y_test[mask_a], y_pred[mask_a], y_test[mask_b], y_pred[mask_b],
            detractor_recall, n_boot=2000,
        )
        gap_rows.append({"group": group_col, "value_a": v_a, "value_b": v_b, **diff})
        print(f"{group_col:16s} {v_a}={detractor_recall(y_test[mask_a], y_pred[mask_a]):.3f} "
              f"vs {v_b}={detractor_recall(y_test[mask_b], y_pred[mask_b]):.3f} | "
              f"diff({v_a}-{v_b})={diff['point']:+.3f} 95% CI [{diff['lower']:+.3f}, {diff['upper']:+.3f}]")
    gap_table = pd.DataFrame(gap_rows)

    print("\n=== Dependents-removal ablation (does the gap survive without the feature?) ===")
    dep_ablations = {
        "with_feature (baseline)": [],
        "minus Dependents + Number of Dependents": ["Dependents", "Number of Dependents"],
        "minus Dependents + Number of Dependents + Household Size Proxy":
            ["Dependents", "Number of Dependents", "Household Size Proxy"],
    }
    ablation_rows = []
    for label, exclude in dep_ablations.items():
        if label == "with_feature (baseline)":
            t_df, yt, yp, yproba = test_df, y_test, y_pred, y_proba
        else:
            _, t_df, yt, yp, yproba = fit_and_score(data, include_geo=False, exclude_features=exclude)
        res = evaluate(yt, yp, yproba)
        values = t_df["Dependents"].to_numpy()
        mask_a, mask_b = values == "No", values == "Yes"
        diff = bootstrap_two_sample_diff_ci(
            yt[mask_a], yp[mask_a], yt[mask_b], yp[mask_b], detractor_recall, n_boot=2000,
        )
        recall_a = detractor_recall(yt[mask_a], yp[mask_a])
        recall_b = detractor_recall(yt[mask_b], yp[mask_b])
        print(f"{label:60s} QWK={res['qwk']:.3f} | No={recall_a:.3f} vs Yes={recall_b:.3f} | "
              f"diff={diff['point']:+.3f} 95% CI [{diff['lower']:+.3f}, {diff['upper']:+.3f}]")
        ablation_rows.append({
            "ablation": label, "qwk": res["qwk"], "macro_f1": res["macro_f1"],
            "recall_Detractor_No": recall_a, "recall_Detractor_Yes": recall_b,
            "gap": diff["point"], "gap_ci_lower": diff["lower"], "gap_ci_upper": diff["upper"],
        })
    ablation_table = pd.DataFrame(ablation_rows)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full_table = pd.concat([table_baseline, table_geo], ignore_index=True)
    full_table_path = config.REPORTS_DIR / "fairness_audit.csv"
    full_table.to_csv(full_table_path, index=False)
    gap_table_path = config.REPORTS_DIR / "fairness_audit_gaps.csv"
    gap_table.to_csv(gap_table_path, index=False)
    ablation_table_path = config.REPORTS_DIR / "fairness_audit_dependents_ablation.csv"
    ablation_table.to_csv(ablation_table_path, index=False)
    print(f"\nSaved -> {full_table_path}")
    print(f"Saved -> {gap_table_path}")
    print(f"Saved -> {ablation_table_path}")

    fig_path_baseline = config.REPORTS_FIGURES_DIR / "fairness_detractor_recall_no_geo.png"
    plot_group_recall(table_baseline, "no_geo", fig_path_baseline)
    fig_path_geo = config.REPORTS_FIGURES_DIR / "fairness_detractor_recall_with_geo.png"
    plot_group_recall(table_geo, "with_geo", fig_path_geo)
    print(f"Saved -> {fig_path_baseline}")
    print(f"Saved -> {fig_path_geo}")


if __name__ == "__main__":
    main()
