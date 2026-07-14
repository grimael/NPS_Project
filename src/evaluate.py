
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from config import NPS_CATEGORIES, REPORTS_FIGURES_DIR


def compute_cumulative_lift(y_true_binary: np.ndarray, y_score: np.ndarray,
                             n_bins: int = 10) -> np.ndarray:
    """Cumulative lift at each decile of predicted score, sorted descending.

    lift[k] = (positive rate among the top (k+1)/n_bins scored customers)
              / (overall positive rate)
    A model with no signal has lift ~= 1 everywhere; a useful model for
    prioritizing outreach has lift > 1 in the top deciles.
    """
    y_true_binary = np.asarray(y_true_binary)
    order = np.argsort(-np.asarray(y_score))
    y_sorted = y_true_binary[order]
    n = len(y_sorted)
    baseline_rate = y_sorted.mean()
    edges = np.linspace(0, n, n_bins + 1)[1:].astype(int)
    lifts = np.array([
        y_sorted[:edge].mean() / baseline_rate if baseline_rate > 0 else np.nan
        for edge in edges
    ])
    return lifts


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_lift_curve(lifts: np.ndarray, positive_class: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    deciles = np.arange(1, len(lifts) + 1) * (100 // len(lifts))
    ax.plot(deciles, lifts, marker="o", color="#C44E52", label=f"{positive_class} lift")
    ax.axhline(1.0, color="gray", linestyle="--", label="Baseline (no lift)")
    ax.set_xlabel("Top % of customers by predicted score")
    ax.set_ylabel("Cumulative lift")
    ax.set_title(f"Cumulative lift curve — {positive_class}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibration_curves(y_true: np.ndarray, y_proba: np.ndarray,
                              labels: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.5),
                              sharey=True)
    for ax, cls in zip(axes, labels):
        y_bin = (y_true == cls).astype(int)
        col = labels.index(cls)
        n_bins = 10 if y_bin.sum() >= 20 else 5
        frac_pos, mean_pred = calibration_curve(
            y_bin, y_proba[:, col], n_bins=n_bins, strategy="quantile",
        )
        ax.plot(mean_pred, frac_pos, marker="o", color="#4C72B0")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title(cls)
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed frequency")
    plt.suptitle("Calibration (reliability) curves — one-vs-rest")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def expected_calibration_error(
    y_true_binary,
    y_proba_class,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """One-vs-rest Expected Calibration Error: ECE = sum_b (n_b/N) *
    |accuracy_b - confidence_b|, the weighted-average gap between mean
    predicted probability and observed frequency across bins. Uses the
    same quantile-binning convention as `evaluate()`'s calibration
    curves, computed directly (not derived from sklearn's
    `calibration_curve`) so bin membership and weights are exact.
    """
    y_true_binary = np.asarray(y_true_binary, dtype=float)
    y_proba_class = np.asarray(y_proba_class, dtype=float)
    n = len(y_true_binary)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_proba_class, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    bin_ids = np.digitize(y_proba_class, edges[1:-1], right=True)

    ece = 0.0
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        n_b = mask.sum()
        confidence_b = y_proba_class[mask].mean()
        accuracy_b = y_true_binary[mask].mean()
        ece += (n_b / n) * abs(accuracy_b - confidence_b)
    return float(ece)


def evaluate(
    y_true,
    y_pred,
    y_proba: np.ndarray | None = None,
    *,
    labels: list[str] | None = None,
    positive_class: str = "Detractor",
    save_dir: Path | None = None,
    run_name: str | None = None,
) -> dict:
    """

    Parameters
    ----------
    y_true, y_pred : array-like of category strings (e.g. "Detractor").
    y_proba : optional array (n_samples, n_classes), columns in the exact
        order of `labels`. Required for the lift curve and calibration
        curves; without it, only the label-based metrics are computed.
    labels : ordinal class order, defaults to config.NPS_CATEGORIES.
        Passed explicitly to every sklearn call so quadratic weights (QWK)
        and confusion-matrix axes follow the true ordinal order rather
        than alphabetical sorting.
    save_dir : if given, saves confusion matrix / lift / calibration
        figures as PNGs there (defaults to reports/figures if `run_name`
        is given but `save_dir` is not).
    run_name : filename prefix for saved figures, e.g. "logreg_random_split".

    Returns
    -------
    dict with macro_f1, balanced_accuracy, qwk, recall_per_class,
    confusion_matrix, and (if y_proba given) lift_detractor and
    figure paths.
    """
    labels = labels or NPS_CATEGORIES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = dict(zip(labels, recalls))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "n_samples": len(y_true),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "qwk": qwk,
        "recall_per_class": recall_per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }

    if run_name and save_dir is None:
        save_dir = REPORTS_FIGURES_DIR
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{run_name}_" if run_name else ""
        cm_path = save_dir / f"{prefix}confusion_matrix.png"
        _plot_confusion_matrix(cm, labels, cm_path)
        results["confusion_matrix_path"] = cm_path

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.shape[1] != len(labels):
            raise ValueError(
                f"y_proba has {y_proba.shape[1]} columns, expected {len(labels)} "
                f"(one per label, in the order of `labels`)."
            )
        pos_idx = labels.index(positive_class)
        y_true_binary = (y_true == positive_class).astype(int)
        lifts = compute_cumulative_lift(y_true_binary, y_proba[:, pos_idx])
        results["lift_positive_class"] = positive_class
        results["lift_by_decile"] = lifts

        results["ece_per_class"] = {
            cls: expected_calibration_error(
                (y_true == cls).astype(int), y_proba[:, labels.index(cls)],
            )
            for cls in labels
        }

        if save_dir is not None:
            lift_path = save_dir / f"{prefix}lift_curve.png"
            _plot_lift_curve(lifts, positive_class, lift_path)
            results["lift_curve_path"] = lift_path

            calib_path = save_dir / f"{prefix}calibration_curves.png"
            _plot_calibration_curves(y_true, y_proba, labels, calib_path)
            results["calibration_curve_path"] = calib_path

    return results


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap CI for a single model's metric on a fixed test set.

    Resamples test-set rows (with replacement) and recomputes `metric_fn`
    each time. This quantifies test-set sampling noise only — it does NOT
    capture variance from which 15% of customers happened to end up in
    the training sample (the realistic split's `Customer ID`s are fixed
    by `config.SEED`, per the reproducibility rule). A gap that survives
    this bootstrap is real given *this* trained model; it says nothing
    about how much the model itself would change under a different
    respondent draw. See docs/decisions.md D16.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        values[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
    return {"point": metric_fn(y_true, y_pred), "lower": lower, "upper": upper,
            "values": values}


def bootstrap_paired_diff_ci(
    y_true,
    y_pred_a,
    y_pred_b,
    metric_fn,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI for metric(model_a) - metric(model_b).

    Both models are evaluated on the *same* resampled indices at each
    iteration (paired), which is more powerful than comparing two
    independent bootstrap CIs when both models are scored on the same
    test set. `P(diff > 0)` is the fraction of bootstrap iterations
    where model_a beats model_b — a Bayesian-flavored complement to the
    percentile CI, not a p-value.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    point = metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b)
    return {"point": point, "lower": lower, "upper": upper,
            "p_a_better": float(np.mean(diffs > 0)), "values": diffs}


def print_evaluation(results: dict, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"n = {results['n_samples']}")
    print(f"macro-F1            : {results['macro_f1']:.3f}")
    print(f"balanced accuracy   : {results['balanced_accuracy']:.3f}")
    print(f"QWK                 : {results['qwk']:.3f}")
    print("recall per class    :", {k: round(v, 3) for k, v in
                                     results["recall_per_class"].items()})
    if "lift_by_decile" in results:
        top_decile_lift = results["lift_by_decile"][0]
        print(f"lift @ top decile ({results['lift_positive_class']}): "
              f"{top_decile_lift:.2f}x")
    if "ece_per_class" in results:
        print("ECE per class       :", {k: round(v, 3) for k, v in
                                         results["ece_per_class"].items()})


if __name__ == "__main__":
    # Quick sanity test: majority-class baseline (always predict Detractor)
    # on the real labeled dataset. Not a substitute for unit tests on edge
    # cases, but catches any regression in the metric computations using
    # real, deterministic data rather than synthetic fixtures.
    import pandas as pd

    from config import LABELED_DATASET_PATH, TARGET_COL

    df = pd.read_parquet(LABELED_DATASET_PATH)
    y_true = df[TARGET_COL].astype(str).to_numpy()

    y_pred_majority = np.full_like(y_true, fill_value="Detractor")
    rng = np.random.RandomState(42)
    # A deliberately uninformative y_proba (uniform noise) so the lift
    # curve / calibration code paths are exercised too.
    y_proba_random = rng.dirichlet(alpha=[1, 1, 1], size=len(y_true))

    results = evaluate(
        y_true, y_pred_majority, y_proba_random,
        save_dir=REPORTS_FIGURES_DIR, run_name="_smoke_test_majority_baseline",
    )
    print_evaluation(results, "Smoke test: majority-class baseline")

    assert results["recall_per_class"]["Detractor"] == 1.0, "Majority class recall must be 1.0"
    assert results["recall_per_class"]["Passive"] == 0.0
    assert results["recall_per_class"]["Promoter"] == 0.0
    assert -1.0 <= results["qwk"] <= 1.0
    assert 0.0 <= results["macro_f1"] <= 1.0
    assert 0.0 <= results["balanced_accuracy"] <= 1.0
    assert results["confusion_matrix"].sum() == len(y_true)
    print("\nOK: evaluate() smoke test passed.")

    # Bootstrap smoke test: a perfect model vs. the majority baseline on the
    # same (real) labels should show a CI for the perfect model pinned at
    # (1, 1, 1) and a paired diff whose CI excludes 0 in favor of the
    # perfect model.
    qwk_fn = lambda yt, yp: cohen_kappa_score(yt, yp, labels=NPS_CATEGORIES, weights="quadratic")
    perfect_ci = bootstrap_metric_ci(y_true, y_true, qwk_fn, n_boot=200)
    assert perfect_ci["point"] == perfect_ci["lower"] == perfect_ci["upper"] == 1.0

    diff = bootstrap_paired_diff_ci(y_true, y_true, y_pred_majority, qwk_fn, n_boot=200)
    assert diff["lower"] > 0, "Perfect model must beat majority baseline with CI excluding 0"
    assert diff["p_a_better"] == 1.0
    print("OK: bootstrap_metric_ci / bootstrap_paired_diff_ci smoke test passed.")

    # ECE smoke test: perfectly calibrated probabilities (P(class)==true
    # frequency in every bin, by construction) should score ~0; a
    # deliberately overconfident-and-wrong model should score high.
    rng = np.random.RandomState(42)
    y_true_binary_perfect = rng.binomial(1, 0.3, size=5000)
    proba_perfect = np.full(5000, 0.3) + rng.normal(0, 1e-6, size=5000)
    ece_perfect = expected_calibration_error(y_true_binary_perfect, proba_perfect)
    assert ece_perfect < 0.02, f"Expected near-zero ECE, got {ece_perfect}"

    y_true_binary_bad = np.zeros(1000, dtype=int)
    proba_overconfident = np.full(1000, 0.95)
    ece_bad = expected_calibration_error(y_true_binary_bad, proba_overconfident)
    assert ece_bad > 0.9, f"Expected high ECE for overconfident-wrong predictions, got {ece_bad}"
    print(f"OK: expected_calibration_error smoke test passed "
          f"(perfect={ece_perfect:.4f}, overconfident-wrong={ece_bad:.4f}).")
