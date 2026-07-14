"""Text-augmented final model: tabular features (D28, 35 features) + a
config.TEXT_PCA_COMPONENTS-dimension PCA reduction of sentence-transformer
verbatim embeddings (text_features.py).

Answers challenge brief section 4.4: "combine this text signal with your
tabular features... explain what the text adds or doesn't add on top of
the tabular baseline." See docs/decisions.md D31 for the full result and
-- essential context, not an afterthought -- the label-conditioning
caveat that must accompany any interpretation of the numbers this script
prints (the verbatims were generated FROM the label, with noise; see
text_features.py's module docstring).
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from evaluate import bootstrap_paired_diff_ci, evaluate
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix
from text_features import encode_verbatims, fit_pca, load_verbatims
from train import OrdinalLogisticClassifier


def main() -> None:
    labeled = pd.read_parquet(config.LABELED_DATASET_PATH)
    split = pd.read_parquet(config.REALISTIC_SPLIT_PATH)
    verbatims = load_verbatims()

    X_tabular = build_feature_matrix(labeled, include_geo=False)
    merged = (
        X_tabular
        .merge(labeled[[config.ID_COL, config.TARGET_COL]], on=config.ID_COL)
        .merge(split, on=config.ID_COL)
        .merge(verbatims, on=config.ID_COL)
    )
    assert len(merged) == len(labeled), (
        f"Row count changed after merging verbatims ({len(merged)} vs "
        f"{len(labeled)}) -- every customer should have exactly one verbatim."
    )

    embeddings = encode_verbatims(merged[[config.ID_COL, "verbatim"]])
    train_mask = (merged["split"] == "respondent_train").to_numpy()

    pca = fit_pca(embeddings[train_mask])
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA({config.TEXT_PCA_COMPONENTS}) fit on {train_mask.sum()} "
          f"respondent_train embeddings only -- explained variance: {explained:.1%}")

    reduced = pca.transform(embeddings)
    text_columns = [f"text_pca_{i}" for i in range(reduced.shape[1])]
    for i, col in enumerate(text_columns):
        merged[col] = reduced[:, i]

    numeric_features_text = NUMERIC_FEATURES + text_columns
    feature_columns = numeric_features_text + CATEGORICAL_FEATURES

    train_df = merged[merged["split"] == "respondent_train"]
    test_df = merged[merged["split"] == "silent_test"]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features_text),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", OrdinalLogisticClassifier(alpha=1.0, balanced=True)),
    ])

    X_train = train_df[feature_columns]
    y_train = train_df[config.TARGET_COL].astype(str)
    pipeline.fit(X_train, y_train)
    print(f"\nFit on {len(train_df)} rows (respondent_train) with "
          f"{len(feature_columns)} features ({len(NUMERIC_FEATURES)} tabular numeric + "
          f"{len(text_columns)} text PCA + {len(CATEGORICAL_FEATURES)} categorical).")

    y_test = test_df[config.TARGET_COL].astype(str)
    y_pred_text = pipeline.predict(test_df[feature_columns])
    y_proba_text = pipeline.predict_proba(test_df[feature_columns])

    results_text = evaluate(y_test, y_pred_text, y_proba_text)
    print(f"\n=== Tabular + text (silent_test, n={results_text['n_samples']}) ===")
    print(f"QWK={results_text['qwk']:.3f}  macro-F1={results_text['macro_f1']:.3f}  "
          f"balanced_acc={results_text['balanced_accuracy']:.3f}")
    print("recall per class:", {k: round(v, 3) for k, v in results_text["recall_per_class"].items()})

    # Tabular-only comparison point: the actually-shipped model (D28),
    # loaded fresh rather than refit here, so the comparison is against
    # exactly what is in production, not a freshly-refit twin that could
    # differ from it by ordinary fitting noise.
    baseline_artifact = joblib.load(config.FINAL_MODEL_PATH)
    baseline_pipeline = baseline_artifact["pipeline"]
    baseline_feature_columns = baseline_artifact["feature_columns"]
    y_pred_tabular = baseline_pipeline.predict(test_df[baseline_feature_columns])
    y_proba_tabular = baseline_pipeline.predict_proba(test_df[baseline_feature_columns])
    results_tabular = evaluate(y_test, y_pred_tabular, y_proba_tabular)
    print(f"\n=== Tabular-only baseline (D28 shipped model, same silent_test rows) ===")
    print(f"QWK={results_tabular['qwk']:.3f}  macro-F1={results_tabular['macro_f1']:.3f}  "
          f"balanced_acc={results_tabular['balanced_accuracy']:.3f}")

    qwk_fn = lambda yt, yp: cohen_kappa_score(yt, yp, labels=config.NPS_CATEGORIES, weights="quadratic")
    diff = bootstrap_paired_diff_ci(y_test.to_numpy(), y_pred_text, y_pred_tabular, qwk_fn, n_boot=2000)
    print(f"\nPaired bootstrap QWK diff (text+tabular MINUS tabular-only): "
          f"{diff['point']:+.3f}  95% CI [{diff['lower']:+.3f}, {diff['upper']:+.3f}]  "
          f"P(text+tabular better)={diff['p_a_better']:.3f}")
    print(
        "\nReminder before reading the numbers above as \"text helps\": the "
        "verbatims were generated conditioned on each customer's actual NPS "
        "category (~17.5% deliberately discordant). Any positive gap here is "
        "an upper bound on what genuinely independent customer text could "
        "plausibly contribute, not a measurement of it -- see D31."
    )

    artifact = {
        "pipeline": pipeline,
        "numeric_features": numeric_features_text,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": feature_columns,
        "text_pca": pca,
        "text_embedding_model": config.TEXT_EMBEDDING_MODEL,
        "nps_categories": config.NPS_CATEGORIES,
        "trained_on": "respondent_train (realistic split) + verbatim text PCA",
        "n_train_rows": len(train_df),
        "model_description": (
            "mord.LogisticAT (alpha=1.0, class-balanced), tabular features "
            f"(D28) + {config.TEXT_PCA_COMPONENTS}-component PCA of "
            f"{config.TEXT_EMBEDDING_MODEL} verbatim embeddings. See D31 for "
            "the comparison against the tabular-only model and the label-"
            "conditioning caveat."
        ),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.FINAL_MODEL_TEXT_PATH)
    print(f"\nSaved -> {config.FINAL_MODEL_TEXT_PATH}")


if __name__ == "__main__":
    main()
