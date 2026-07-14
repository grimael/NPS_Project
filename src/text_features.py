"""Sentence-transformer embeddings + PCA for the synthetic verbatims
(challenge brief section 4.4, "combine this text signal with your
tabular features... explain what the text adds or doesn't add").

Encodes each customer's generated French verbatim (src/generate_verbatims.py)
into a dense vector (config.TEXT_EMBEDDING_MODEL, local, no API key), then
reduces to config.TEXT_PCA_COMPONENTS via PCA fit ONLY on respondent_train
(D8) to avoid leaking silent_test structure into the reduction -- the same
leakage-safe discipline as every other fitted transform in this project
(scalers, encoders, calibration D19).

Critical caveat for interpreting anything built on top of this module,
not just a docstring note: the verbatims are LLM-generated conditioned on
each customer's actual NPS category, with ~17.5% deliberately discordant
(generate_verbatims.py TONE_INSTRUCTIONS). The text is therefore a noisy
paraphrase of the label, not an independent signal the way a real
customer's own words would be. Any predictive gain from adding it should
be read as an upper bound on what genuinely independent verbatim text
could plausibly contribute -- not evidence that real customer writing is
this informative. See docs/decisions.md D31.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config


def load_verbatims() -> pd.DataFrame:
    """Customer ID -> verbatim text only. Deliberately excludes
    nps_category/is_discordant from generate_verbatims.py's output --
    those are generation metadata (the label itself, plus noise), not
    features; including them directly would be pure leakage, not the
    "encode the free text" task this module implements."""
    df = pd.read_csv(config.VERBATIMS_CSV_PATH)
    return df[[config.ID_COL, "verbatim"]]


def encode_verbatims(verbatims_df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
    """verbatims_df: has [config.ID_COL, "verbatim"]. Returns a (n, 384)
    embedding matrix in the SAME row order as verbatims_df.

    Cached to disk since encoding ~7000 texts is the slow step and this
    project's scripts are meant to be re-run often. The cache is keyed by
    customer ID, not row position or count: a caller merging verbatims.csv
    through a different chain of joins (e.g. train_text.py's tabular+
    target+split merge vs. this module's own __main__) will end up with a
    different row order than whatever order the cache was written in --
    validating on row count alone would silently return embeddings
    misaligned to the wrong customers. Re-ordering by ID on every read
    makes that class of bug structurally impossible rather than relying
    on every caller happening to preserve the same merge order.
    """
    ids_path = config.TEXT_EMBEDDINGS_CACHE_PATH.with_suffix(".ids.npy")
    if use_cache and config.TEXT_EMBEDDINGS_CACHE_PATH.exists() and ids_path.exists():
        cached_embeddings = np.load(config.TEXT_EMBEDDINGS_CACHE_PATH)
        cached_ids = np.load(ids_path, allow_pickle=True)
        if set(cached_ids) == set(verbatims_df[config.ID_COL]):
            lookup = dict(zip(cached_ids, cached_embeddings))
            print(f"Loaded cached embeddings -> {config.TEXT_EMBEDDINGS_CACHE_PATH} "
                  f"({len(lookup)} customers)")
            return np.stack([lookup[cid] for cid in verbatims_df[config.ID_COL]])
        print("Cache customer set does not match current verbatims.csv -- re-encoding.")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.TEXT_EMBEDDING_MODEL)
    embeddings = model.encode(
        verbatims_df["verbatim"].tolist(), show_progress_bar=True, batch_size=64,
        convert_to_numpy=True,
    )
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.TEXT_EMBEDDINGS_CACHE_PATH, embeddings)
    np.save(ids_path, verbatims_df[config.ID_COL].to_numpy())
    print(f"Saved embeddings -> {config.TEXT_EMBEDDINGS_CACHE_PATH} ({embeddings.shape})")
    return embeddings


def fit_pca(embeddings_train: np.ndarray, n_components: int = config.TEXT_PCA_COMPONENTS) -> PCA:
    pca = PCA(n_components=n_components, random_state=config.SEED)
    pca.fit(embeddings_train)
    return pca


def build_text_pca_frame(customer_ids: pd.Series, pca_embeddings: np.ndarray) -> pd.DataFrame:
    columns = [f"text_pca_{i}" for i in range(pca_embeddings.shape[1])]
    frame = pd.DataFrame(pca_embeddings, columns=columns)
    frame.insert(0, config.ID_COL, customer_ids.to_numpy())
    return frame


if __name__ == "__main__":
    verbatims = load_verbatims()
    print(f"Loaded {len(verbatims)} verbatims.")

    embeddings = encode_verbatims(verbatims)

    split = pd.read_parquet(config.REALISTIC_SPLIT_PATH)
    merged = verbatims.merge(split, on=config.ID_COL)
    train_mask = (merged["split"] == "respondent_train").to_numpy()

    pca = fit_pca(embeddings[train_mask])
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA({config.TEXT_PCA_COMPONENTS}) fit on {train_mask.sum()} respondent_train "
          f"embeddings -- explained variance: {explained:.1%}")

    reduced = pca.transform(embeddings)
    text_frame = build_text_pca_frame(verbatims[config.ID_COL], reduced)
    print(text_frame.head())
