"""Global configuration: paths, seed, column lists, label mapping.

Single source of truth so every script (make_dataset, build_target,
make_splits, features, train, evaluate) agrees on the same constants.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_VERBATIMS_DIR = DATA_DIR / "verbatims"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

ANALYTICAL_DATASET_PATH = DATA_PROCESSED_DIR / "analytical_dataset.parquet"
LABELED_DATASET_PATH = DATA_PROCESSED_DIR / "labeled_dataset.parquet"
REALISTIC_SPLIT_PATH = DATA_PROCESSED_DIR / "split_realistic.parquet"
RANDOM_SPLIT_PATH = DATA_PROCESSED_DIR / "split_random.parquet"

RAW_FILES = {
    "churn": DATA_RAW_DIR / "Telco_customer_churn.xlsx",
    "demographics": DATA_RAW_DIR / "Telco_customer_churn_demographics.xlsx",
    "location": DATA_RAW_DIR / "Telco_customer_churn_location.xlsx",
    "population": DATA_RAW_DIR / "Telco_customer_churn_population.xlsx",
    "services": DATA_RAW_DIR / "Telco_customer_churn_services.xlsx",
    "status": DATA_RAW_DIR / "Telco_customer_churn_status.xlsx",
}

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
ID_COL = "Customer ID"

# --------------------------------------------------------------------------
# Leakage control
# --------------------------------------------------------------------------
# These columns must NEVER enter the feature matrix. Satisfaction Score is
# the source of the label itself; the others are churn-outcome fields that
# would not be known at scoring time for a customer who hasn't churned yet.
LEAKAGE_BLACKLIST = [
    "Churn Score",
    "Churn Value",
    "Churn Label",
    "Churn Reason",
    "Churn Category",
    "Customer Status",
    "CLTV",
    "Satisfaction Score",
]

# --------------------------------------------------------------------------
# Target: Satisfaction Score (1-5) -> NPS category
# --------------------------------------------------------------------------
NPS_CATEGORIES = ["Detractor", "Passive", "Promoter"]  # ordinal, low to high
NPS_CATEGORY_TO_CODE = {cat: i for i, cat in enumerate(NPS_CATEGORIES)}


SATISFACTION_TO_NPS = {
    1: "Detractor",
    2: "Detractor",
    3: "Detractor",
    4: "Passive",
    5: "Promoter",
}

TARGET_COL = "NPS Category"
TARGET_COL_NOISY = "NPS Category Noisy"

# Ordinal noise injection for the sensitivity-test variant: fraction of rows
# whose label is reassigned to an *adjacent* ordinal class.
NOISY_LABEL_REASSIGNMENT_RATE = 0.10

# --------------------------------------------------------------------------
# Realistic ("silent majority") split
# --------------------------------------------------------------------------
# 15% voluntary survey respondents (reflecting typical telecom response rates:
# Anderson 1998, TM Forum 2021). The remaining 85% silent base is split 50/50 into
# an out-of-domain validation set (silent_val) for tuning/decisions and an
# untouched holdout test set (silent_test) to eliminate test-set snooping.
RESPONDENT_FRACTION = 0.15
SILENT_VAL_FRACTION = 0.50
SILENT_TEST_FRACTION = 0.50

RANDOM_SPLIT_TEST_SIZE = 0.30

# --------------------------------------------------------------------------
# Synthetic verbatims (phase 4)
# --------------------------------------------------------------------------
VERBATIMS_CSV_PATH = DATA_VERBATIMS_DIR / "verbatims.csv"
VERBATIMS_PROMPT_PATH = DATA_VERBATIMS_DIR / "prompt_used.txt"
VERBATIMS_PILOT_CSV_PATH = DATA_VERBATIMS_DIR / "verbatims_pilot.csv"


VERBATIM_PROVIDER_DEFAULT = "gemini"

# claude-api skill default: always Opus unless the user names another model.
VERBATIM_MODEL = "claude-opus-4-8"

GROQ_MODEL = "llama-3.1-8b-instant"

GEMINI_MODEL = "gemini-3.1-flash-lite"

GEMINI_BATCH_SIZE = 25

GEMINI_REQUEST_DELAY_SECONDS = 4.5

VERBATIM_TEMPERATURE = 0.9
VERBATIM_MAX_TOKENS = 200
VERBATIM_DISCORDANT_RATE = 0.175  # midpoint of the challenge's ~15-20% band
VERBATIM_PILOT_SIZE = 50

# --------------------------------------------------------------------------
# Text features (phase 4 step 2): sentence-transformer embeddings + PCA
# --------------------------------------------------------------------------

TEXT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TEXT_EMBEDDINGS_CACHE_PATH = DATA_VERBATIMS_DIR / "verbatim_embeddings.npy"
TEXT_PCA_COMPONENTS = 20

# --------------------------------------------------------------------------
# Model artifacts (phase 3+)
# --------------------------------------------------------------------------
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_MODEL_TEXT_PATH = MODELS_DIR / "final_model_text.joblib"
