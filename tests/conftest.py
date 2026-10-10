"""Pytest fixtures for NPS Prediction tests."""
import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import pytest

from src.config import LABELED_DATASET_PATH, REALISTIC_SPLIT_PATH


@pytest.fixture(scope="session")
def labeled_data():
    """Load the full labeled dataset."""
    assert LABELED_DATASET_PATH.exists(), f"Missing {LABELED_DATASET_PATH}"
    return pd.read_parquet(LABELED_DATASET_PATH)


@pytest.fixture(scope="session")
def realistic_splits():
    """Load realistic 3-way split."""
    assert REALISTIC_SPLIT_PATH.exists(), f"Missing {REALISTIC_SPLIT_PATH}"
    return pd.read_parquet(REALISTIC_SPLIT_PATH)
