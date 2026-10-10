"""Ordinal Logistic Regression model wrapper for scikit-learn.

Wraps mord.LogisticAT to provide a full scikit-learn classifier contract:
- Operates on string labels ('Detractor', 'Passive', 'Promoter') or integer codes.
- Compatible with Pipeline, CalibratedClassifierCV, cross_val_score, and GridSearchCV.
- Exposes predict, predict_proba, and classes_.
- Serializes cleanly via joblib without __main__ namespace pollution.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

import mord

try:
    from src.config import NPS_CATEGORIES, NPS_CATEGORY_TO_CODE
except ImportError:
    from config import NPS_CATEGORIES, NPS_CATEGORY_TO_CODE

CODE_TO_CATEGORY = {v: k for k, v in NPS_CATEGORY_TO_CODE.items()}


class OrdinalLogisticClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal logistic regression using mord.LogisticAT with balanced weights.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter for mord.LogisticAT (L2 penalty).
    balanced : bool, default=True
        Whether to compute class-balanced sample weights automatically.
    """

    def __init__(self, alpha: float = 1.0, balanced: bool = True):
        self.alpha = alpha
        self.balanced = balanced

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y)
        # Handle string labels or integer codes
        if set(np.unique(y_arr).astype(str)) <= set(NPS_CATEGORIES):
            y_code = np.array([NPS_CATEGORY_TO_CODE[str(label)] for label in y_arr])
        else:
            y_code = y_arr.astype(int)

        if sample_weight is None and self.balanced:
            sample_weight = compute_sample_weight("balanced", y_code)

        self.model_ = mord.LogisticAT(alpha=self.alpha)
        self.model_.fit(X, y_code, sample_weight=sample_weight)
        self.classes_ = np.array(NPS_CATEGORIES)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        codes = self.model_.predict(X)
        return np.array([CODE_TO_CATEGORY[c] for c in codes])
