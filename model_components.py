"""Small joblib-picklable sklearn transformers shared by training and inference."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClipTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, lower=-6.0, upper=6.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X), self.lower, self.upper).astype(np.float32)
