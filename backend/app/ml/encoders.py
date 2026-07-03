from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns by the frequency of each category.

    Works for arbitrarily high-cardinality columns without exploding the feature
    space (unlike one-hot). Unseen categories at transform time map to 0.
    """

    def __init__(self) -> None:
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        self.frequency_maps_ = {}
        for col in self.columns_:
            counts = X[col].astype(str).value_counts(normalize=True)
            self.frequency_maps_[col] = counts.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._as_frame(X)
        output = np.zeros((len(X), len(self.columns_)), dtype=np.float64)
        for idx, col in enumerate(self.columns_):
            mapping = self.frequency_maps_.get(col, {})
            output[:, idx] = X[col].astype(str).map(mapping).fillna(0.0).to_numpy()
        return output

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.columns_
        return np.asarray([f"{name}_freq" for name in names], dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
