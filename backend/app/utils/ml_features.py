from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_features_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime/bool/object columns into sklearn-safe dtypes."""
    work = X.copy()
    for col in work.columns:
        series = work[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            values = np.full(len(parsed), np.nan, dtype=np.float64)
            mask = parsed.notna()
            values[mask.to_numpy()] = parsed[mask].astype(np.int64).to_numpy() / 1e9
            work[col] = values
        elif pd.api.types.is_bool_dtype(series):
            work[col] = series.astype(int)
        elif not pd.api.types.is_numeric_dtype(series):
            work[col] = series.astype(str).replace("nan", np.nan)
    return work
