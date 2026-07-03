from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.utils.column_roles import detect_column_roles, get_ml_excluded_columns
from app.utils.ml_features import sanitize_features_for_ml


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict[str, Any]:
    column_roles = detect_column_roles(df)
    excluded = get_ml_excluded_columns(df, column_roles)
    feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No usable feature columns for anomaly detection.")

    work = sanitize_features_for_ml(df[feature_cols].copy())
    for col in work.columns:
        if work[col].dtype == object or str(work[col].dtype) == "string":
            work[col] = pd.Categorical(work[col].astype(str)).codes
    work = work.fillna(work.median(numeric_only=True)).fillna(0)

    if len(work) < 20:
        raise ValueError("Need at least 20 rows for anomaly detection.")

    contamination = min(max(contamination, 0.01), 0.25)
    scaler = StandardScaler()
    X = scaler.fit_transform(work)

    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)

    anomaly_mask = preds == -1
    anomaly_indices = np.where(anomaly_mask)[0].tolist()[:100]

    return {
        "method": "isolation_forest",
        "contamination": contamination,
        "total_rows": int(len(work)),
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_rate": float(anomaly_mask.mean()),
        "anomaly_indices": anomaly_indices,
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        },
        "feature_columns": feature_cols,
    }
