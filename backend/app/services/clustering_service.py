from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.utils.column_roles import detect_column_roles, get_ml_excluded_columns
from app.utils.ml_features import sanitize_features_for_ml


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int | None = None,
    max_clusters: int = 8,
) -> dict[str, Any]:
    column_roles = detect_column_roles(df)
    excluded = get_ml_excluded_columns(df, column_roles)
    feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No usable feature columns for clustering.")

    work = sanitize_features_for_ml(df[feature_cols].copy())
    for col in work.columns:
        if work[col].dtype == object or str(work[col].dtype) == "string":
            work[col] = pd.Categorical(work[col].astype(str)).codes
    work = work.fillna(work.median(numeric_only=True)).fillna(0)

    if len(work) < 10:
        raise ValueError("Need at least 10 rows for clustering.")

    scaler = StandardScaler()
    X = scaler.fit_transform(work)

    best_k = n_clusters
    if best_k is None:
        best_k, scores = _auto_select_k(X, max_clusters=max_clusters)
    else:
        scores = {}

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = None
    if best_k > 1 and len(set(labels)) > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    scatter = [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": int(labels[i])}
        for i in range(min(len(coords), 2000))
    ]

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "n_clusters": int(best_k),
        "silhouette_score": sil,
        "k_scores": scores,
        "feature_columns": feature_cols,
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "scatter": scatter,
        "method": "kmeans",
    }


def _auto_select_k(X: np.ndarray, max_clusters: int) -> tuple[int, dict[str, float]]:
    max_k = min(max_clusters, max(2, len(X) // 20))
    scores: dict[str, float] = {}
    best_k = 2
    best_score = float("-inf")
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        try:
            score = float(silhouette_score(X, labels))
            scores[str(k)] = score
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k, scores
