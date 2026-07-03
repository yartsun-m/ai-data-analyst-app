from __future__ import annotations

import re
from typing import Any

import pandas as pd


def build_artifact_chunks(session_data: dict[str, Any]) -> list[dict[str, str]]:
    """Build searchable text chunks from session analysis artifacts."""
    chunks: list[dict[str, str]] = []

    profile = session_data.get("profile") or {}
    if profile:
        text = (
            f"Dataset has {profile.get('shape', {}).get('rows', '?')} rows and "
            f"{profile.get('shape', {}).get('columns', '?')} columns. "
            f"Duplicates: {profile.get('duplicate_rows', 0)}. "
            f"Task: {profile.get('task_type')}. Target: {profile.get('target_column')}."
        )
        chunks.append({"id": "profile", "text": text, "source": "profile"})

    validation = session_data.get("validation_report") or profile.get("validation_report")
    if validation:
        issues = "; ".join(validation.get("issues", [])[:8])
        chunks.append({
            "id": "validation",
            "text": f"Data validation passed={validation.get('passed')}. Issues: {issues}",
            "source": "validation",
        })

    ml = session_data.get("ml_results") or {}
    if ml:
        metrics = ml.get("best_metrics", {})
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items() if k != "primary_score")
        cv = ml.get("cross_validation", {})
        cv_str = f" CV mean={cv.get('mean')}" if cv.get("mean") is not None else ""
        warnings = "; ".join(ml.get("warnings", []))
        chunks.append({
            "id": "ml",
            "text": f"ML task={ml.get('task_type')} best={ml.get('best_model')} metrics={metric_str}{cv_str}. Warnings: {warnings}",
            "source": "ml",
        })
        explain = (ml.get("explainability") or {}).get("top_features") or []
        if explain:
            feats = ", ".join(f"{f['feature']}({f.get('importance')})" for f in explain[:8])
            chunks.append({"id": "importance", "text": f"Top features: {feats}", "source": "ml"})

    eda = session_data.get("eda") or {}
    if eda.get("insights"):
        chunks.append({
            "id": "eda",
            "text": "EDA insights: " + "; ".join(eda["insights"]),
            "source": "eda",
        })

    cleaning = session_data.get("cleaning_report") or {}
    if cleaning:
        chunks.append({
            "id": "cleaning",
            "text": f"Cleaning: {cleaning.get('rows_before')}->{cleaning.get('rows_after')} rows. Steps: {'; '.join(cleaning.get('steps', []))}",
            "source": "cleaning",
        })

    clustering = session_data.get("clustering") or {}
    if clustering:
        chunks.append({
            "id": "clustering",
            "text": f"Clustering: k={clustering.get('n_clusters')} silhouette={clustering.get('silhouette_score')}",
            "source": "clustering",
        })

    anomaly = session_data.get("anomaly") or {}
    if anomaly:
        chunks.append({
            "id": "anomaly",
            "text": f"Anomalies: {anomaly.get('anomaly_count')} of {anomaly.get('total_rows')} rows ({anomaly.get('anomaly_rate', 0):.1%})",
            "source": "anomaly",
        })

    return chunks


def retrieve_relevant_chunks(chunks: list[dict[str, str]], question: str, top_k: int = 4) -> list[dict[str, str]]:
    if not chunks:
        return []
    tokens = set(_tokenize(question))
    if not tokens:
        return chunks[:top_k]

    scored: list[tuple[float, dict[str, str]]] = []
    for chunk in chunks:
        chunk_tokens = set(_tokenize(chunk["text"]))
        overlap = len(tokens & chunk_tokens)
        bonus = 2.0 if chunk["source"] in _source_boost(question) else 0.0
        scored.append((overlap + bonus, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score > 0] or [c for _, c in scored[:top_k]]


def format_rag_context(chunks: list[dict[str, str]]) -> str:
    if not chunks:
        return ""
    lines = ["## Retrieved Context (RAG)"]
    for chunk in chunks:
        lines.append(f"- [{chunk['source']}] {chunk['text']}")
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2]


def _source_boost(question: str) -> set[str]:
    q = question.lower()
    boosts: set[str] = set()
    if any(w in q for w in ("model", "metric", "r2", "accuracy", "feature", "predict")):
        boosts.add("ml")
    if any(w in q for w in ("quality", "missing", "duplicate", "valid")):
        boosts.update({"validation", "cleaning"})
    if any(w in q for w in ("cluster", "segment", "group")):
        boosts.add("clustering")
    if any(w in q for w in ("anomal", "outlier", "unusual")):
        boosts.add("anomaly")
    if any(w in q for w in ("correlat", "trend", "chart", "eda")):
        boosts.add("eda")
    return boosts
