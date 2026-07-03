from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.storage import DatasetSession, session_store


def execute_tools(session: DatasetSession, question: str) -> list[dict[str, Any]]:
    """Run lightweight analysis tools based on question intent."""
    results: list[dict[str, Any]] = []
    q = question.lower()
    df = session_store.get_active_df(session)

    if any(w in q for w in ("correlat", "relationship", "association")):
        results.append(_tool_correlations(df))

    if any(w in q for w in ("missing", "null", "quality", "duplicate")):
        results.append(_tool_data_quality(df, session))

    if any(w in q for w in ("feature", "important", "predict", "model", "metric", "r2", "accuracy")):
        results.append(_tool_ml_summary(session))

    if any(w in q for w in ("top", "category", "distribution", "count", "value")):
        col = _guess_column(q, df.columns.tolist())
        if col:
            results.append(_tool_column_summary(df, col))

    if not results:
        results.append(_tool_dataset_overview(df, session))

    return results


def format_tool_results(results: list[dict[str, Any]]) -> str:
    lines = ["## Tool Results"]
    for item in results:
        lines.append(f"### {item['tool']}")
        lines.append(item["output"])
    return "\n".join(lines)


def _tool_correlations(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"tool": "correlations", "output": "Not enough numeric columns for correlation analysis."}
    corr = numeric.corr(numeric_only=True)
    pairs: list[str] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= 0.5:
                pairs.append(f"{a} vs {b}: r={val:.3f}")
    output = "; ".join(pairs[:10]) if pairs else "No strong correlations (|r| >= 0.5) found."
    return {"tool": "correlations", "output": output}


def _tool_data_quality(df: pd.DataFrame, session: DatasetSession) -> dict[str, Any]:
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    validation = session.validation_report or (session.profile or {}).get("validation_report") or {}
    issues = validation.get("issues", [])
    output = f"Missing cells: {missing}. Duplicate rows: {dupes}. Validation issues: {'; '.join(issues[:5]) or 'none'}."
    return {"tool": "data_quality", "output": output}


def _tool_ml_summary(session: DatasetSession) -> dict[str, Any]:
    ml = session.ml_results
    if not ml:
        return {"tool": "ml_summary", "output": "No ML results yet. Train a model first."}
    metrics = ", ".join(f"{k}={v}" for k, v in (ml.get("best_metrics") or {}).items() if k != "primary_score")
    feats = (ml.get("explainability") or {}).get("top_features") or ml.get("feature_importance") or []
    feat_str = ", ".join(f"{f['feature']}" for f in feats[:5])
    warnings = "; ".join(ml.get("warnings", []))
    output = f"Best model: {ml.get('best_model')}. Metrics: {metrics}. Top features: {feat_str}. Warnings: {warnings or 'none'}."
    return {"tool": "ml_summary", "output": output}


def _tool_column_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        output = f"{col}: mean={desc.get('mean', 0):.2f}, std={desc.get('std', 0):.2f}, min={desc.get('min')}, max={desc.get('max')}"
    else:
        top = series.astype(str).value_counts().head(5)
        output = f"{col} top values: {dict(top)}"
    return {"tool": "column_summary", "output": output}


def _tool_dataset_overview(df: pd.DataFrame, session: DatasetSession) -> dict[str, Any]:
    profile = session.profile or {}
    output = (
        f"Rows: {len(df)}, columns: {len(df.columns)}, "
        f"task: {profile.get('task_type')}, target: {profile.get('target_column')}"
    )
    return {"tool": "dataset_overview", "output": output}


def _guess_column(question: str, columns: list[str]) -> str | None:
    q = question.lower()
    for col in columns:
        if col.lower() in q:
            return col
    return None
