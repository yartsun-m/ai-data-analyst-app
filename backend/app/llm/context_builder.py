from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.type_detection import detect_column_types


SYSTEM_PROMPT = """You are an expert data analyst assistant.
You receive summarized structured statistics about a dataset — never raw rows.
Answer clearly, cite numbers from the context, and suggest actionable next steps.
If information is missing, say so explicitly. Do not invent data."""


def build_llm_context(
    df: pd.DataFrame,
    profile: dict[str, Any] | None,
    cleaning_report: dict[str, Any] | None,
    eda: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
    explainability: dict[str, Any] | None,
    question: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    column_types = profile.get("column_types") if profile else detect_column_types(df)

    sections = [
        f"User question: {question}",
        _section_chat_history(chat_history),
        _section_dataset_overview(df, profile),
        _section_column_stats(df, column_types),
        _section_cleaning(cleaning_report),
        _section_eda(eda),
        _section_ml(ml_results),
        _section_explainability(explainability),
    ]
    return "\n\n".join(section for section in sections if section)


def _section_chat_history(chat_history: list[dict[str, str]] | None) -> str:
    if not chat_history:
        return ""
    lines = ["## Recent Conversation"]
    for msg in chat_history[-6:]:
        role = msg.get("role", "user")
        content = (msg.get("content") or "")[:500]
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def _section_dataset_overview(df: pd.DataFrame, profile: dict[str, Any] | None) -> str:
    lines = [
        "## Dataset Overview",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        f"- Column names: {', '.join(map(str, df.columns[:30]))}",
    ]
    if profile:
        lines.append(f"- Duplicate rows: {profile.get('duplicate_rows', 'unknown')}")
        lines.append(f"- Total missing cells: {profile.get('total_missing_cells', 'unknown')}")
        if profile.get("task_type"):
            lines.append(f"- Detected ML task: {profile['task_type']}")
        if profile.get("target_column"):
            lines.append(f"- Target column: {profile['target_column']}")
    return "\n".join(lines)


def _section_column_stats(df: pd.DataFrame, column_types: dict[str, str]) -> str:
    lines = ["## Column Statistics (summarized)"]
    for col in df.columns[:25]:
        col_type = column_types.get(col, "unknown")
        series = df[col]
        missing = int(series.isna().sum())
        line = f"- {col} ({col_type}): missing={missing}, unique={series.nunique(dropna=True)}"
        if col_type == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            line += f", mean={numeric.mean():.4f}, std={numeric.std():.4f}, min={numeric.min():.4f}, max={numeric.max():.4f}"
        elif col_type in {"categorical", "boolean"}:
            top = series.astype(str).value_counts().head(3)
            line += f", top_values={dict(top)}"
        lines.append(line)
    return "\n".join(lines)


def _section_cleaning(cleaning_report: dict[str, Any] | None) -> str:
    if not cleaning_report:
        return ""
    return (
        "## Cleaning Report\n"
        f"- Rows before/after: {cleaning_report.get('rows_before')} → {cleaning_report.get('rows_after')}\n"
        f"- Columns before/after: {cleaning_report.get('columns_before')} → {cleaning_report.get('columns_after')}\n"
        f"- Steps: {'; '.join(cleaning_report.get('steps', []))}"
    )


def _section_eda(eda: dict[str, Any] | None) -> str:
    if not eda:
        return ""
    insights = eda.get("insights") or []
    insight_text = "; ".join(insights) if insights else "No automated insights generated."
    return (
        "## EDA Summary\n"
        f"- Charts generated: {eda.get('chart_count', 0)}\n"
        f"- Numeric columns: {', '.join(eda.get('numeric_columns', []))}\n"
        f"- Insights: {insight_text}"
    )


def _section_ml(ml_results: dict[str, Any] | None) -> str:
    if not ml_results:
        return ""
    best = ml_results.get("best_model")
    metrics = ml_results.get("best_metrics", {})
    metric_text = ", ".join(f"{k}={v}" for k, v in metrics.items() if k != "primary_score")
    leaderboard = ml_results.get("leaderboard", [])
    models = ", ".join(entry.get("model", "?") for entry in leaderboard if "model" in entry)
    cv = ml_results.get("cross_validation", {})
    cv_text = ""
    if cv.get("mean") is not None:
        cv_text = f"\n- Cross-validation ({cv.get('scoring')}): mean={cv['mean']:.4f}, std={cv.get('std', 0):.4f}"
    warnings = ml_results.get("warnings") or []
    warn_text = f"\n- Warnings: {'; '.join(warnings)}" if warnings else ""
    return (
        "## ML Results\n"
        f"- Task type: {ml_results.get('task_type')}\n"
        f"- Best model: {best}\n"
        f"- Best metrics: {metric_text}{cv_text}{warn_text}\n"
        f"- Models evaluated: {models}"
    )


def _section_explainability(explainability: dict[str, Any] | None) -> str:
    if not explainability:
        return ""
    top = explainability.get("top_features") or explainability.get("shap_summary") or []
    if not top:
        return ""
    features = ", ".join(
        f"{item.get('feature')} ({item.get('importance') or item.get('mean_abs_shap')})" for item in top[:10]
    )
    return f"## Feature Importance\n- Top features: {features}"
