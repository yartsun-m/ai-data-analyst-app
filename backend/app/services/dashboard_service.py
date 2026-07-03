from __future__ import annotations

from typing import Any

import pandas as pd


def build_dashboard(
    df: pd.DataFrame,
    profile: dict[str, Any] | None,
    cleaning_report: dict[str, Any] | None,
    eda: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
) -> dict[str, Any]:
    kpis = _build_kpis(df, profile, cleaning_report, ml_results)

    return {
        "title": "AI Data Analyst Dashboard",
        "kpis": kpis,
        "profile_summary": _profile_summary(profile),
        "cleaning_summary": _cleaning_summary(cleaning_report),
        "charts": (eda or {}).get("charts", [])[:12],
        "ml_summary": _ml_summary(ml_results),
        "report_sections": _report_sections(kpis, profile, cleaning_report, eda, ml_results),
    }


def _build_kpis(
    df: pd.DataFrame,
    profile: dict[str, Any] | None,
    cleaning_report: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    kpis = [
        {"label": "Rows", "value": len(df)},
        {"label": "Columns", "value": len(df.columns)},
        {"label": "Missing Cells", "value": int(df.isna().sum().sum())},
        {"label": "Duplicate Rows", "value": int(df.duplicated().sum())},
    ]

    if cleaning_report:
        kpis.append(
            {
                "label": "Rows After Cleaning",
                "value": cleaning_report.get("rows_after", len(df)),
            }
        )

    if ml_results and ml_results.get("best_model"):
        kpis.append({"label": "Best Model", "value": ml_results["best_model"]})
        metrics = ml_results.get("best_metrics", {})
        for key in ("r2", "accuracy", "f1_weighted", "rmse"):
            if key in metrics:
                kpis.append({"label": key.upper(), "value": round(metrics[key], 4)})

    if profile and profile.get("task_type"):
        kpis.insert(0, {"label": "Task Type", "value": profile["task_type"]})

    return kpis


def _profile_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    return {
        "shape": profile.get("shape"),
        "column_types": profile.get("column_types"),
        "suggested_targets": profile.get("suggested_targets"),
        "target_column": profile.get("target_column"),
    }


def _cleaning_summary(cleaning_report: dict[str, Any] | None) -> dict[str, Any]:
    if not cleaning_report:
        return {}
    return {
        "steps": cleaning_report.get("steps", []),
        "imputation": cleaning_report.get("imputation", {}),
        "encoded_categorical_columns": cleaning_report.get("encoded_categorical_columns", []),
    }


def _ml_summary(ml_results: dict[str, Any] | None) -> dict[str, Any]:
    if not ml_results:
        return {}
    return {
        "task_type": ml_results.get("task_type"),
        "best_model": ml_results.get("best_model"),
        "best_metrics": ml_results.get("best_metrics"),
        "feature_importance": ml_results.get("feature_importance", []),
        "leaderboard": ml_results.get("leaderboard", []),
    }


def _report_sections(
    kpis: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    cleaning_report: dict[str, Any] | None,
    eda: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
) -> list[dict[str, str]]:
    sections = [
        {
            "title": "Executive Summary",
            "content": _executive_summary(kpis, profile, ml_results),
        }
    ]

    if cleaning_report:
        sections.append(
            {
                "title": "Data Quality",
                "content": "Automatic cleaning removed duplicates, imputed missing values, and encoded categorical fields.",
            }
        )

    if eda and eda.get("insights"):
        sections.append({"title": "Key Trends", "content": "; ".join(eda["insights"])})

    if ml_results and ml_results.get("best_model"):
        metrics = ml_results.get("best_metrics", {})
        sections.append(
            {
                "title": "Model Performance",
                "content": f"Best model: {ml_results['best_model']}. Metrics: {metrics}",
            }
        )

    return sections


def _executive_summary(
    kpis: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
) -> str:
    rows = next((k["value"] for k in kpis if k["label"] == "Rows"), "?")
    cols = next((k["value"] for k in kpis if k["label"] == "Columns"), "?")
    task = profile.get("task_type") if profile else None
    parts = [f"Dataset contains {rows} rows and {cols} columns."]
    if task:
        parts.append(f"Detected task type: {task}.")
    if ml_results and ml_results.get("best_model"):
        parts.append(f"Top model: {ml_results['best_model']}.")
    return " ".join(parts)
