from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.column_roles import (
    detect_column_roles,
    get_ml_excluded_columns,
    suggest_ml_targets,
)
from app.utils.type_detection import detect_column_types, summarize_column


def profile_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    column_types = detect_column_types(df)
    column_roles = detect_column_roles(df)
    columns_summary = {
        col: summarize_column(df[col], column_types[col]) for col in df.columns
    }

    duplicate_count = int(df.duplicated().sum())
    missing_by_column = {col: int(df[col].isna().sum()) for col in df.columns}

    suggested_targets = suggest_ml_targets(df, column_types, column_roles)
    ml_excluded_columns = get_ml_excluded_columns(df, column_roles, target_column=target_column)
    task_type = None
    if target_column and target_column in df.columns:
        task_type = detect_task_type(df[target_column], column_types.get(target_column, "unknown"))

    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": list(df.columns),
        "column_types": column_types,
        "column_roles": column_roles,
        "columns_summary": columns_summary,
        "missing_by_column": missing_by_column,
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_count,
        "suggested_targets": suggested_targets,
        "ml_excluded_columns": ml_excluded_columns,
        "target_column": target_column,
        "task_type": task_type,
    }


def detect_task_type(series: pd.Series, col_type: str) -> str:
    if col_type == "datetime":
        return "time_series"
    if col_type in {"numeric", "unknown"}:
        unique_count = series.nunique(dropna=True)
        if unique_count <= max(20, int(len(series) * 0.05)):
            return "classification"
        return "regression"
    if col_type in {"categorical", "boolean", "phone", "email", "text"}:
        return "classification"
    return "classification"
