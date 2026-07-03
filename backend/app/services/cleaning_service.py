from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from app.utils.type_detection import coerce_column, detect_column_types

OutlierStrategy = Literal["none", "clip", "winsorize", "remove"]


def clean_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
    outlier_strategy: OutlierStrategy = "winsorize",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "steps": [],
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
        "outlier_strategy": outlier_strategy,
    }

    cleaned = df.copy()
    column_types = detect_column_types(cleaned)

    for col, col_type in column_types.items():
        if col == target_column:
            continue
        cleaned[col] = coerce_column(cleaned[col], col_type)

    report["steps"].append("Parsed column types (numeric, datetime, boolean)")

    duplicates_before = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    report["steps"].append(f"Removed {duplicates_before} duplicate rows")

    outlier_report = _treat_outliers(cleaned, column_types, target_column, outlier_strategy)
    if outlier_report["columns_treated"]:
        cleaned = outlier_report["dataframe"]
        report["outliers"] = outlier_report
        report["steps"].append(
            f"Outlier treatment ({outlier_strategy}): {len(outlier_report['columns_treated'])} numeric columns"
        )

    imputation: dict[str, str] = {}
    for col in cleaned.columns:
        if col == target_column:
            continue
        col_type = column_types.get(col, "unknown")
        missing = cleaned[col].isna().sum()
        if missing == 0:
            continue

        if col_type == "numeric":
            strategy = "median"
            value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(value)
        elif col_type in {"categorical", "boolean"}:
            strategy = "mode"
            mode = cleaned[col].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(value)
        elif col_type == "datetime":
            strategy = "forward_fill"
            cleaned[col] = cleaned[col].ffill().bfill()
        else:
            strategy = "empty_string"
            cleaned[col] = cleaned[col].fillna("")
        imputation[col] = strategy

    report["imputation"] = imputation
    report["steps"].append(f"Imputed missing values for {len(imputation)} columns")
    report["encoded_categorical_columns"] = []
    report["steps"].append("Preserved categorical columns for ML pipeline encoding")

    report["rows_after"] = int(len(cleaned))
    report["columns_after"] = int(len(cleaned.columns))
    report["missing_after"] = int(cleaned.isna().sum().sum())

    return cleaned, report


def _treat_outliers(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None,
    strategy: OutlierStrategy,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "strategy": strategy,
        "columns_treated": [],
        "rows_removed": 0,
        "dataframe": df,
    }
    if strategy == "none":
        return result

    work = df.copy()
    numeric_cols = [
        c
        for c, t in column_types.items()
        if t == "numeric" and c in work.columns and c != target_column
    ]
    if not numeric_cols:
        return result

    if strategy == "remove":
        mask = pd.Series(True, index=work.index)
        for col in numeric_cols:
            series = pd.to_numeric(work[col], errors="coerce")
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask &= series.between(lower, upper) | series.isna()
        before = len(work)
        work = work.loc[mask]
        result["rows_removed"] = before - len(work)
        result["columns_treated"] = numeric_cols
        result["dataframe"] = work
        return result

    for col in numeric_cols:
        series = pd.to_numeric(work[col], errors="coerce")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if strategy == "clip":
            work[col] = series.clip(lower, upper)
        elif strategy == "winsorize":
            p01, p99 = series.quantile(0.01), series.quantile(0.99)
            work[col] = series.clip(p01, p99)
        result["columns_treated"].append(col)

    result["dataframe"] = work
    return result
