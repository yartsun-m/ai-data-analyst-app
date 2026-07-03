from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.utils.type_detection import coerce_column, detect_column_types


def clean_dataframe(df: pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "steps": [],
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
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

    # Keep categorical columns intact for target selection and ML encoding.
    # One-hot encoding is handled by the ML preprocessing pipeline.
    report["encoded_categorical_columns"] = []
    report["steps"].append("Preserved categorical columns for ML pipeline encoding")

    report["rows_after"] = int(len(cleaned))
    report["columns_after"] = int(len(cleaned.columns))
    report["missing_after"] = int(cleaned.isna().sum().sum())

    return cleaned, report
