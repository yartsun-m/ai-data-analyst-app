from __future__ import annotations

from typing import Any

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema


def build_schema_from_profile(df: pd.DataFrame, column_types: dict[str, str]) -> DataFrameSchema:
    columns: dict[str, Column] = {}
    for col in df.columns:
        col_type = column_types.get(col, "unknown")
        if col_type == "numeric":
            columns[col] = Column(float, nullable=True, coerce=True)
        elif col_type == "boolean":
            columns[col] = Column(bool, nullable=True, coerce=True)
        else:
            columns[col] = Column(str, nullable=True, coerce=True)
    return DataFrameSchema(columns=columns, coerce=True)


def validate_dataframe(df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "passed": True,
        "issues": [],
        "checks_run": 0,
        "rows_validated": len(df),
    }

    if df.empty:
        report["passed"] = False
        report["issues"].append("Dataset is empty.")
        return report

    # Structural checks
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        report["passed"] = False
        report["issues"].append(f"Duplicate column names: {dupes}")

    null_rows = int(df.isna().all(axis=1).sum())
    if null_rows > 0:
        report["issues"].append(f"{null_rows} completely empty rows detected.")

    for col, col_type in column_types.items():
        if col not in df.columns:
            continue
        report["checks_run"] += 1
        series = df[col]
        missing_pct = float(series.isna().mean() * 100)
        if missing_pct > 95:
            report["issues"].append(f"Column '{col}' is {missing_pct:.1f}% missing.")
        if col_type == "numeric":
            coerced = pd.to_numeric(series, errors="coerce")
            invalid = int((series.notna() & coerced.isna()).sum())
            if invalid > 0:
                report["passed"] = False
                report["issues"].append(f"Column '{col}' has {invalid} non-numeric values.")
        if col_type == "categorical" and series.nunique(dropna=True) == len(series.dropna()):
            if len(series) > 20:
                report["issues"].append(
                    f"Column '{col}' has unique values for every row — may be an identifier."
                )

    # Pandera schema validation (lenient — sample for large data)
    sample = df.head(min(len(df), 5000))
    try:
        schema = build_schema_from_profile(sample, column_types)
        schema.validate(sample, lazy=True)
    except pa.errors.SchemaErrors as exc:
        report["passed"] = False
        for err in exc.failure_cases["check"].head(5).tolist():
            report["issues"].append(f"Schema check failed: {err}")
    except Exception as exc:
        report["issues"].append(f"Schema validation skipped: {exc}")

    if report["issues"] and report["passed"]:
        report["passed"] = len([i for i in report["issues"] if "failed" in i.lower()]) == 0

    return report
