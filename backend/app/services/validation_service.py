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
        "quality_report": {},
    }

    if df.empty:
        report["passed"] = False
        report["issues"].append("Dataset is empty.")
        return report

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        report["passed"] = False
        report["issues"].append(f"Duplicate column names: {dupes}")

    null_rows = int(df.isna().all(axis=1).sum())
    if null_rows > 0:
        report["issues"].append(f"{null_rows} completely empty rows detected.")

    column_quality: list[dict[str, Any]] = []
    for col, col_type in column_types.items():
        if col not in df.columns:
            continue
        report["checks_run"] += 1
        series = df[col]
        missing_pct = float(series.isna().mean() * 100)
        unique_ratio = float(series.nunique(dropna=True) / max(len(series.dropna()), 1))
        col_report = {
            "column": col,
            "type": col_type,
            "missing_pct": round(missing_pct, 2),
            "unique_values": int(series.nunique(dropna=True)),
            "unique_ratio": round(unique_ratio, 4),
            "expectations": [],
        }

        if missing_pct > 95:
            report["issues"].append(f"Column '{col}' is {missing_pct:.1f}% missing.")
            col_report["expectations"].append("expect_column_values_to_not_be_null (FAILED)")

        if col_type == "numeric":
            coerced = pd.to_numeric(series, errors="coerce")
            invalid = int((series.notna() & coerced.isna()).sum())
            if invalid > 0:
                report["passed"] = False
                report["issues"].append(f"Column '{col}' has {invalid} non-numeric values.")
                col_report["expectations"].append("expect_column_values_to_be_numeric (FAILED)")
            else:
                col_report["expectations"].append("expect_column_values_to_be_numeric (PASSED)")
            stats = coerced.describe()
            col_report["stats"] = {
                "mean": float(stats.get("mean", 0) or 0),
                "std": float(stats.get("std", 0) or 0),
                "min": float(stats.get("min", 0) or 0),
                "max": float(stats.get("max", 0) or 0),
            }

        if col_type == "categorical" and series.nunique(dropna=True) == len(series.dropna()):
            if len(series) > 20:
                report["issues"].append(
                    f"Column '{col}' has unique values for every row — may be an identifier."
                )
                col_report["expectations"].append("expect_column_unique_value_count_to_be_between (WARN)")

        column_quality.append(col_report)

    report["quality_report"] = {
        "framework": "Pandera + custom expectations",
        "columns": column_quality,
        "summary": {
            "total_columns": len(column_quality),
            "columns_with_high_missing": sum(1 for c in column_quality if c["missing_pct"] > 50),
            "identifier_like_columns": sum(1 for c in column_quality if c["unique_ratio"] >= 0.99),
        },
    }

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
