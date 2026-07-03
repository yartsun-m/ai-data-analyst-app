from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

_PHONE_RE = re.compile(r"^[\d\s.\-+\(\)xX]{7,}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


COLUMN_TYPES = ("numeric", "categorical", "datetime", "text", "boolean", "phone", "email", "unknown")


def _is_boolean_series(series: pd.Series) -> bool:
    if series.dtype == bool:
        return True
    unique = series.dropna().unique()
    if len(unique) == 0:
        return False
    normalized = {str(v).strip().lower() for v in unique}
    return normalized.issubset({"true", "false", "0", "1", "yes", "no", "y", "n"})


def _is_datetime_series(series: pd.Series, threshold: float = 0.8) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False, format="mixed")
    return parsed.notna().mean() >= threshold


def _looks_like_phone(series: pd.Series, threshold: float = 0.7) -> bool:
    sample = series.dropna().astype(str).str.strip().head(100)
    if sample.empty:
        return False
    return float(sample.apply(lambda v: bool(_PHONE_RE.match(v))).mean()) >= threshold


def _looks_like_email(series: pd.Series, threshold: float = 0.7) -> bool:
    sample = series.dropna().astype(str).str.strip().head(100)
    if sample.empty:
        return False
    return float(sample.apply(lambda v: bool(_EMAIL_RE.match(v))).mean()) >= threshold


def _is_numeric_series(series: pd.Series, threshold: float = 0.8) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    # Avoid treating phone-like strings as numeric (e.g. 846-790-4623)
    if _looks_like_phone(sample, threshold=0.5):
        return False
    converted = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    return converted.notna().mean() >= threshold


def _is_categorical_series(series: pd.Series, max_unique_ratio: float = 0.5) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    return unique_ratio <= max_unique_ratio and non_null.nunique() <= 100


def detect_column_type(series: pd.Series) -> str:
    if series.isna().all():
        return "unknown"
    if _is_boolean_series(series):
        return "boolean"
    if _looks_like_email(series):
        return "email"
    if _looks_like_phone(series):
        return "phone"
    if _is_datetime_series(series):
        return "datetime"
    if _is_numeric_series(series):
        return "numeric"
    if _is_categorical_series(series):
        return "categorical"
    return "text"


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    return {col: detect_column_type(df[col]) for col in df.columns}


def coerce_column(series: pd.Series, col_type: str) -> pd.Series:
    if col_type == "numeric":
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    if col_type == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=False)
    if col_type == "boolean":
        mapping = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
        return series.astype(str).str.strip().str.lower().map(mapping)
    return series


def summarize_column(series: pd.Series, col_type: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dtype": col_type,
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if col_type == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        summary.update(
            {
                "mean": _safe_float(numeric.mean()),
                "median": _safe_float(numeric.median()),
                "std": _safe_float(numeric.std()),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
            }
        )
    elif col_type in {"categorical", "boolean"}:
        top = series.value_counts(dropna=True).head(5)
        summary["top_values"] = {str(k): int(v) for k, v in top.items()}
    elif col_type == "datetime":
        parsed = pd.to_datetime(series, errors="coerce")
        summary.update(
            {
                "min_date": _safe_str(parsed.min()),
                "max_date": _safe_str(parsed.max()),
            }
        )
    else:
        sample = series.dropna().astype(str).head(3).tolist()
        summary["sample_values"] = sample

    return summary


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)


def _safe_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
