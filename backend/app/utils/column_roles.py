from __future__ import annotations

import re
from typing import Any

import pandas as pd

COLUMN_ROLES = (
    "identifier",
    "phone",
    "email",
    "url",
    "person_name",
    "feature",
)

_PHONE_RE = re.compile(r"^[\d\s.\-+\(\)xX]{7,}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_NAME_HINTS = ("first name", "last name", "firstname", "lastname", "surname", "given name", "family name")
_ID_HINTS = ("uuid", "guid", "ean", "internal id", "sku", "customer id", "user id", "order id", "account id", "record id")
_PHONE_HINTS = ("phone", "tel", "mobile", "fax", "cell")
_EMAIL_HINTS = ("email", "e-mail", "e_mail")
_URL_HINTS = ("website", "url", "link", "homepage")


def dataset_thresholds(row_count: int) -> dict[str, int | float]:
    """Size-aware thresholds used across profiling and ML.

    Scales with dataset length so small files stay conservative and large
    files are not artificially capped at fixed numbers like 30 or 100.
    """
    n = max(int(row_count), 1)
    return {
        # Categoricals with fewer uniques than this get one-hot encoded.
        "onehot_max_categories": max(5, min(150, int(n**0.5 * 2))),
        # Below this class count, classification targets are considered ideal.
        "ideal_target_classes": max(2, min(50, int(n * 0.05))),
        # Warn when unique target values exceed this share of rows.
        "high_cardinality_ratio": 0.2,
        # Warn when nearly every row is a distinct class (likely an ID).
        "identifier_ratio": 0.9,
    }


def onehot_max_categories(row_count: int) -> int:
    return int(dataset_thresholds(row_count)["onehot_max_categories"])


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _name_matches(col: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize_name(col)
    return any(hint in normalized for hint in hints)


def _name_is_id(col: str) -> bool:
    normalized = _normalize_name(col)
    if any(hint in normalized for hint in _ID_HINTS):
        return True
    return normalized == "id" or normalized.endswith(" id")


def _is_unique_identifier(series: pd.Series, row_count: int) -> bool:
    if row_count == 0:
        return False
    non_null = series.dropna()
    if len(non_null) < row_count * 0.9:
        return False
    return non_null.nunique() >= len(non_null) * 0.98 and len(non_null) > 20


def _is_sequential_index(series: pd.Series, row_count: int) -> bool:
    if row_count == 0:
        return False
    if _normalize_name(series.name or "") in {"index", "unnamed: 0"}:
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if len(non_null) != row_count or non_null.nunique() != row_count:
        return False
    return float(non_null.min()) == 1.0 and float(non_null.max()) == float(row_count)


def _value_match_rate(series: pd.Series, pattern: re.Pattern[str], sample_size: int = 80) -> float:
    sample = series.dropna().astype(str).str.strip().head(sample_size)
    if sample.empty:
        return 0.0
    return float(sample.apply(lambda v: bool(pattern.match(v))).mean())


def detect_column_role(col: str, series: pd.Series, row_count: int) -> str:
    if _name_is_id(col) or _is_sequential_index(series, row_count):
        return "identifier"
    if _name_matches(col, _PHONE_HINTS) or _value_match_rate(series, _PHONE_RE) >= 0.7:
        return "phone"
    if _name_matches(col, _EMAIL_HINTS) or _value_match_rate(series, _EMAIL_RE) >= 0.7:
        return "email"
    if _name_matches(col, _URL_HINTS) or _value_match_rate(series, _URL_RE) >= 0.7:
        return "url"
    if _name_matches(col, _NAME_HINTS) or _normalize_name(col) == "name":
        return "person_name"
    if not pd.api.types.is_numeric_dtype(series) and _is_unique_identifier(series, row_count):
        return "identifier"
    return "feature"


def detect_column_roles(df: pd.DataFrame) -> dict[str, str]:
    row_count = len(df)
    return {col: detect_column_role(col, df[col], row_count) for col in df.columns}


def get_ml_excluded_columns(
    df: pd.DataFrame,
    column_roles: dict[str, str],
    target_column: str | None = None,
) -> list[str]:
    excluded: list[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        role = column_roles.get(col, "feature")
        if role in {"identifier", "phone", "email", "url", "person_name"}:
            excluded.append(col)
    return excluded


def suggest_ml_targets(
    df: pd.DataFrame,
    column_types: dict[str, str],
    column_roles: dict[str, str],
) -> list[str]:
    """Suggest columns that make sensible prediction targets, preferring lower cardinality."""
    categorical_candidates: list[tuple[int, str]] = []
    numeric_candidates: list[str] = []
    thresholds = dataset_thresholds(len(df))
    ideal_classes = int(thresholds["ideal_target_classes"])

    for col in df.columns:
        role = column_roles.get(col, "feature")
        if role in {"identifier", "phone", "email", "url", "person_name"}:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if nunique < 2:
            continue
        col_type = column_types.get(col, "unknown")

        if col_type == "numeric":
            numeric_candidates.append(col)
        elif col_type in {"categorical", "boolean", "text"}:
            categorical_candidates.append((nunique, col))

    # Prefer categoricals at or below the ideal class count, then the rest.
    categorical_candidates.sort(key=lambda item: (item[0] > ideal_classes, item[0]))
    ordered = [col for _, col in categorical_candidates] + numeric_candidates
    return ordered[:10]


def validate_classification_target(y: pd.Series) -> dict[str, Any]:
    """Return warnings about target suitability; never blocks training."""
    labels = y.astype(str)
    counts = labels.value_counts()
    n_classes = int(len(counts))
    n_rows = int(len(labels))
    thresholds = dataset_thresholds(n_rows)
    ideal_classes = int(thresholds["ideal_target_classes"])
    high_ratio = float(thresholds["high_cardinality_ratio"])
    id_ratio = float(thresholds["identifier_ratio"])
    class_ratio = n_classes / max(n_rows, 1)

    warnings: list[str] = []

    if class_ratio >= id_ratio:
        warnings.append(
            f"Target has {n_classes} unique values in {n_rows} rows ({class_ratio:.0%} of rows). "
            "It likely behaves like an identifier — model metrics may be meaningless."
        )
    elif class_ratio >= high_ratio:
        warnings.append(
            f"Target has {n_classes} classes ({class_ratio:.0%} of rows). "
            "Training may be slow and metrics unreliable; a lower-cardinality column is usually better."
        )
    elif n_classes > ideal_classes:
        warnings.append(
            f"Target has {n_classes} classes (ideal for this dataset size: ≤{ideal_classes}). "
            "Results may be less stable than with fewer categories."
        )

    singletons = int((counts < 2).sum())
    if singletons > 0:
        warnings.append(
            f"{singletons} of {n_classes} classes appear only once; hold-out metrics can be very unstable."
        )

    avg_per_class = n_rows / max(n_classes, 1)
    if avg_per_class < 5 and class_ratio < id_ratio:
        warnings.append(
            f"Average of {avg_per_class:.1f} rows per class — consider more data or fewer categories."
        )

    return {"n_classes": n_classes, "warnings": warnings}
