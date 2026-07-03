from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.utils.json_utils import to_json_safe

MAX_PAGE_SIZE = 500
DEFAULT_PAGE_SIZE = 50


def get_dataset_page(
    df: pd.DataFrame,
    column_types: dict[str, str] | None = None,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    sort_by: str | None = None,
    sort_order: str = "asc",
    search: str | None = None,
) -> dict[str, Any]:
    """Return a paginated, optionally filtered/sorted slice of a dataframe."""
    if df.empty:
        return _empty_page(page, page_size, column_types or {})

    page = max(page, 1)
    page_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    column_types = column_types or {}

    work = df.copy()
    total_rows = len(work)

    if search and search.strip():
        query = search.strip()
        str_frame = work.astype(str)
        mask = str_frame.apply(
            lambda row: row.str.contains(query, case=False, na=False, regex=False).any(),
            axis=1,
        )
        work = work.loc[mask]
    filtered_rows = len(work)

    if sort_by and sort_by in work.columns:
        ascending = sort_order.lower() != "desc"
        work = work.sort_values(by=sort_by, ascending=ascending, kind="mergesort", na_position="last")

    total_pages = max(1, math.ceil(filtered_rows / page_size)) if filtered_rows else 1
    page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = work.iloc[start:end]

    columns = [
        {"name": str(col), "type": column_types.get(str(col), _infer_column_type(work[col]))}
        for col in work.columns
    ]
    rows = [_row_to_record(row, work.columns) for _, row in page_df.iterrows()]

    return {
        "columns": columns,
        "rows": rows,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "total_pages": total_pages,
        "row_offset": start,
        "sort_by": sort_by,
        "sort_order": "desc" if sort_order.lower() == "desc" else "asc",
        "search": search.strip() if search and search.strip() else None,
    }


def _empty_page(page: int, page_size: int, column_types: dict[str, str]) -> dict[str, Any]:
    return {
        "columns": [{"name": name, "type": dtype} for name, dtype in column_types.items()],
        "rows": [],
        "page": page,
        "page_size": page_size,
        "total_rows": 0,
        "filtered_rows": 0,
        "total_pages": 1,
        "row_offset": 0,
        "sort_by": None,
        "sort_order": "asc",
        "search": None,
    }


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _row_to_record(row: pd.Series, columns: pd.Index) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for col in columns:
        value = row[col]
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            record[str(col)] = None
            continue
        if isinstance(value, pd.Timestamp):
            record[str(col)] = value.isoformat()
            continue
        record[str(col)] = to_json_safe(value)
    return record
