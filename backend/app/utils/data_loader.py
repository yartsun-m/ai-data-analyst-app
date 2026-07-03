from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def dataframe_preview(df: pd.DataFrame, rows: int = 20) -> list[dict]:
    preview = df.head(rows).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
    preview = preview.where(preview.notna(), None)
    return preview.to_dict(orient="records")
