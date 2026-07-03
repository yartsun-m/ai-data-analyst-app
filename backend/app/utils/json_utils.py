from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return None
    if isinstance(obj, pd.Series):
        return to_json_safe(obj.tolist())
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def plotly_figure_to_dict(fig: Any) -> dict[str, Any]:
    """Convert a Plotly figure to a JSON-safe dict."""
    return json.loads(fig.to_json())
