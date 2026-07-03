from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import settings


def save_model_pipeline(session_id: str, pipeline: Pipeline) -> Path:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    path = settings.models_dir / f"{session_id}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model_pipeline(model_path: Path | str) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_records(pipeline: Pipeline, records: list[dict[str, Any]]) -> list[Any]:
    if not records:
        return []
    df = pd.DataFrame(records)
    preds = pipeline.predict(df)
    return preds.tolist()
