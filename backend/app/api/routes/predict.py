from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.model_store import load_model_pipeline, predict_records
from app.utils.storage import session_store

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    session_id: str
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


@router.post("/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.model_path:
        raise HTTPException(status_code=400, detail="No trained model. Run training first.")

    try:
        pipeline = load_model_pipeline(session.model_path)
        predictions = predict_records(pipeline, payload.records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "session_id": payload.session_id,
        "model": (session.ml_results or {}).get("best_model"),
        "task_type": (session.ml_results or {}).get("task_type"),
        "predictions": predictions,
        "count": len(predictions),
    }
