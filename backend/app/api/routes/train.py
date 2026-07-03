from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["train"])


class TrainRequest(BaseModel):
    session_id: str
    target_column: str


@router.post("/train")
def train_model(payload: TrainRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        ml_results = analysis_orchestrator.train_session(session, target_column=payload.target_column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {"session_id": payload.session_id, "ml_results": ml_results}
