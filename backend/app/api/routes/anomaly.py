from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["anomaly"])


class AnomalyRequest(BaseModel):
    session_id: str
    contamination: float = Field(default=0.05, ge=0.01, le=0.25)


@router.post("/anomaly")
def run_anomaly_detection(payload: AnomalyRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = analysis_orchestrator.anomaly_session(session, contamination=payload.contamination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": payload.session_id, "anomaly": result}


@router.get("/anomaly")
def get_anomaly(session_id: str = Query(...)) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.anomaly is None:
        result = analysis_orchestrator.anomaly_session(session)
        return {"session_id": session_id, "anomaly": result}
    return {"session_id": session_id, "anomaly": session.anomaly}
