from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["anomaly"])


class AnomalyRequest(BaseModel):
    session_id: str
    contamination: float = 0.05


@router.post("/anomaly")
def run_anomaly_detection(payload: AnomalyRequest) -> dict:
    if not (0.01 <= payload.contamination <= 0.25):
        raise HTTPException(status_code=400, detail="contamination must be between 0.01 and 0.25")

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
