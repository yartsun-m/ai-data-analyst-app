from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["eda"])


@router.get("/eda")
def get_eda(session_id: str = Query(...)) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    eda = analysis_orchestrator.eda_session(session)
    return {"session_id": session_id, "eda": eda}
