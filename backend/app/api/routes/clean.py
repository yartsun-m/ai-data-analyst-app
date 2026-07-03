from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["clean"])


class CleanRequest(BaseModel):
    session_id: str
    target_column: str | None = None


@router.post("/clean")
def clean_dataset(payload: CleanRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.target_column:
        session.target_column = payload.target_column

    report = analysis_orchestrator.clean_session(session)
    return {
        "session_id": payload.session_id,
        "cleaning_report": report,
        "rows_after": report.get("rows_after"),
        "columns_after": report.get("columns_after"),
    }
