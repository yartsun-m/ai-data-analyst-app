from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["clean"])


class CleanRequest(BaseModel):
    session_id: str
    target_column: str | None = None
    outlier_strategy: str = Field(default="winsorize", pattern="^(none|clip|winsorize|remove)$")


@router.post("/clean")
def clean_dataset(payload: CleanRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.target_column:
        session.target_column = payload.target_column

    report = analysis_orchestrator.clean_session(
        session,
        outlier_strategy=payload.outlier_strategy or settings.default_outlier_strategy,
    )
    return {
        "session_id": payload.session_id,
        "cleaning_report": report,
        "rows_after": report.get("rows_after"),
        "columns_after": report.get("columns_after"),
    }
