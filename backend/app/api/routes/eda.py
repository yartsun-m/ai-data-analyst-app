from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["eda"])


class CustomEdaRequest(BaseModel):
    session_id: str
    x_column: str
    y_column: str | None = None
    chart_type: str = Field(default="scatter", pattern="^(scatter|line|box|histogram)$")


@router.get("/eda")
def get_eda(session_id: str = Query(...)) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    eda = analysis_orchestrator.eda_session(session)
    return {"session_id": session_id, "eda": eda}


@router.post("/eda/custom")
def custom_eda(payload: CustomEdaRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        chart = analysis_orchestrator.custom_eda_session(
            session,
            x_column=payload.x_column,
            y_column=payload.y_column,
            chart_type=payload.chart_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": payload.session_id, "chart": chart}
