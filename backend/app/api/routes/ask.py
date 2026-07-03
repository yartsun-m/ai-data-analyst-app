from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3, max_length=2000)


@router.post("/ask")
async def ask_question(payload: AskRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = await analysis_orchestrator.ask_session(session, payload.question)
    return {"session_id": payload.session_id, **result}
