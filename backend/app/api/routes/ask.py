from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.middleware.rate_limit import limiter
from app.services.analysis_orchestrator import analysis_orchestrator
from app.llm.client import get_llm_client
from app.llm.context_builder import SYSTEM_PROMPT, build_llm_context
from app.utils.storage import session_store

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3, max_length=2000)
    stream: bool = False


@router.post("/ask")
@limiter.limit(settings.rate_limit_ask)
async def ask_question(request: Request, payload: AskRequest):
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.stream:
        return StreamingResponse(
            _stream_answer(session, payload.question),
            media_type="text/event-stream",
        )

    result = await analysis_orchestrator.ask_session(session, payload.question)
    return {"session_id": payload.session_id, **result}


async def _stream_answer(session, question: str):
    df = session_store.get_active_df(session)
    explainability = (session.ml_results or {}).get("explainability")
    context = build_llm_context(
        df=df,
        profile=session.profile,
        cleaning_report=session.cleaning_report,
        eda=session.eda,
        ml_results=session.ml_results,
        explainability=explainability,
        question=question,
        chat_history=session.chat_history,
    )
    client = get_llm_client()
    full_answer: list[str] = []
    async for chunk in client.stream_chat(SYSTEM_PROMPT, context):
        full_answer.append(chunk)
        yield f"data: {json.dumps({'token': chunk})}\n\n"
    answer = "".join(full_answer)
    session_store.append_chat(session, "user", question)
    session_store.append_chat(session, "assistant", answer)
    yield f"data: {json.dumps({'done': True, 'model_used': 'gemini'})}\n\n"
