from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.data_loader import dataframe_preview
from app.utils.json_utils import to_json_safe
from app.utils.storage import session_store

router = APIRouter(tags=["profile"])


@router.get("/profile")
def get_profile(
    session_id: str = Query(...),
    target_column: str | None = Query(default=None),
) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    profile = analysis_orchestrator.profile_session(session, target_column=target_column)
    df = session_store.ensure_raw_df(session)
    active_df = session_store.get_active_df(session)
    return to_json_safe({
        "session_id": session_id,
        "profile": profile,
        "preview": dataframe_preview(df),
        "active_columns": list(active_df.columns),
    })
