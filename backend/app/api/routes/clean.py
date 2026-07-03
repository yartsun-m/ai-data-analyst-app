from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.json_utils import to_json_safe
from app.utils.storage import session_store

router = APIRouter(tags=["clean"])

VALID_OUTLIER_STRATEGIES = {"none", "clip", "winsorize", "remove"}


class CleanRequest(BaseModel):
    session_id: str
    target_column: str | None = None
    outlier_strategy: str = "winsorize"


@router.post("/clean")
def clean_dataset(payload: CleanRequest) -> dict:
    if payload.outlier_strategy not in VALID_OUTLIER_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"outlier_strategy must be one of: {', '.join(sorted(VALID_OUTLIER_STRATEGIES))}",
        )

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
    return to_json_safe({
        "session_id": payload.session_id,
        "cleaning_report": report,
        "rows_after": report.get("rows_after"),
        "columns_after": report.get("columns_after"),
    })
