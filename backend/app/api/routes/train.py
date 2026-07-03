from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.middleware.rate_limit import limiter
from app.config import settings
from app.services.analysis_orchestrator import analysis_orchestrator
from app.services.job_service import get_job_status, submit_training_job
from app.utils.storage import session_store

router = APIRouter(tags=["train"])


class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    async_mode: bool = True


@router.post("/train")
@limiter.limit(settings.rate_limit_train)
def train_model(request: Request, payload: TrainRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.async_mode:
        job_id = submit_training_job(payload.session_id, payload.target_column)
        return {
            "session_id": payload.session_id,
            "job_id": job_id,
            "status": "pending",
            "message": "Training started. Poll GET /train/status?job_id=... for results.",
        }

    try:
        ml_results = analysis_orchestrator.train_session(session, target_column=payload.target_column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {"session_id": payload.session_id, "ml_results": ml_results}


@router.get("/train/status")
def train_status(job_id: str = Query(...)) -> dict:
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    response = {
        "job_id": job["job_id"],
        "session_id": job["session_id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
    }
    if job["status"] == "completed" and job.get("result"):
        response["ml_results"] = job["result"]
    return response
