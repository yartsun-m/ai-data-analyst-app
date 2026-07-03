from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.db.sqlite_store import db
from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


def submit_training_job(session_id: str, target_column: str) -> str:
    job_id = str(uuid.uuid4())
    db.create_job(job_id, session_id, "train")
    _executor.submit(_run_training, job_id, session_id, target_column)
    return job_id


def _run_training(job_id: str, session_id: str, target_column: str) -> None:
    db.update_job(job_id, status="running", progress=0.1)
    try:
        session = session_store.get(session_id)
        db.update_job(job_id, progress=0.3)
        ml_results = analysis_orchestrator.train_session(session, target_column=target_column)
        session_store.persist(session)
        db.update_job(job_id, status="completed", progress=1.0, result=ml_results)
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        db.update_job(job_id, status="failed", error=str(exc), progress=1.0)


def get_job_status(job_id: str) -> dict[str, Any] | None:
    return db.get_job(job_id)
