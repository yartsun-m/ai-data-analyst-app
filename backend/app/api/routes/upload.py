from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.middleware.rate_limit import limiter
from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.data_loader import dataframe_preview
from app.utils.json_utils import to_json_safe
from app.utils.storage import session_store

router = APIRouter(tags=["upload"])


@router.post("/upload")
@limiter.limit(settings.rate_limit_upload)
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    session = session_store.create_session(file.filename, settings.upload_dir / "placeholder")
    dest = settings.upload_dir / f"{session.session_id}{suffix}"
    dest.write_bytes(content)
    session.raw_path = dest

    try:
        df = session_store.ensure_raw_df(session)
        profile = analysis_orchestrator.profile_session(session)
        preview = dataframe_preview(df, rows=settings.preview_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    session_store.persist(session)
    return to_json_safe({
        "session_id": session.session_id,
        "filename": file.filename,
        "preview": preview,
        "columns": list(df.columns),
        "profile": profile,
    })
