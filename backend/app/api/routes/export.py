from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.utils.storage import session_store

router = APIRouter(tags=["export"])


@router.get("/export")
def export_dataset(
    session_id: str = Query(...),
    variant: str = Query(default="raw", pattern="^(raw|cleaned)$"),
) -> StreamingResponse:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if variant == "cleaned":
        if session.cleaned_df is None:
            raise HTTPException(status_code=400, detail="No cleaned dataset. Run cleaning first.")
        df = session.cleaned_df
        suffix = "cleaned"
    else:
        df = session_store.ensure_raw_df(session)
        suffix = "raw"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    base = session.filename.rsplit(".", 1)[0] if session.filename else "dataset"
    filename = f"{base}-{suffix}.csv"

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
