from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.dataset_viewer_service import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, get_dataset_page
from app.utils.storage import session_store

router = APIRouter(tags=["dataset"])


@router.get("/dataset")
def get_dataset(
    session_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str | None = Query(default=None),
    sort_order: str = Query(default="asc", pattern="^(asc|desc)$"),
    search: str | None = Query(default=None),
    variant: str = Query(default="raw", pattern="^(raw|cleaned)$"),
) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if variant == "cleaned":
        if session.cleaned_df is None:
            raise HTTPException(
                status_code=400,
                detail="No cleaned dataset available. Run the cleaning pipeline first or use variant=raw.",
            )
        df = session.cleaned_df
    else:
        df = session_store.ensure_raw_df(session)

    column_types = (session.profile or {}).get("column_types", {})
    payload = get_dataset_page(
        df,
        column_types,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )
    payload.update(
        {
            "session_id": session_id,
            "variant": variant,
            "filename": session.filename,
            "has_cleaned": session.cleaned_df is not None,
        }
    )
    return payload
