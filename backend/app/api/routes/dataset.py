from fastapi import APIRouter, HTTPException, Query

from app.services.dataset_viewer_service import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, get_dataset_page
from app.utils.json_utils import to_json_safe
from app.utils.storage import session_store

router = APIRouter(tags=["dataset"])


@router.get("/dataset")
def get_dataset(
    session_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str | None = Query(default=None),
    sort_order: str = Query(default="asc"),
    search: str | None = Query(default=None),
    variant: str = Query(default="raw"),
) -> dict:
    if sort_order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="sort_order must be asc or desc")
    if variant not in {"raw", "cleaned"}:
        raise HTTPException(status_code=400, detail="variant must be raw or cleaned")

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
    return to_json_safe(payload)
