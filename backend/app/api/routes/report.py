from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.services.analysis_orchestrator import analysis_orchestrator
from app.services.report_service import build_html_report
from app.utils.storage import session_store

router = APIRouter(tags=["report"])


@router.get("/report")
def download_report(session_id: str = Query(...)) -> HTMLResponse:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Ensure EDA exists so charts are included in the report.
    if session.eda is None:
        analysis_orchestrator.eda_session(session)

    df = session_store.get_active_df(session)
    html_content = build_html_report(
        df=df,
        profile=session.profile,
        cleaning_report=session.cleaning_report,
        eda=session.eda,
        ml_results=session.ml_results,
        filename=session.filename,
    )
    safe_name = (session.filename or "report").rsplit(".", 1)[0]
    headers = {"Content-Disposition": f'attachment; filename="{safe_name}-report.html"'}
    return HTMLResponse(content=html_content, headers=headers)
