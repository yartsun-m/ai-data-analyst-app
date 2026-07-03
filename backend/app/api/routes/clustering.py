from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.analysis_orchestrator import analysis_orchestrator
from app.utils.storage import session_store

router = APIRouter(tags=["clustering"])


class ClusteringRequest(BaseModel):
    session_id: str
    n_clusters: int | None = None


@router.post("/clustering")
def run_clustering(payload: ClusteringRequest) -> dict:
    try:
        session = session_store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.n_clusters is not None and not (2 <= payload.n_clusters <= 20):
        raise HTTPException(status_code=400, detail="n_clusters must be between 2 and 20")

    try:
        result = analysis_orchestrator.clustering_session(session, n_clusters=payload.n_clusters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": payload.session_id, "clustering": result}


@router.get("/clustering")
def get_clustering(session_id: str = Query(...)) -> dict:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if session.clustering is None:
        result = analysis_orchestrator.clustering_session(session)
        return {"session_id": session_id, "clustering": result}
    return {"session_id": session_id, "clustering": session.clustering}
