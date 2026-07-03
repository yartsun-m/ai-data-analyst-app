from __future__ import annotations

from fastapi import APIRouter

from app.services.llm_health_service import run_llm_health_check

router = APIRouter(tags=["health"])


@router.get("/health/llm")
async def health_llm() -> dict:
    """Probe configured Gemini keys/models (for debugging Render env)."""
    return await run_llm_health_check()
