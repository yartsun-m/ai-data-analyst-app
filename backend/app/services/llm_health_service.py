from __future__ import annotations

import httpx

from app.config import settings

DEFAULT_PROBE_MODEL = "gemini-2.5-flash"


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def _extract_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        error = data.get("error") or {}
        return str(error.get("message") or response.text)[:240]
    except Exception:
        return response.text[:240]


async def probe_gemini_key(api_key: str, model: str = DEFAULT_PROBE_MODEL) -> dict[str, str | int]:
    """Send a minimal generateContent request to validate a key/model pair."""
    base_url = settings.gemini_base_url.rstrip("/")
    url = f"{base_url}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": "Reply with exactly: ok"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8},
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.RequestError as exc:
        return {"status": "error", "http_status": 0, "detail": f"network: {exc}"}

    if response.status_code == 200:
        return {"status": "ok", "http_status": 200, "detail": "ok"}

    return {
        "status": "error",
        "http_status": response.status_code,
        "detail": _extract_error(response),
    }


async def run_llm_health_check() -> dict:
    keys = settings.gemini_api_key_list
    models = settings.gemini_model_list or [DEFAULT_PROBE_MODEL]
    probe_models = models[:3]

    if not keys:
        return {
            "configured": False,
            "key_count": 0,
            "models": probe_models,
            "results": [],
            "hint": "Set GEMINI_API_KEYS in Render environment variables.",
        }

    results = []
    for index, key in enumerate(keys, start=1):
        for model in probe_models:
            probe = await probe_gemini_key(key, model)
            results.append(
                {
                    "key_index": index,
                    "key_hint": _mask_key(key),
                    "model": model,
                    **probe,
                }
            )

    any_ok = any(item.get("status") == "ok" for item in results)
    hint = None
    if not any_ok:
        details = " ".join(
            str(item.get("detail", "")).lower() for item in results if item.get("detail")
        )
        if "referer" in details or "referrer" in details or "blocked" in details:
            hint = (
                "API key appears restricted (HTTP referrer / app restriction). "
                "Create a new unrestricted key at https://aistudio.google.com/apikey for server use."
            )
        elif "not valid" in details or "invalid" in details:
            hint = "Keys on Render do not match valid AI Studio keys. Re-paste GEMINI_API_KEYS in Render."
        else:
            hint = (
                "All probes failed. Set GEMINI_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite "
                "and verify keys in Render match your local .env exactly."
            )

    return {
        "configured": True,
        "key_count": len(keys),
        "models": probe_models,
        "any_ok": any_ok,
        "results": results,
        "hint": hint,
    }
