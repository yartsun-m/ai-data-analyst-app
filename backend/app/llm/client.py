from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODELS = (
    "gemini-3.5-flash,"
    "gemini-3.0-flash,"
    "gemini-2.5-flash,"
    "gemini-2.5-flash-lite,"
    "gemini-3.1-flash-lite"
)


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        raise NotImplementedError


def _fallback_response(user_prompt: str, reason: str) -> dict[str, str]:
    return {
        "answer": (
            f"{reason}\n\n"
            "Here is an answer based on the computed dataset summary (no LLM call):\n\n"
            f"{user_prompt[:3000]}"
        ),
        "model_used": "none",
    }


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class GeminiClient(LLMClient):
    """Google Gemini client with multi-key and multi-model fallback."""

    def __init__(self) -> None:
        self.api_keys = settings.gemini_api_key_list
        self.base_url = settings.gemini_base_url.rstrip("/")
        self.models = settings.gemini_model_list

    async def chat(self, system_prompt: str, user_prompt: str) -> dict[str, str]:
        if not self.api_keys:
            return _fallback_response(
                user_prompt,
                "LLM is not configured. Set GEMINI_API_KEYS in your environment (comma-separated).",
            )

        if not self.models:
            return _fallback_response(user_prompt, "No Gemini models configured.")

        last_error: str | None = None
        primary_model = self.models[0]

        async with httpx.AsyncClient(timeout=90.0) as client:
            for model in self.models:
                for key_index, api_key in enumerate(self.api_keys):
                    for attempt in range(settings.gemini_max_retries):
                        try:
                            answer = await self._generate(
                                client, api_key, model, system_prompt, user_prompt
                            )
                            if model != primary_model or key_index > 0:
                                note_parts = []
                                if model != primary_model:
                                    note_parts.append(f"model `{model}`")
                                if key_index > 0:
                                    note_parts.append(f"API key #{key_index + 1}")
                                answer = (
                                    f"_Note: Fallback used ({', '.join(note_parts)})._\n\n{answer}"
                                )
                            return {"answer": answer, "model_used": model}
                        except httpx.HTTPStatusError as exc:
                            status = exc.response.status_code
                            detail = _extract_error_message(exc.response)

                            if status == 404:
                                last_error = f"Model `{model}` not found"
                                logger.warning("Gemini model not found: %s", model)
                                break

                            if status == 429:
                                wait = _parse_retry_after(exc.response) or (2**attempt)
                                last_error = f"Rate limit on `{model}` (key #{key_index + 1})"
                                logger.warning(
                                    "Gemini 429 model=%s key=#%s attempt=%s wait=%.1fs",
                                    model,
                                    key_index + 1,
                                    attempt + 1,
                                    wait,
                                )
                                if attempt < settings.gemini_max_retries - 1:
                                    await asyncio.sleep(wait)
                                    continue
                                break

                            if status in {401, 403}:
                                last_error = f"Auth failed for key #{key_index + 1} ({status})"
                                logger.warning("Gemini auth error key=#%s: %s", key_index + 1, detail)
                                break

                            if status in {500, 502, 503, 504}:
                                last_error = f"Gemini server error {status} on `{model}`"
                                await asyncio.sleep(2**attempt)
                                continue

                            return _fallback_response(
                                user_prompt,
                                f"Gemini API error ({status}) on `{model}`: {detail}",
                            )
                        except httpx.RequestError as exc:
                            last_error = f"Network error: {exc}"
                            break

        tried_models = ", ".join(f"`{m}`" for m in self.models)
        return _fallback_response(
            user_prompt,
            f"All Gemini models/keys were rate-limited or unavailable ({last_error}). "
            f"Tried models: {tried_models} across {len(self.api_keys)} API key(s).",
        )

    async def _generate(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        url = f"{self.base_url}/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        response = await client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise ValueError(f"No response from Gemini{f' ({block})' if block else ''}")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_parts = [p.get("text", "") for p in parts if p.get("text")]
        if not text_parts:
            raise ValueError("Empty response from Gemini")
        return "\n".join(text_parts)


def _extract_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
        error = data.get("error") or {}
        return str(error.get("message") or response.text)[:200]
    except Exception:
        return response.text[:200]


def get_llm_client() -> LLMClient:
    return GeminiClient()
