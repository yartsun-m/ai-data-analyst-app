from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODELS = (
    "gemini-2.5-flash,"
    "gemini-2.5-flash-lite,"
    "gemini-3.5-flash,"
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


def _summarize_failures(failures: list[str]) -> str:
    unique = list(dict.fromkeys(failures))
    if len(unique) == 1:
        return unique[0]
    preview = "; ".join(unique[:3])
    if len(unique) > 3:
        preview += f"; +{len(unique) - 3} more"
    return preview


def _is_invalid_api_key_error(status: int, detail: str) -> bool:
    if status == 401:
        return True
    if status != 403:
        return False
    lowered = detail.lower()
    return any(
        phrase in lowered
        for phrase in (
            "api key not valid",
            "invalid api key",
            "api_key_invalid",
            "key is invalid",
            "api key expired",
        )
    )


def _is_referrer_blocked_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(word in lowered for word in ("referer", "referrer", "blocked", "leaked"))


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

        failures: list[str] = []
        invalid_key_indexes: set[int] = set()
        unavailable_models: set[str] = set()
        primary_model = self.models[0]
        last_detail = ""

        async with httpx.AsyncClient(timeout=90.0) as client:
            for model in self.models:
                if model in unavailable_models:
                    continue

                for key_index, api_key in enumerate(self.api_keys):
                    if key_index in invalid_key_indexes:
                        continue

                    for attempt in range(settings.gemini_max_retries):
                        try:
                            answer = await self._generate_with_fallbacks(
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
                            last_detail = detail

                            if status == 404:
                                msg = f"Model `{model}` not found"
                                failures.append(msg)
                                unavailable_models.add(model)
                                logger.warning("Gemini model not found: %s", model)
                                break

                            if status == 429:
                                msg = f"Rate limit on `{model}` (key #{key_index + 1})"
                                failures.append(msg)
                                logger.warning(
                                    "Gemini 429 model=%s key=#%s attempt=%s",
                                    model,
                                    key_index + 1,
                                    attempt + 1,
                                )
                                wait = _parse_retry_after(exc.response) or (2**attempt)
                                if attempt < settings.gemini_max_retries - 1:
                                    await asyncio.sleep(wait)
                                    continue
                                break

                            if status in {401, 403}:
                                if _is_invalid_api_key_error(status, detail):
                                    msg = f"Invalid API key #{key_index + 1}: {detail}"
                                    failures.append(msg)
                                    invalid_key_indexes.add(key_index)
                                else:
                                    msg = f"`{model}` key #{key_index + 1}: {detail or status}"
                                    failures.append(msg)
                                logger.warning(
                                    "Gemini auth/permission error model=%s key=#%s: %s",
                                    model,
                                    key_index + 1,
                                    detail,
                                )
                                break

                            if status in {500, 502, 503, 504}:
                                msg = f"Gemini server error {status} on `{model}`"
                                failures.append(msg)
                                await asyncio.sleep(2**attempt)
                                continue

                            return _fallback_response(
                                user_prompt,
                                f"Gemini API error ({status}) on `{model}`: {detail}",
                            )
                        except httpx.RequestError as exc:
                            failures.append(f"Network error on `{model}` (key #{key_index + 1}): {exc}")
                            break

        tried_models = ", ".join(f"`{m}`" for m in self.models)
        summary = _summarize_failures(failures)
        hint = ""
        if _is_referrer_blocked_error(last_detail):
            hint = (
                " API key appears restricted for server use (referrer/IP restriction). "
                "Create a new unrestricted key at https://aistudio.google.com/apikey."
            )
        elif invalid_key_indexes and len(invalid_key_indexes) == len(self.api_keys):
            hint = (
                " All API keys were rejected. Re-paste GEMINI_API_KEYS in Render — "
                "local .env changes do not update Render automatically."
            )
        else:
            hint = (
                " Check https://YOUR-BACKEND.onrender.com/health/llm for a live key probe. "
                "Set GEMINI_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite on Render."
            )

        return _fallback_response(
            user_prompt,
            f"Gemini unavailable ({summary}).{hint} "
            f"Tried models: {tried_models} across {len(self.api_keys)} API key(s).",
        )

    async def _generate_with_fallbacks(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            return await self._generate(
                client, api_key, model, system_prompt, user_prompt, use_system_instruction=True
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {400, 403}:
                raise
            return await self._generate(
                client, api_key, model, system_prompt, user_prompt, use_system_instruction=False
            )

    async def _generate(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        use_system_instruction: bool = True,
    ) -> str:
        url = f"{self.base_url}/models/{model}:generateContent"
        if use_system_instruction:
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.2},
            }
        else:
            combined = f"{system_prompt}\n\n{user_prompt}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": combined}]}],
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
