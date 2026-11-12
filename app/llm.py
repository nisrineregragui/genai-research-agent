"""Single shared OpenAI-compatible client (works with the LiteLLM proxy).

Retries and timeout come from settings: the SDK waits for the delay the server asks
for on rate limits (HTTP 429), so enough retries turn a rate limit into a delay.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import load_settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    settings = load_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=float(settings.llm_timeout_seconds),
        max_retries=settings.llm_max_retries,
    )