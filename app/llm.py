"""Single shared OpenAI-compatible client (works with the LiteLLM proxy)."""

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
        timeout=60.0,   # never hang forever on a slow proxy
        max_retries=2,  # built-in retry with backoff for transient errors
    )