"""Central configuration: every credential and model name comes from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str | None
    chat_model: str
    embedding_model: str


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        openai_api_key=_required("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_UR") or None,
        chat_model=_required("OPENAI_MODEL"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )