"""Central configuration.

Credentials and model names come only from environment variables. Tunable
parameters have one documented default here and can be overridden by an
environment variable without touching the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Loads .env if present. Never overrides variables that are already set,
# so values injected by `docker run --env-file` always win.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults for tunable parameters (environment variable that overrides each one in brackets).
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"      # [OPENAI_EMBEDDING_MODEL]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"   # [CORPUS_DIR]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"             # [CACHE_DIR]
# ~900 tokens: keeps typical sections whole and splits only the few oversized ones.
DEFAULT_CHUNK_MAX_CHARS = 3500                          # [CHUNK_MAX_CHARS]
# Texts per embeddings request: keeps each request well under the API's size limits.
DEFAULT_EMBEDDING_BATCH_SIZE = 100                      # [EMBEDDING_BATCH_SIZE]
# Chunks returned by a single unfiltered search (also the safety-net search of a plan).
DEFAULT_RETRIEVAL_TOP_K = 5                             # [RETRIEVAL_TOP_K]
# Chunks kept per planned retrieval task (beyond ~3 the hits were mostly noise).
DEFAULT_TASK_TOP_K = 3                                  # [TASK_TOP_K]
# Upper bound on chunks passed to the LLM for one question.
DEFAULT_MAX_CONTEXT_CHUNKS = 12                         # [MAX_CONTEXT_CHUNKS]
# Upper bound on retrieval tasks the Planner may produce for one question.
DEFAULT_MAX_PLAN_TASKS = 6                              # [MAX_PLAN_TASKS]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str | None
    chat_model: str
    embedding_model: str
    corpus_dir: Path
    cache_dir: Path
    chunk_max_chars: int
    embedding_batch_size: int
    retrieval_top_k: int
    task_top_k: int
    max_context_chunks: int
    max_plan_tasks: int


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise RuntimeError(f"Environment variable {name} must be >= {minimum}, got {value}")
    return value


def load_settings() -> Settings:
    return Settings(
        openai_api_key=_required("OPENAI_API_KEY"),
        # Accept both spellings: the .env provided by Syntax says OPENAI_BASE_UR (missing "L").
        openai_base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_UR") or None,
        chat_model=_required("OPENAI_MODEL"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        corpus_dir=Path(os.getenv("CORPUS_DIR") or DEFAULT_CORPUS_DIR),
        cache_dir=Path(os.getenv("CACHE_DIR") or DEFAULT_CACHE_DIR),
        chunk_max_chars=_int("CHUNK_MAX_CHARS", DEFAULT_CHUNK_MAX_CHARS),
        embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE),
        retrieval_top_k=_int("RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K),
        task_top_k=_int("TASK_TOP_K", DEFAULT_TASK_TOP_K),
        max_context_chunks=_int("MAX_CONTEXT_CHUNKS", DEFAULT_MAX_CONTEXT_CHUNKS),
        max_plan_tasks=_int("MAX_PLAN_TASKS", DEFAULT_MAX_PLAN_TASKS),
    )