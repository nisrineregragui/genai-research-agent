"""Composition root: builds the long-lived components once, from settings.

This is the only module that knows how all the layers fit together; the layers
themselves only receive the values they need.
"""

from __future__ import annotations

from app.agent import Planner
from app.config import Settings
from app.ingestion import load_corpus
from app.llm import get_client
from app.retrieval import Embedder, TaskRetriever, VectorIndex, load_or_compute_embeddings


def build_retriever(settings: Settings) -> TaskRetriever:
    chunks = load_corpus(settings.corpus_dir, settings.chunk_max_chars)
    embedder = Embedder(get_client(), settings.embedding_model, settings.embedding_batch_size)
    matrix = load_or_compute_embeddings(embedder, [c.embedding_text for c in chunks], settings.cache_dir)
    return TaskRetriever(
        VectorIndex(chunks, matrix),
        embedder,
        top_k_per_task=settings.task_top_k,
        safety_net_top_k=settings.retrieval_top_k,
        max_chunks=settings.max_context_chunks,
    )


def build_planner(settings: Settings, retriever: TaskRetriever) -> Planner:
    return Planner(
        get_client(),
        settings.chat_model,
        available_drugs=retriever.available_drugs,
        available_section_types=retriever.available_section_types,
        max_tasks=settings.max_plan_tasks,
    )