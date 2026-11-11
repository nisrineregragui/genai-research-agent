"""Retrieval: embeddings, filters, in-memory search and plan execution over corpus chunks."""

from app.retrieval.embeddings import Embedder, load_or_compute_embeddings
from app.retrieval.filters import ChunkFilter
from app.retrieval.tasks import SAFETY_NET_LABEL, RetrievalTask, RetrievedChunk, TaskRetriever
from app.retrieval.vector_index import SearchResult, VectorIndex

__all__ = [
    "SAFETY_NET_LABEL",
    "ChunkFilter",
    "Embedder",
    "RetrievalTask",
    "RetrievedChunk",
    "SearchResult",
    "TaskRetriever",
    "VectorIndex",
    "load_or_compute_embeddings",
]