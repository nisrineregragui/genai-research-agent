"""Retrieval: embeddings, filters and in-memory search over corpus chunks."""

from app.retrieval.embeddings import Embedder, load_or_compute_embeddings
from app.retrieval.filters import ChunkFilter
from app.retrieval.vector_index import SearchResult, VectorIndex

__all__ = ["ChunkFilter", "Embedder", "SearchResult", "VectorIndex", "load_or_compute_embeddings"]