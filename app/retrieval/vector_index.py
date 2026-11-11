"""In-memory vector index: one NumPy matrix, cosine similarity as a dot product."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.ingestion import Chunk
from app.retrieval.filters import ChunkFilter


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {embeddings.shape[0]} embedding rows")
        self._chunks = list(chunks)
        self._matrix = embeddings  # rows are unit-length, so dot product == cosine similarity

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        chunk_filter: ChunkFilter | None = None,
    ) -> list[SearchResult]:
        scores = self._matrix @ query_vector
        if chunk_filter is not None:
            mask = np.fromiter(
                (chunk_filter.accepts(c) for c in self._chunks), dtype=bool, count=len(self._chunks)
            )
            scores = np.where(mask, scores, -np.inf)

        # A full sort is simple and instant at this corpus size (462 rows).
        ranked = [i for i in np.argsort(-scores) if np.isfinite(scores[i])]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in ranked[:top_k]]