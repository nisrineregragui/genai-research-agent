"""Embedding computation with an on-disk cache.

Vectors are L2-normalized once here, so cosine similarity later is a plain dot product.
The cache key covers the model name and every chunk text: changing either one
(new model, edited corpus, different chunking) automatically triggers re-embedding.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, client: OpenAI, model: str, batch_size: int) -> None:
        self._client = client
        self.model = model
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of unit-length vectors, in input order."""
        if not texts:
            raise ValueError("Nothing to embed")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(model=self.model, input=batch)
            # Sort by index so the order is right whatever order the proxy returns.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def load_or_compute_embeddings(embedder: Embedder, texts: Sequence[str], cache_dir: Path) -> np.ndarray:
    path = cache_dir / f"embeddings-{_cache_key(embedder.model, texts)}.npy"
    if path.exists():
        matrix = np.load(path)
        if matrix.shape[0] == len(texts):
            logger.info("Loaded %d embeddings from cache (%s)", len(texts), path.name)
            return matrix
        logger.warning("Ignoring cache file with unexpected shape: %s", path.name)

    logger.info("Computing embeddings for %d texts with %s ...", len(texts), embedder.model)
    matrix = embedder.embed(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp_path, matrix)
    tmp_path.replace(path)  # atomic rename: an interrupted run never leaves a corrupt cache
    return matrix


def _cache_key(model: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256(model.encode("utf-8"))
    for text in texts:
        digest.update(b"\0")  # separator, so ["ab", "c"] and ["a", "bc"] hash differently
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero on an empty vector
    return matrix / norms