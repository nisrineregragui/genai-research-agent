"""Retrieval tests with hand-made vectors and a fake embeddings client (no API calls)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ingestion import Chunk
from app.retrieval import ChunkFilter, Embedder, VectorIndex, load_or_compute_embeddings


def _chunk(i: int, drug: str = "Warfarin", section_type: str = "interactions") -> Chunk:
    return Chunk(
        id=f"doc#{i}", file=f"dailymed-{drug.lower()}.md", drug=drug, source="dailymed",
        source_url=None, section_path=("Section",), section_type=section_type, text=f"text {i}",
    )


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeEmbeddingsAPI:
    """Mimics client.embeddings: vector = [len(text), 1], returned out of order on purpose."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        self.calls += 1
        items = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)]
        return SimpleNamespace(data=items[::-1])


def _fake_embedder(batch_size: int = 2) -> tuple[Embedder, _FakeEmbeddingsAPI]:
    api = _FakeEmbeddingsAPI()
    return Embedder(SimpleNamespace(embeddings=api), model="fake-model", batch_size=batch_size), api


def test_search_ranks_by_cosine_similarity() -> None:
    matrix = np.stack([_unit(1, 0), _unit(0, 1), _unit(1, 1)])
    index = VectorIndex([_chunk(0), _chunk(1), _chunk(2)], matrix)
    results = index.search(_unit(1, 0.1), top_k=3)
    assert [r.chunk.id for r in results] == ["doc#0", "doc#2", "doc#1"]


def test_search_applies_filters_and_top_k() -> None:
    chunks = [
        _chunk(0, "Warfarin", "interactions"),
        _chunk(1, "Apixaban", "interactions"),
        _chunk(2, "Warfarin", "contraindications"),
    ]
    index = VectorIndex(chunks, np.stack([_unit(1, 0)] * 3))
    query = _unit(1, 0)

    warfarin = index.search(query, top_k=10, chunk_filter=ChunkFilter.build(drugs=["warfarin"]))
    assert {r.chunk.id for r in warfarin} == {"doc#0", "doc#2"}

    narrow = ChunkFilter.build(drugs=["WARFARIN"], section_types=["contraindications"])
    assert [r.chunk.id for r in index.search(query, top_k=10, chunk_filter=narrow)] == ["doc#2"]

    assert len(index.search(query, top_k=1)) == 1


def test_filter_matching_nothing_returns_empty() -> None:
    index = VectorIndex([_chunk(0)], np.stack([_unit(1, 0)]))
    assert index.search(_unit(1, 0), top_k=5, chunk_filter=ChunkFilter.build(drugs=["aspirin"])) == []


def test_index_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError):
        VectorIndex([_chunk(0)], np.zeros((2, 2), dtype=np.float32))


def test_embedder_batches_keeps_order_and_normalizes() -> None:
    embedder, api = _fake_embedder(batch_size=2)
    texts = ["a", "bbb", "cc", "dddd", "e"]
    matrix = embedder.embed(texts)

    assert api.calls == 3  # 5 texts in batches of 2
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert np.allclose(matrix, np.stack([_unit(len(t), 1) for t in texts]))  # input order kept


def test_embeddings_are_cached(tmp_path) -> None:
    embedder, api = _fake_embedder()
    first = load_or_compute_embeddings(embedder, ["x", "yy"], tmp_path)
    second = load_or_compute_embeddings(embedder, ["x", "yy"], tmp_path)
    assert api.calls == 1  # second call served from the cache
    assert np.array_equal(first, second)

    load_or_compute_embeddings(embedder, ["x", "changed"], tmp_path)
    assert api.calls == 2  # different texts -> different cache key -> recomputed