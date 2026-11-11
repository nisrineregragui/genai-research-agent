"""TaskRetriever tests with hand-made vectors and a fake embedder (no API calls)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ingestion import Chunk
from app.retrieval import SAFETY_NET_LABEL, RetrievalTask, TaskRetriever, VectorIndex


def _chunk(chunk_id: str, drug: str, section_type: str) -> Chunk:
    return Chunk(
        id=chunk_id, file=f"dailymed-{drug.lower()}.md", drug=drug, source="dailymed",
        source_url=None, section_path=("Section",), section_type=section_type, text=chunk_id,
    )


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeEmbedder:
    """Returns a predefined vector per text and counts API-like calls."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vectors[t] for t in texts])


def _retriever(entries, queries, top_k=3, safety_top_k=3, max_chunks=12):
    chunks = [_chunk(cid, drug, section) for cid, drug, section, _ in entries]
    index = VectorIndex(chunks, np.stack([vec for *_, vec in entries]))
    embedder = _FakeEmbedder(queries)
    return TaskRetriever(index, embedder, top_k, safety_top_k, max_chunks), embedder


def test_drug_filter_is_strict_and_both_drugs_are_covered() -> None:
    # Every apixaban chunk is a better match than the warfarin one: without the
    # strict filter, apixaban would fill all the slots (the q2 failure).
    entries = [
        ("w-contra", "Warfarin", "contraindications", _unit(0.8, 0.6)),
        ("a-contra", "Apixaban", "contraindications", _unit(1, 0)),
        ("a-pk", "Apixaban", "clinical_pharmacology", _unit(1, 0.05)),
    ]
    queries = {"warfarin contra": _unit(1, 0), "apixaban contra": _unit(1, 0)}
    retriever, _ = _retriever(entries, queries, top_k=1)

    results = retriever.run([
        RetrievalTask("warfarin contra", drug="warfarin"),
        RetrievalTask("apixaban contra", drug="apixaban"),
    ])

    assert {r.chunk.drug for r in results} == {"Warfarin", "Apixaban"}
    for r in results:
        if "warfarin contra" in r.found_by:
            assert r.chunk.drug == "Warfarin"


def test_section_is_a_preference_not_a_filter() -> None:
    # The query matches a pharmacology chunk best; the contraindications chunk is preferred
    # by the task. Both must be returned: preferred section first, then the best hit
    # anywhere in the drug. An unrelated chunk is not returned.
    entries = [
        ("w-pk", "Warfarin", "clinical_pharmacology", _unit(0, 1, 0)),
        ("w-contra", "Warfarin", "contraindications", _unit(0.6, 0.8, 0)),
        ("w-other", "Warfarin", "storage", _unit(0, 0, 1)),
    ]
    retriever, _ = _retriever(entries, {"q": _unit(0, 1, 0)}, top_k=2)

    results = retriever.run([RetrievalTask("q", drug="warfarin", section_types=("contraindications",))])

    assert [r.chunk.id for r in results] == ["w-contra", "w-pk"]


def test_safety_net_searches_the_original_question_without_filters() -> None:
    entries = [
        ("w-inter", "Warfarin", "interactions", _unit(0, 1, 0)),
        ("s-bleed", "Sertraline", "warnings", _unit(0, 0, 1)),
    ]
    queries = {"warfarin task": _unit(0, 1, 0), "full question": _unit(0, 0, 1)}
    retriever, _ = _retriever(entries, queries, top_k=1, safety_top_k=1)

    results = retriever.run([RetrievalTask("warfarin task", drug="warfarin")], original_question="full question")

    by_id = {r.chunk.id: r for r in results}
    assert by_id["w-inter"].found_by == ("warfarin task",)
    assert by_id["s-bleed"].found_by == (SAFETY_NET_LABEL,)


def test_duplicates_are_merged_with_provenance() -> None:
    entries = [("w-contra", "Warfarin", "contraindications", _unit(1, 0))]
    queries = {"first": _unit(1, 0), "second": _unit(1, 0)}
    retriever, _ = _retriever(entries, queries)

    results = retriever.run([RetrievalTask("first", drug="warfarin"), RetrievalTask("second", drug="warfarin")])

    assert len(results) == 1
    assert results[0].found_by == ("first", "second")


def test_round_robin_keeps_every_task_top_hit_under_the_cap() -> None:
    # Warfarin hits all score higher than apixaban hits; with a cap of 2,
    # a global sort would keep two warfarin chunks. Round-robin keeps one of each.
    entries = [
        ("w1", "Warfarin", "contraindications", _unit(1, 0)),
        ("w2", "Warfarin", "warnings", _unit(0.99, 0.14)),
        ("a1", "Apixaban", "contraindications", _unit(0.6, 0.8)),
        ("a2", "Apixaban", "warnings", _unit(0.5, 0.87)),
    ]
    queries = {"w": _unit(1, 0), "a": _unit(1, 0)}
    retriever, _ = _retriever(entries, queries, top_k=2, max_chunks=2)

    results = retriever.run([RetrievalTask("w", drug="warfarin"), RetrievalTask("a", drug="apixaban")])

    assert [r.chunk.id for r in results] == ["w1", "a1"]


def test_whole_plan_is_embedded_in_one_call() -> None:
    entries = [("w", "Warfarin", "contraindications", _unit(1, 0))]
    queries = {"t1": _unit(1, 0), "t2": _unit(1, 0), "question": _unit(1, 0)}
    retriever, embedder = _retriever(entries, queries)

    retriever.run([RetrievalTask("t1", drug="warfarin"), RetrievalTask("t2")], original_question="question")

    assert embedder.calls == 1


@pytest.mark.parametrize(
    "task",
    [
        RetrievalTask("q", drug="ibuprofen-xr"),                  # drug not in the corpus
        RetrievalTask("q", section_types=("not_a_section",)),     # unknown section type
        RetrievalTask("   ", drug="warfarin"),                    # empty query
    ],
)
def test_invalid_tasks_are_rejected(task: RetrievalTask) -> None:
    entries = [("w", "Warfarin", "contraindications", _unit(1, 0))]
    retriever, _ = _retriever(entries, {"q": _unit(1, 0)})
    with pytest.raises(ValueError):
        retriever.run([task])