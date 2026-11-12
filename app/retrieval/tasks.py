"""Execute a plan of retrieval tasks against the vector index.

Rules (validated on the public questions before this code was written):
- The drug filter is STRICT: a task about warfarin only returns warfarin chunks,
  so one drug can never crowd the other out of a comparison.
- Section types are a PREFERENCE: each task runs a section-filtered search AND a
  complementary search on the same drug without the section filter, and alternates
  their hits. Relevant text stored under another section (e.g. interaction studies
  under Clinical Pharmacology) is therefore still reachable.
- An unfiltered search on the original question is a safety net against plan errors.
- Task results are merged round-robin (every task's best hit before any task's second
  hit), deduplicated with provenance, and capped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.ingestion import Chunk
from app.retrieval.embeddings import Embedder
from app.retrieval.filters import ChunkFilter
from app.retrieval.vector_index import SearchResult, VectorIndex

SAFETY_NET_LABEL = "original question"


@dataclass(frozen=True)
class RetrievalTask:
    query: str
    drug: str | None = None              # strict filter when set
    section_types: tuple[str, ...] = ()  # preference, never exclusive


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float                 # best similarity among the searches that found it
    found_by: tuple[str, ...]    # queries of the tasks that retrieved it (provenance)


class TaskRetriever:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        top_k_per_task: int,
        safety_net_top_k: int,
        max_chunks: int,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._top_k = top_k_per_task
        self._safety_net_top_k = safety_net_top_k
        self._max_chunks = max_chunks
        # Derived from the corpus, never hardcoded. The Planner will receive these lists.
        self.available_drugs = sorted({c.drug.lower() for c in index.chunks})
        self.available_section_types = sorted({c.section_type for c in index.chunks})
    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._index.chunks

    def run(self, tasks: Sequence[RetrievalTask], original_question: str | None = None) -> list[RetrievedChunk]:
        self._validate(tasks)
        queries = [t.query for t in tasks] + ([original_question] if original_question else [])
        if not queries:
            raise ValueError("Nothing to retrieve: no tasks and no question")
        vectors = self._embedder.embed(queries)  # one API call for the whole plan

        ranked: list[tuple[str, list[SearchResult]]] = [
            (task.query, self._search_task(task, vectors[i])) for i, task in enumerate(tasks)
        ]
        if original_question:
            ranked.append((SAFETY_NET_LABEL, self._index.search(vectors[-1], self._safety_net_top_k)))
        return self._merge_round_robin(ranked)

    def _search_task(self, task: RetrievalTask, vector: np.ndarray) -> list[SearchResult]:
        drugs = [task.drug] if task.drug else None
        whole_drug = self._index.search(vector, self._top_k, ChunkFilter.build(drugs=drugs))
        if not task.section_types:
            return whole_drug
        preferred = self._index.search(
            vector, self._top_k, ChunkFilter.build(drugs=drugs, section_types=task.section_types)
        )
        # Alternate: best preferred-section hit, best hit anywhere in the drug, ...
        return _interleave_unique([preferred, whole_drug])[: self._top_k]

    def _merge_round_robin(self, ranked: list[tuple[str, list[SearchResult]]]) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        depth = max((len(results) for _, results in ranked), default=0)
        for rank in range(depth):
            for label, results in ranked:
                if rank >= len(results):
                    continue
                hit = results[rank]
                existing = merged.get(hit.chunk.id)
                if existing is not None:
                    found_by = existing.found_by if label in existing.found_by else (*existing.found_by, label)
                    merged[hit.chunk.id] = RetrievedChunk(hit.chunk, max(existing.score, hit.score), found_by)
                elif len(merged) < self._max_chunks:
                    merged[hit.chunk.id] = RetrievedChunk(hit.chunk, hit.score, (label,))
        return list(merged.values())  # insertion order = round-robin priority

    def _validate(self, tasks: Sequence[RetrievalTask]) -> None:
        for task in tasks:
            if not task.query.strip():
                raise ValueError("Retrieval task with an empty query")
            if task.drug is not None and task.drug.strip().lower() not in self.available_drugs:
                raise ValueError(f"Unknown drug {task.drug!r}; available: {', '.join(self.available_drugs)}")
            unknown = set(task.section_types) - set(self.available_section_types)
            if unknown:
                raise ValueError(f"Unknown section types: {', '.join(sorted(unknown))}")


def _interleave_unique(lists: Sequence[list[SearchResult]]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for rank in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if rank < len(lst) and lst[rank].chunk.id not in seen:
                seen.add(lst[rank].chunk.id)
                out.append(lst[rank])
    return out