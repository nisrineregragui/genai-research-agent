"""Retrieval check on the public evaluation questions (no LLM).

Compares, at an equal chunk budget:
  - single query:    one unfiltered search with the full question (Step 5 baseline)
  - reference plan:  the TaskRetriever executing the hand-written reference tasks
and reports how many expected sections each strategy retrieves.

Run: python -m scripts.check_retrieval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, load_settings
from app.ingestion import Chunk, load_corpus
from app.llm import get_client
from app.retrieval import Embedder, RetrievalTask, TaskRetriever, VectorIndex, load_or_compute_embeddings


def _hits(chunks: list[Chunk], expected: list[dict]) -> list[dict]:
    return [e for e in expected if any(c.file == e["file"] and e["section"] in c.section_path for c in chunks)]


def _task(spec: dict) -> RetrievalTask:
    return RetrievalTask(query=spec["query"], drug=spec.get("drug"), section_types=tuple(spec.get("section_types", ())))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)  # our logs only, not the HTTP libraries

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    settings = load_settings()
    chunks = load_corpus(settings.corpus_dir, settings.chunk_max_chars)
    embedder = Embedder(get_client(), settings.embedding_model, settings.embedding_batch_size)
    index = VectorIndex(chunks, load_or_compute_embeddings(embedder, [c.embedding_text for c in chunks], settings.cache_dir))
    retriever = TaskRetriever(
        index, embedder, settings.task_top_k, settings.retrieval_top_k, settings.max_context_chunks
    )

    totals = {"single query": 0, "reference plan": 0}
    expected_total = 0
    for q in json.loads(args.questions.read_text(encoding="utf-8")):
        expected = q["expected_sections"]
        expected_total += len(expected)

        plan_results = retriever.run([_task(t) for t in q["reference_tasks"]], original_question=q["question"])
        budget = len(plan_results)  # equal budget: the baseline gets as many chunks as the plan
        single = [r.chunk for r in index.search(embedder.embed_query(q["question"]), budget)]
        plan = [r.chunk for r in plan_results]

        print(f"\n[{q['id']}] {q['question']}   (budget: {budget} chunks)")
        for mode, retrieved in (("single query", single), ("reference plan", plan)):
            hits = _hits(retrieved, expected)
            totals[mode] += len(hits)
            missing = [f"{e['file']} > {e['section']}" for e in expected if e not in hits]
            print(f"  {mode:<15} sections {len(hits)}/{len(expected)}"
                  + (f"   missing: {'; '.join(missing)}" if missing else ""))

        print("  plan results:")
        for r in plan_results:
            mark = "*" if _hits([r.chunk], expected) else " "
            print(f"    {mark} {r.score:.3f}  {r.chunk.breadcrumb}   <- {', '.join(r.found_by)}")

    print(f"\nSection recall, single query:   {totals['single query']}/{expected_total}")
    print(f"Section recall, reference plan: {totals['reference plan']}/{expected_total}")


if __name__ == "__main__":
    main()