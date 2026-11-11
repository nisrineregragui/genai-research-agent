"""Dense-retrieval check on the public evaluation questions (no agent yet).

Shows what plain vector search returns for each question and whether the
expected source files appear in the top-k results.

Run: python -m scripts.check_retrieval [--top-k N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, load_settings
from app.ingestion import load_corpus
from app.llm import get_client
from app.retrieval import Embedder, VectorIndex, load_or_compute_embeddings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, help="results per question (default: RETRIEVAL_TOP_K)")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    settings = load_settings()
    top_k = args.top_k or settings.retrieval_top_k

    chunks = load_corpus(settings.corpus_dir, settings.chunk_max_chars)
    embedder = Embedder(get_client(), settings.embedding_model, settings.embedding_batch_size)
    matrix = load_or_compute_embeddings(embedder, [c.embedding_text for c in chunks], settings.cache_dir)
    index = VectorIndex(chunks, matrix)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    found_total = expected_total = 0
    for q in questions:
        expected = set(q["expected_sources"])
        results = index.search(embedder.embed_query(q["question"]), top_k)
        retrieved = {r.chunk.file for r in results}
        found = expected & retrieved
        found_total += len(found)
        expected_total += len(expected)

        print(f"\n[{q['id']}] {q['question']}")
        for rank, r in enumerate(results, start=1):
            mark = "*" if r.chunk.file in expected else " "
            print(f"  {rank}. {mark} {r.score:.3f}  {r.chunk.breadcrumb}")
        missing = sorted(expected - retrieved)
        print(f"  expected files found: {len(found)}/{len(expected)}"
              + (f"  (missing: {', '.join(missing)})" if missing else ""))

    print(f"\nSource recall@{top_k}: {found_total}/{expected_total}")


if __name__ == "__main__":
    main()