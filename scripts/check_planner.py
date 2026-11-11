"""Run the LLM Planner on the public questions and measure retrieval with its plans.

  python -m scripts.check_planner                      # score vs. the reference plans
  python -m scripts.check_planner --question "..."     # inspect the plan for any question
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from app.agent import Plan
from app.bootstrap import build_planner, build_retriever
from app.config import PROJECT_ROOT, load_settings
from app.retrieval import RetrievalTask, RetrievedChunk


def _section_hits(results: list[RetrievedChunk], expected: list[dict]) -> list[dict]:
    return [
        e for e in expected
        if any(r.chunk.file == e["file"] and e["section"] in r.chunk.section_path for r in results)
    ]


def _print_plan(plan: Plan, seconds: float) -> None:
    print(f"  plan ({seconds:.1f}s):")
    for s in plan.sub_questions:
        scope = f"{s.drug or 'any drug'} | {', '.join(s.section_types) or 'any section'}"
        print(f"    - [{scope}] {s.search_query}")
    for note in plan.notes:
        print(f"    note: {note}")
    if plan.unknown_drugs:
        print(f"    not in corpus: {', '.join(plan.unknown_drugs)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", help="plan this question only (no scoring)")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    settings = load_settings()
    retriever = build_retriever(settings)
    planner = build_planner(settings, retriever)

    if args.question:
        start = time.perf_counter()
        plan = planner.plan(args.question)
        print(f"\n{args.question}")
        _print_plan(plan, time.perf_counter() - start)
        return

    llm_total = reference_total = expected_total = 0
    for q in json.loads(args.questions.read_text(encoding="utf-8")):
        expected = q["expected_sections"]
        expected_total += len(expected)

        start = time.perf_counter()
        plan = planner.plan(q["question"])
        seconds = time.perf_counter() - start

        llm_hits = _section_hits(retriever.run(plan.to_tasks(), original_question=q["question"]), expected)
        reference_tasks = [
            RetrievalTask(t["query"], t.get("drug"), tuple(t.get("section_types", ())))
            for t in q["reference_tasks"]
        ]
        reference_hits = _section_hits(retriever.run(reference_tasks, original_question=q["question"]), expected)
        llm_total += len(llm_hits)
        reference_total += len(reference_hits)

        print(f"\n[{q['id']}] {q['question']}")
        _print_plan(plan, seconds)
        missing = [f"{e['file']} > {e['section']}" for e in expected if e not in llm_hits]
        print(f"  sections with LLM plan: {len(llm_hits)}/{len(expected)}"
              + (f"   missing: {'; '.join(missing)}" if missing else ""))

    print(f"\nSection recall, LLM planner:    {llm_total}/{expected_total}")
    print(f"Section recall, reference plan: {reference_total}/{expected_total}")


if __name__ == "__main__":
    main()