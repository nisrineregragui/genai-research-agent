"""Ask a question end to end (plan -> retrieve -> agent tools -> answer), without the API.

  python -m scripts.ask "What are the contraindications of warfarin?"
  python -m scripts.ask      # public questions: checks that the expected files are cited
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from app.bootstrap import (
    build_answer_generator,
    build_investigator,
    build_planner,
    build_retriever,
    build_toolbox,
)
from app.config import PROJECT_ROOT, load_settings


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to answer (omit to run the public questions)")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    settings = load_settings()
    retriever = build_retriever(settings)
    planner = build_planner(settings, retriever)
    investigator = build_investigator(settings, build_toolbox(settings, retriever))
    generator = build_answer_generator(settings)
    tool_usage: Counter[str] = Counter()

    def ask(question: str):
        t0 = time.perf_counter()
        plan = planner.plan(question)
        t1 = time.perf_counter()
        retrieved = retriever.run(plan.to_tasks(), original_question=question)
        t2 = time.perf_counter()
        investigation = investigator.run(question, plan, retrieved)
        t3 = time.perf_counter()
        answer = generator.generate(question, plan, investigation.retrieved)
        t4 = time.perf_counter()

        print(f"\n=== {question}\n")
        if not investigation.tool_calls:
            print("  [agent] no tool needed: the planned retrieval was judged sufficient")
        for call in investigation.tool_calls:
            tool_usage[call.name] += 1
            arguments = ", ".join(f"{k}={v}" for k, v in call.arguments.items())
            print(f"  [tool] {call.name}({arguments}) -> +{len(call.new_chunk_ids)} passages")
        for note in investigation.notes:
            print(f"  [agent] note: {note}")
        print()
        print(answer.to_markdown())
        for warning in answer.warnings:
            print(f"\n  WARNING: {warning}")
        print(f"\n  [timing] plan {t1 - t0:.1f}s | retrieve {t2 - t1:.1f}s | tools {t3 - t2:.1f}s"
              f" | answer {t4 - t3:.1f}s | {len(investigation.retrieved)} passages")
        return answer

    if args.question:
        ask(args.question)
        return

    cited_total = expected_total = 0
    for q in json.loads(args.questions.read_text(encoding="utf-8")):
        answer = ask(q["question"])
        cited_files = {c.file for c in answer.citations}
        expected = set(q["expected_sources"])
        cited_total += len(expected & cited_files)
        expected_total += len(expected)
        print(f"  [{q['id']}] expected sources cited: {len(expected & cited_files)}/{len(expected)}")

    print(f"\nExpected sources cited: {cited_total}/{expected_total}")
    print(f"Tools chosen by the agent: {dict(tool_usage) or 'none'}")


if __name__ == "__main__":
    main()