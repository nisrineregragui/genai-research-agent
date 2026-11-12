"""Ask a question through the LangGraph agent, without the API.

  python -m scripts.ask "What are the contraindications of warfarin?"
  python -m scripts.ask      # public questions: checks that the expected files are cited
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from app.agent import AgentResult
from app.bootstrap import build_agent
from app.config import PROJECT_ROOT, load_settings


def _print_result(result: AgentResult) -> None:
    print(f"\n=== {result.question}\n")
    print("  [graph] " + " -> ".join(f"{s.node} {s.seconds:.1f}s" for s in result.trace))
    for call in result.tool_calls:
        arguments = ", ".join(f"{k}={v}" for k, v in call.arguments.items())
        print(f"  [tool] {call.name}({arguments}) -> +{len(call.new_chunk_ids)} passages")
    for note in (*result.plan.notes, *result.investigation_notes):
        print(f"  [note] {note}")
    print()
    print(result.answer.to_markdown())
    for issue in result.review.issues:
        print(f'\n  [{issue.severity} | {issue.kind} | {issue.detected_by}] "{issue.excerpt[:120]}"')
    total = sum(s.seconds for s in result.trace)
    print(f"\n  => STATUS: {result.status.upper()} | revisions: {result.revisions} | total {total:.1f}s")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to answer (omit to run the public questions)")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    agent = build_agent(load_settings())

    if args.question:
        _print_result(agent.ask(args.question))
        return

    cited_total = expected_total = 0
    statuses: Counter[str] = Counter()
    for q in json.loads(args.questions.read_text(encoding="utf-8")):
        result = agent.ask(q["question"])
        _print_result(result)
        expected = set(q["expected_sources"])
        cited = expected & {c.file for c in result.answer.citations}
        cited_total += len(cited)
        expected_total += len(expected)
        statuses[result.status] += 1
        print(f"  [{q['id']}] expected sources cited: {len(cited)}/{len(expected)}")

    print(f"\nExpected sources cited: {cited_total}/{expected_total}")
    print(f"Final statuses: {dict(statuses)}")


if __name__ == "__main__":
    main()