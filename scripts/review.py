"""Draft, review and (if needed) revise an answer, printing what the Critic finds.

TEMPORARY loop until the LangGraph assembly: it applies the validated contract
(should_revise decides, every revision is followed by a review, the status comes
from the last review).

  python -m scripts.review              # public questions
  python -m scripts.review "question"   # any question
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAIError

from app.agent import Answer, Review, should_revise, verification_status
from app.bootstrap import (
    build_answer_generator,
    build_critic,
    build_investigator,
    build_planner,
    build_retriever,
    build_toolbox,
)
from app.config import PROJECT_ROOT, load_settings


def _print_round(label: str, answer: Answer, review: Review, seconds: float) -> None:
    print(f"\n--- {label} ({seconds:.1f}s)\n")
    print(answer.text)
    print()
    if not review.reviewed:
        print("  CRITIC: the LLM review could not run")
    elif not review.issues:
        print("  CRITIC: no issue found")
    for issue in review.issues:
        print(f'  [{issue.severity} | {issue.kind} | {issue.detected_by}] "{issue.excerpt[:120]}"')
        print(f"      why: {issue.explanation}")
        print(f"      fix: {issue.suggestion}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to review (omit to run the public questions)")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval_questions.json")
    args = parser.parse_args()

    settings = load_settings()
    retriever = build_retriever(settings)
    planner = build_planner(settings, retriever)
    investigator = build_investigator(settings, build_toolbox(settings, retriever))
    generator = build_answer_generator(settings)
    critic = build_critic(settings)

    def run(question: str) -> tuple[str, int]:
        print(f"\n=== {question}")
        plan = planner.plan(question)
        evidence = investigator.run(question, plan, retriever.run(plan.to_tasks(), original_question=question)).retrieved

        t0 = time.perf_counter()
        answer = generator.generate(question, plan, evidence)
        review = critic.review(question, plan, evidence, answer)
        _print_round("draft + review 1", answer, review, time.perf_counter() - t0)

        revisions = 0
        while should_revise(review, revisions, settings.max_revisions):
            revisions += 1
            t0 = time.perf_counter()
            try:
                answer = generator.generate(
                    question, plan, evidence, previous_draft=answer.text, reviewer_feedback=review.to_feedback()
                )
            except (OpenAIError, ValueError) as exc:
                print(f"\n  revision failed ({type(exc).__name__}): keeping the previous draft, UNVERIFIED")
                review = Review(review.issues, reviewed=False)
                break
            # Every revision is reviewed, against the issues it was meant to fix.
            review = critic.review(question, plan, evidence, answer, previous_review=review)
            _print_round(f"revision {revisions} + review {revisions + 1}", answer, review, time.perf_counter() - t0)

        status = verification_status(review)
        print(f"\n  => STATUS: {status.upper()} | revisions: {revisions} | clinical flags: {len(review.clinical_flags)}")
        return status, revisions

    if args.question:
        run(args.question)
        return

    results = [run(q["question"]) for q in json.loads(args.questions.read_text(encoding="utf-8"))]
    print(f"\nFinal statuses: {dict(Counter(status for status, _ in results))}")
    print(f"Revisions performed: {sum(revisions for _, revisions in results)}")


if __name__ == "__main__":
    main()