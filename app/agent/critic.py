"""Critic: verifies a draft answer against its sources before it is shown.

- Merges the deterministic check issues (from the answer) with the LLM's issues, so a
  problem found by code forces a revision even if the LLM misses it.
- Judges support only against the provided sources (same S1..Sn numbering as the answer).
- Closed taxonomy; "invalid_citation" is code-only and never accepted from the LLM.
- Whether to revise, and the final status, are decided by code, not by an LLM verdict.
- If the LLM review cannot run, the review is marked as not reviewed, never passed silently.

Loop contract (implemented by the graph): every revision is followed by a new review, at
most `max_revisions` revisions, and the shown status comes from the LAST review.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from app.agent.answer import Answer, format_sources, number_sources
from app.agent.issues import CRITIC_KINDS, SEVERITIES, ReviewIssue
from app.agent.planner import Plan
from app.agent.prompts import CRITIC_SYSTEM_PROMPT
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

VERIFIED = "verified"
UNVERIFIED = "unverified"

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(CRITIC_KINDS)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "excerpt": {"type": "string"},
                    "explanation": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["kind", "severity", "excerpt", "explanation", "suggestion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Review:
    issues: tuple[ReviewIssue, ...] = ()
    reviewed: bool = True  # False when the LLM review could not run: the UI must say so

    @property
    def needs_revision(self) -> bool:
        """Decided by code from the issues, never by a separate LLM verdict."""
        return any(issue.severity == "major" for issue in self.issues)

    @property
    def clinical_flags(self) -> tuple[ReviewIssue, ...]:
        return tuple(issue for issue in self.issues if issue.kind == "clinically_ambiguous")

    def to_feedback(self) -> str:
        """Issues formatted for the AnswerGenerator's revision prompt."""
        return "\n".join(
            f'- [{i.severity} {i.kind}] "{i.excerpt}": {i.explanation} Fix: {i.suggestion}'
            for i in self.issues
        )


def should_revise(review: Review, revisions_done: int, max_revisions: int) -> bool:
    """Revise only for a major issue, and only while revisions remain."""
    return review.needs_revision and revisions_done < max_revisions


def verification_status(review: Review) -> str:
    """Status of the answer actually shown, computed from its LAST review."""
    return VERIFIED if review.reviewed and not review.needs_revision else UNVERIFIED


class Critic:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def review(
        self,
        question: str,
        plan: Plan,
        retrieved: Sequence[RetrievedChunk],
        answer: Answer,
        previous_review: Review | None = None,
    ) -> Review:
        """Review a draft. When `previous_review` is given, the draft is a revision of a
        reviewed draft: the Critic checks those issues first, so the target cannot move."""
        deterministic = answer.check_issues
        if not retrieved or not answer.citations:
            # Nothing was claimed from the sources, so there is nothing for the LLM to verify.
            # An answer that should cite but does not still carries the deterministic
            # "cites no source" issue, so it is still revised.
            return Review(deterministic)

        sources = number_sources(retrieved)
        try:
            user_message = _user_message(question, plan, sources, answer, previous_review)
            data = json.loads(self._call_llm(user_message))
            parsed = [_parse_issue(item) for item in data["issues"]]
        except (OpenAIError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Critic failed (%s: %s); answer left unreviewed", type(exc).__name__, exc)
            return Review(deterministic, reviewed=False)
        return Review(deterministic + tuple(issue for issue in parsed if issue is not None))

    def _call_llm(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer_review", "strict": True, "schema": _REVIEW_SCHEMA},
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty review from the model")
        return content


def _parse_issue(item: dict) -> ReviewIssue | None:
    if item["kind"] not in CRITIC_KINDS or item["severity"] not in SEVERITIES:
        return None  # defensive: also rejects "invalid_citation", which only code may report
    return ReviewIssue(
        kind=item["kind"],
        severity=item["severity"],
        excerpt=str(item["excerpt"]).strip(),
        explanation=str(item["explanation"]).strip(),
        suggestion=str(item["suggestion"]).strip(),
        detected_by="critic",
    )


def _user_message(
    question: str,
    plan: Plan,
    sources: dict[str, RetrievedChunk],
    answer: Answer,
    previous_review: Review | None,
) -> str:
    lines = [
        f"Question: {question}",
        f"Drugs mentioned but NOT in the corpus: {', '.join(plan.unknown_drugs) or 'none'}",
        "",
        "Sources:",
        format_sources(sources),
        "Draft answer:",
        answer.text,
    ]
    if answer.not_covered:
        lines += ["", "Draft 'not covered' list:", *(f"- {item}" for item in answer.not_covered)]
    if answer.warnings:
        lines += ["", "Automatic checks already flagged:", *(f"- {w}" for w in answer.warnings)]
    if previous_review is not None and previous_review.issues:
        lines += ["", "Previous reviewer issues (this draft is a revision):", previous_review.to_feedback()]
    return "\n".join(lines)