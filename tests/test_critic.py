"""Critic tests with a fake chat client (no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from app.agent import (
    UNVERIFIED,
    VERIFIED,
    Answer,
    Citation,
    Critic,
    Plan,
    Review,
    ReviewIssue,
    SubQuestion,
    should_revise,
    verification_status,
)
from app.ingestion import Chunk
from app.retrieval import RetrievedChunk


class _FakeCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls = 0
        self.kwargs: dict = {}

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls += 1
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _critic(*issues: dict, error: Exception | None = None):
    completions = _FakeCompletions(json.dumps({"issues": list(issues)}), error)
    return Critic(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "fake-model"), completions


def _llm_issue(kind: str, severity: str) -> dict:
    return {"kind": kind, "severity": severity, "excerpt": "below 30 mL/min",
            "explanation": "units differ", "suggestion": "state both units"}


def _issue(severity: str, kind: str = "clinically_ambiguous") -> ReviewIssue:
    return ReviewIssue(kind, severity, "excerpt", "explanation", "suggestion")


PLAN = Plan(sub_questions=(SubQuestion("Is metformin contraindicated?", "metformin eGFR", "metformin"),))
CHUNK = Chunk(
    id="dailymed-metformin.md#0", file="dailymed-metformin.md", drug="Metformin", source="dailymed",
    source_url=None, section_path=("4 CONTRAINDICATIONS",), section_type="contraindications",
    text="Severe renal impairment (eGFR below 30 mL/min/1.73 m2).",
)
RETRIEVED = [RetrievedChunk(CHUNK, 0.9, ("task",))]
CITATION = Citation("S1", CHUNK.file, CHUNK.breadcrumb, None)
CLEAN_ANSWER = Answer(text="Metformin is contraindicated below 30 mL/min [S1].", citations=(CITATION,))
INVALID_CITATION = ReviewIssue(
    "invalid_citation", "major", "[S9]", "citation [S9] does not match any provided source",
    "cite only the provided source IDs", detected_by="check",
)
ANSWER_WITH_CHECK_ISSUE = Answer(
    text="Contraindicated below 30 [S1]. Also [S9].", citations=(CITATION,), check_issues=(INVALID_CITATION,),
)


def test_prompt_contains_sources_draft_and_automatic_checks() -> None:
    critic, completions = _critic()
    critic.review("question", PLAN, RETRIEVED, ANSWER_WITH_CHECK_ISSUE)

    user_message = completions.kwargs["messages"][1]["content"]
    assert "[S1] Metformin (DailyMed) > 4 CONTRAINDICATIONS" in user_message
    assert "eGFR below 30 mL/min/1.73 m2" in user_message
    assert ANSWER_WITH_CHECK_ISSUE.text in user_message
    assert "citation [S9] does not match any provided source" in user_message


def test_major_llm_issue_requires_revision() -> None:
    critic, _ = _critic(_llm_issue("clinically_ambiguous", "major"))
    review = critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER)
    assert review.needs_revision
    assert len(review.clinical_flags) == 1
    assert review.issues[0].detected_by == "critic"


def test_minor_issues_do_not_require_revision() -> None:
    critic, _ = _critic(_llm_issue("wrong_citation", "minor"))
    review = critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER)
    assert review.reviewed and not review.needs_revision


def test_no_issue_means_reviewed_and_accepted() -> None:
    critic, _ = _critic()
    review = critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER)
    assert review.reviewed and review.issues == () and not review.needs_revision


def test_invalid_or_code_only_kinds_from_the_llm_are_ignored() -> None:
    critic, _ = _critic(
        _llm_issue("made_up_kind", "major"),
        _llm_issue("unsupported", "critical"),
        _llm_issue("invalid_citation", "major"),  # code-only kind: never accepted from the LLM
    )
    review = critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER)
    assert review.issues == ()


def test_deterministic_issue_forces_revision_even_if_the_llm_finds_nothing() -> None:
    critic, _ = _critic()  # the LLM reports no issue
    review = critic.review("question", PLAN, RETRIEVED, ANSWER_WITH_CHECK_ISSUE)
    assert review.needs_revision
    assert [(i.kind, i.detected_by) for i in review.issues] == [("invalid_citation", "check")]


def test_critic_failure_is_reported_and_keeps_deterministic_issues() -> None:
    critic, _ = _critic(error=OpenAIError("proxy unavailable"))
    review = critic.review("question", PLAN, RETRIEVED, ANSWER_WITH_CHECK_ISSUE)
    assert review.reviewed is False
    assert review.issues == (INVALID_CITATION,)
    assert verification_status(review) == UNVERIFIED


def test_no_sources_means_no_llm_call() -> None:
    critic, completions = _critic()
    review = critic.review("question", PLAN, [], CLEAN_ANSWER)
    assert completions.calls == 0 and review.reviewed


@pytest.mark.parametrize(
    ("review", "expected"),
    [
        (Review(), VERIFIED),
        (Review((_issue("minor"),)), VERIFIED),          # minor flags stay visible, still verified
        (Review((_issue("major"),)), UNVERIFIED),
        (Review(reviewed=False), UNVERIFIED),            # LLM review could not run
    ],
)
def test_verification_status_comes_from_the_last_review(review: Review, expected: str) -> None:
    assert verification_status(review) == expected


@pytest.mark.parametrize(
    ("review", "revisions_done", "max_revisions", "expected"),
    [
        (Review((_issue("major"),)), 0, 1, True),    # major issue, one revision allowed
        (Review((_issue("major"),)), 1, 1, False),   # revision used: stop, the next status is final
        (Review((_issue("major"),)), 0, 0, False),   # MAX_REVISIONS=0 disables revision
        (Review((_issue("minor"),)), 0, 1, False),   # minor issues never trigger a revision
    ],
)
def test_should_revise(review: Review, revisions_done: int, max_revisions: int, expected: bool) -> None:
    assert should_revise(review, revisions_done, max_revisions) is expected

def test_re_review_receives_the_previous_issues_and_a_first_review_does_not() -> None:
    critic, completions = _critic()
    critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER)
    assert "Previous reviewer issues" not in completions.kwargs["messages"][1]["content"]

    previous = Review((_issue("major"),))
    critic.review("question", PLAN, RETRIEVED, CLEAN_ANSWER, previous_review=previous)
    user_message = completions.kwargs["messages"][1]["content"]
    assert "Previous reviewer issues (this draft is a revision):" in user_message
    assert previous.to_feedback() in user_message


def test_answer_without_citations_is_not_sent_to_the_llm_critic() -> None:
    critic, completions = _critic()
    uncited = Answer(text="The corpus does not cover this question.", citations=())
    review = critic.review("question", PLAN, RETRIEVED, uncited)
    assert completions.calls == 0
    assert review.reviewed and not review.needs_revision