"""Graph control-flow tests with fake components (no API calls)."""

from __future__ import annotations

import pytest

from app.agent import (
    UNVERIFIED,
    VERIFIED,
    Answer,
    Investigation,
    Plan,
    ResearchAgent,
    Review,
    ReviewIssue,
    SubQuestion,
    build_graph,
)

MAJOR = Review((ReviewIssue("unsupported", "major", "excerpt", "explanation", "suggestion"),))
CLEAN = Review()


class _FakePlanner:
    def plan(self, question: str) -> Plan:
        return Plan(sub_questions=(SubQuestion(question, question, None),))


class _FakeRetriever:
    def run(self, tasks, original_question=None):
        return []


class _FakeInvestigator:
    def run(self, question, plan, retrieved) -> Investigation:
        return Investigation(tuple(retrieved))


class _FakeGenerator:
    def __init__(self, fail_revision: bool = False) -> None:
        self.calls = 0
        self._fail_revision = fail_revision

    def generate(self, question, plan, retrieved, previous_draft=None, reviewer_feedback=None) -> Answer:
        self.calls += 1
        if previous_draft is not None and self._fail_revision:
            raise ValueError("model error during revision")
        return Answer(text=f"answer {self.calls}", citations=())


class _FakeCritic:
    """Returns the scripted reviews in order (the last one repeats)."""

    def __init__(self, reviews: list[Review]) -> None:
        self._reviews = reviews
        self.previous_reviews: list[Review | None] = []

    def review(self, question, plan, retrieved, answer, previous_review=None) -> Review:
        self.previous_reviews.append(previous_review)
        index = min(len(self.previous_reviews), len(self._reviews)) - 1
        return self._reviews[index]


def _ask(reviews: list[Review], max_revisions: int = 1, fail_revision: bool = False):
    generator, critic = _FakeGenerator(fail_revision), _FakeCritic(reviews)
    graph = build_graph(_FakePlanner(), _FakeRetriever(), _FakeInvestigator(), generator, critic, max_revisions)
    return ResearchAgent(graph).ask("question?"), generator, critic


def test_clean_draft_is_verified_without_revision() -> None:
    result, generator, critic = _ask([CLEAN])
    assert result.status == VERIFIED and result.revisions == 0
    assert generator.calls == 1 and len(critic.previous_reviews) == 1
    assert [s.node for s in result.trace] == [
        "plan_question", "retrieve", "investigate", "draft_answer", "review_answer", "finalize"
    ]


def test_persistent_major_issue_stops_after_one_revision_and_is_unverified() -> None:
    result, generator, critic = _ask([MAJOR])
    assert generator.calls == 2                         # draft + one revision
    assert critic.previous_reviews == [None, MAJOR]     # the revision is reviewed against the old issues
    assert result.answer.text == "answer 2"             # the shown answer is the reviewed revision
    assert result.status == UNVERIFIED and result.revisions == 1


def test_revision_that_fixes_the_issue_is_verified() -> None:
    result, _, _ = _ask([MAJOR, CLEAN])
    assert result.status == VERIFIED and result.revisions == 1


def test_max_revisions_zero_disables_revision() -> None:
    result, generator, critic = _ask([MAJOR], max_revisions=0)
    assert generator.calls == 1 and len(critic.previous_reviews) == 1
    assert result.status == UNVERIFIED


def test_failed_revision_keeps_the_reviewed_draft_and_is_unverified() -> None:
    result, _, critic = _ask([MAJOR], fail_revision=True)
    assert result.answer.text == "answer 1"
    assert len(critic.previous_reviews) == 1
    assert result.review.reviewed is False and result.status == UNVERIFIED


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError):
        ResearchAgent(None).ask("   ")