"""Request and response models of the HTTP API.

The response exposes everything the agent did (plan, tool calls, review, timing),
so the frontend can show how the answer was produced, not just the answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.agent import AgentResult


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    id: str
    file: str
    section: str
    url: str | None


class IssueOut(BaseModel):
    kind: str
    severity: str
    excerpt: str
    explanation: str
    suggestion: str
    detected_by: str


class SubQuestionOut(BaseModel):
    sub_question: str
    search_query: str
    drug: str | None
    section_types: list[str]


class PlanOut(BaseModel):
    sub_questions: list[SubQuestionOut]
    unknown_drugs: list[str]
    language: str
    notes: list[str]


class ToolCallOut(BaseModel):
    name: str
    arguments: dict
    summary: str
    new_passages: int


class TraceStepOut(BaseModel):
    node: str
    seconds: float


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_markdown: str
    status: Literal["verified", "unverified"]
    reviewed: bool
    revisions: int
    citations: list[CitationOut]
    not_covered: list[str]
    issues: list[IssueOut]
    disclaimer: str
    plan: PlanOut
    tool_calls: list[ToolCallOut]
    investigation_notes: list[str]
    trace: list[TraceStepOut]
    total_seconds: float


def to_response(result: AgentResult) -> AskResponse:
    answer, review, plan = result.answer, result.review, result.plan
    return AskResponse(
        question=result.question,
        answer=answer.text,
        answer_markdown=answer.to_markdown(),
        status=result.status,
        reviewed=review.reviewed,
        revisions=result.revisions,
        citations=[CitationOut(id=c.source_id, file=c.file, section=c.breadcrumb, url=c.url) for c in answer.citations],
        not_covered=list(answer.not_covered),
        issues=[
            IssueOut(
                kind=i.kind, severity=i.severity, excerpt=i.excerpt,
                explanation=i.explanation, suggestion=i.suggestion, detected_by=i.detected_by,
            )
            for i in review.issues
        ],
        disclaimer=answer.disclaimer,
        plan=PlanOut(
            sub_questions=[
                SubQuestionOut(
                    sub_question=s.sub_question, search_query=s.search_query,
                    drug=s.drug, section_types=list(s.section_types),
                )
                for s in plan.sub_questions
            ],
            unknown_drugs=list(plan.unknown_drugs),
            language=plan.language,
            notes=list(plan.notes),
        ),
        tool_calls=[
            ToolCallOut(name=t.name, arguments=t.arguments, summary=t.summary, new_passages=len(t.new_chunk_ids))
            for t in result.tool_calls
        ],
        investigation_notes=list(result.investigation_notes),
        trace=[TraceStepOut(node=s.node, seconds=s.seconds) for s in result.trace],
        total_seconds=sum(s.seconds for s in result.trace),
    )