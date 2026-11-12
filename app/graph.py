"""LangGraph orchestration of the research agent.

plan_question -> retrieve -> investigate -> draft_answer -> review_answer
review_answer --(major issue and a revision left)--> revise_answer -> review_answer
review_answer --(otherwise)--> finalize

Every node is a thin call to a component tested on its own; the graph only owns the
control flow. Loop contract: every revision is followed by a review, at most
`max_revisions` revisions, and the final status comes from the LAST review.
Node names differ from state keys (a LangGraph requirement).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from app.agent.answer import Answer, AnswerGenerator
from app.agent.critic import Critic, Review, should_revise, verification_status
from app.agent.investigator import Investigator, ToolCallRecord
from app.agent.planner import Plan, Planner
from app.retrieval import RetrievedChunk, TaskRetriever


@dataclass(frozen=True)
class TraceStep:
    node: str
    seconds: float


class AgentState(TypedDict, total=False):
    question: str
    plan: Plan
    retrieved: list[RetrievedChunk]
    evidence: list[RetrievedChunk]
    tool_calls: list[ToolCallRecord]
    investigation_notes: list[str]
    answer: Answer
    review: Review
    revisions: int
    revision_failed: bool
    status: str
    trace: Annotated[list[TraceStep], operator.add]  # each node appends its own step


@dataclass(frozen=True)
class AgentResult:
    question: str
    plan: Plan
    evidence: tuple[RetrievedChunk, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    investigation_notes: tuple[str, ...]
    answer: Answer
    review: Review
    status: str
    revisions: int
    trace: tuple[TraceStep, ...]


def _timed(name: str, fn: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
    def node(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        update = fn(state)
        update["trace"] = [TraceStep(name, time.perf_counter() - start)]
        return update
    return node


def build_graph(
    planner: Planner,
    retriever: TaskRetriever,
    investigator: Investigator,
    generator: AnswerGenerator,
    critic: Critic,
    max_revisions: int,
):
    def plan_question(state: AgentState) -> dict[str, Any]:
        return {"plan": planner.plan(state["question"]), "revisions": 0}

    def retrieve(state: AgentState) -> dict[str, Any]:
        tasks = state["plan"].to_tasks()
        return {"retrieved": retriever.run(tasks, original_question=state["question"])}

    def investigate(state: AgentState) -> dict[str, Any]:
        result = investigator.run(state["question"], state["plan"], state["retrieved"])
        return {
            "evidence": list(result.retrieved),
            "tool_calls": list(result.tool_calls),
            "investigation_notes": list(result.notes),
        }

    def draft_answer(state: AgentState) -> dict[str, Any]:
        return {"answer": generator.generate(state["question"], state["plan"], state["evidence"])}

    def review_answer(state: AgentState) -> dict[str, Any]:
        # After a revision, the Critic receives the issues the revision was meant to fix.
        previous = state.get("review") if state.get("revisions", 0) > 0 else None
        review = critic.review(
            state["question"], state["plan"], state["evidence"], state["answer"], previous_review=previous
        )
        return {"review": review}

    def revise_answer(state: AgentState) -> dict[str, Any]:
        try:
            answer = generator.generate(
                state["question"], state["plan"], state["evidence"],
                previous_draft=state["answer"].text,
                reviewer_feedback=state["review"].to_feedback(),
            )
        except (OpenAIError, ValueError, KeyError, TypeError):
            # Keep the already-reviewed draft; it can no longer be called verified.
            return {"revision_failed": True, "review": Review(state["review"].issues, reviewed=False)}
        return {"answer": answer, "revisions": state["revisions"] + 1}

    def finalize(state: AgentState) -> dict[str, Any]:
        return {"status": verification_status(state["review"])}

    def after_review(state: AgentState) -> str:
        return "revise_answer" if should_revise(state["review"], state["revisions"], max_revisions) else "finalize"

    def after_revision(state: AgentState) -> str:
        return "finalize" if state.get("revision_failed") else "review_answer"

    graph = StateGraph(AgentState)
    for name, fn in (
        ("plan_question", plan_question),
        ("retrieve", retrieve),
        ("investigate", investigate),
        ("draft_answer", draft_answer),
        ("review_answer", review_answer),
        ("revise_answer", revise_answer),
        ("finalize", finalize),
    ):
        graph.add_node(name, _timed(name, fn))

    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve")
    graph.add_edge("retrieve", "investigate")
    graph.add_edge("investigate", "draft_answer")
    graph.add_edge("draft_answer", "review_answer")
    graph.add_conditional_edges(
        "review_answer", after_review, {"revise_answer": "revise_answer", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "revise_answer", after_revision, {"review_answer": "review_answer", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()


class ResearchAgent:
    """Public entry point: one question in, one fully traced result out."""

    def __init__(self, graph) -> None:
        self._graph = graph

    def ask(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("the question is empty")
        state = self._graph.invoke({"question": question, "trace": []})
        return AgentResult(
            question=state["question"],
            plan=state["plan"],
            evidence=tuple(state["evidence"]),
            tool_calls=tuple(state.get("tool_calls", ())),
            investigation_notes=tuple(state.get("investigation_notes", ())),
            answer=state["answer"],
            review=state["review"],
            status=state["status"],
            revisions=state.get("revisions", 0),
            trace=tuple(state["trace"]),
        )