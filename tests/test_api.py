"""API tests with a fake agent (no LLM calls, no corpus loading)."""

from __future__ import annotations
import json
import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from app.agent import (
    VERIFIED,
    AgentResult,
    Answer,
    Citation,
    Plan,
    Review,
    ReviewIssue,
    SubQuestion,
    ToolCallRecord,
    TraceStep,
)
from app.agent.prompts import MEDICAL_DISCLAIMER
from app.api.main import create_app

RESULT = AgentResult(
    question="What are the contraindications of warfarin?",
    plan=Plan(
        sub_questions=(SubQuestion("Contraindications?", "warfarin contraindications", "warfarin", ("contraindications",)),),
        language="English",
    ),
    evidence=(),
    tool_calls=(ToolCallRecord("lookup_glossary", {"term": "INR"}, "- Glossary > INR", ("GLOSSARY.md#1",)),),
    investigation_notes=(),
    answer=Answer(
        text="Warfarin is contraindicated in pregnancy [S1].",
        citations=(Citation("S1", "dailymed-warfarin.md", "Warfarin (DailyMed) > 4 CONTRAINDICATIONS", "https://example.org"),),
    ),
    review=Review((ReviewIssue("clinically_ambiguous", "minor", "excerpt", "explanation", "suggestion"),)),
    status=VERIFIED,
    revisions=0,
    trace=(TraceStep("plan_question", 1.0), TraceStep("finalize", 0.5)),
)


class _FakeAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.questions: list[str] = []
        self._error = error

    def ask(self, question: str) -> AgentResult:
        self.questions.append(question)
        if self._error:
            raise self._error
        return RESULT

    def stream(self, question: str):
        self.questions.append(question)
        if self._error:
            raise self._error
        yield from RESULT.trace
        yield RESULT


def _client(agent: _FakeAgent) -> TestClient:
    return TestClient(create_app(agent=agent, max_question_chars=100))


def test_health_returns_200() -> None:
    with _client(_FakeAgent()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_the_answer_and_everything_the_agent_did() -> None:
    with _client(_FakeAgent()) as client:
        response = client.post("/ask", json={"question": "What are the contraindications of warfarin?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified" and body["reviewed"] is True
    assert body["citations"][0]["file"] == "dailymed-warfarin.md"
    assert body["issues"][0]["kind"] == "clinically_ambiguous"
    assert body["plan"]["sub_questions"][0]["drug"] == "warfarin"
    assert body["tool_calls"][0] == {
        "name": "lookup_glossary", "arguments": {"term": "INR"}, "summary": "- Glossary > INR", "new_passages": 1
    }
    assert [s["node"] for s in body["trace"]] == ["plan_question", "finalize"]
    assert body["total_seconds"] == 1.5
    assert body["disclaimer"] == MEDICAL_DISCLAIMER
    assert MEDICAL_DISCLAIMER in body["answer_markdown"]


def test_the_question_is_stripped_before_reaching_the_agent() -> None:
    agent = _FakeAgent()
    with _client(agent) as client:
        client.post("/ask", json={"question": "   warfarin?  "})
    assert agent.questions == ["warfarin?"]


@pytest.mark.parametrize("question", ["", "    "])
def test_empty_question_is_rejected(question: str) -> None:
    agent = _FakeAgent()
    with _client(agent) as client:
        response = client.post("/ask", json={"question": question})
    assert response.status_code == 422
    assert agent.questions == []  # the agent is never called


def test_too_long_question_is_rejected() -> None:
    with _client(_FakeAgent()) as client:
        response = client.post("/ask", json={"question": "x" * 101})
    assert response.status_code == 422
    assert "100" in response.json()["detail"]


def test_missing_question_field_is_rejected() -> None:
    with _client(_FakeAgent()) as client:
        response = client.post("/ask", json={"query": "warfarin?"})
    assert response.status_code == 422


def test_model_provider_failure_returns_502() -> None:
    with _client(_FakeAgent(error=OpenAIError("proxy down"))) as client:
        response = client.post("/ask", json={"question": "warfarin?"})
    assert response.status_code == 502


def test_unexpected_failure_returns_500_without_leaking_details() -> None:
    with _client(_FakeAgent(error=RuntimeError("secret internal detail"))) as client:
        response = client.post("/ask", json={"question": "warfarin?"})
    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def _events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_home_page_is_served() -> None:
    with _client(_FakeAgent()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/ask/stream" in response.text


def test_stream_sends_each_step_then_the_result() -> None:
    with _client(_FakeAgent()) as client:
        response = client.post("/ask/stream", json={"question": "warfarin?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [name for name, _ in events] == ["step", "step", "result"]
    assert events[0][1] == {"node": "plan_question", "seconds": 1.0}
    assert events[-1][1]["status"] == "verified"


def test_stream_reports_a_model_failure_as_an_error_event() -> None:
    with _client(_FakeAgent(error=OpenAIError("proxy down"))) as client:
        response = client.post("/ask/stream", json={"question": "warfarin?"})
    name, data = _events(response.text)[-1]
    assert name == "error" and "language model service failed" in data["detail"]


def test_stream_rejects_an_empty_question_before_streaming() -> None:
    with _client(_FakeAgent()) as client:
        response = client.post("/ask/stream", json={"question": "  "})
    assert response.status_code == 422