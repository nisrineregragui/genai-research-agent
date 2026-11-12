"""ToolBox and Investigator tests (fake vectors, scripted fake LLM, no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from app.agent import Investigator, Plan, SubQuestion, ToolBox
from app.config import DEFAULT_CORPUS_DIR, DEFAULT_GLOSSARY_FILENAME
from app.ingestion import Chunk, load_glossary
from app.retrieval import RetrievedChunk, TaskRetriever, VectorIndex


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _chunk(chunk_id: str, drug: str, section_type: str) -> Chunk:
    return Chunk(
        id=chunk_id, file=f"dailymed-{drug.lower()}.md", drug=drug, source="dailymed",
        source_url=None, section_path=(chunk_id,), section_type=section_type, text=f"text of {chunk_id}",
    )


def _term(term: str, definition: str) -> Chunk:
    return Chunk(
        id=f"GLOSSARY.md#{term}", file="GLOSSARY.md", drug="", source="glossary", source_url=None,
        section_path=("Safety and monitoring", term), section_type="glossary", text=f"{term}: {definition}",
    )


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([_unit(1, 0)] * len(texts))


GLOSSARY = [
    _term("INR (International Normalized Ratio)", "a blood test monitoring warfarin's effect."),
    _term("DOAC", "direct oral anticoagulant. Apixaban, rivaroxaban."),
]


def _toolbox(max_results: int = 2) -> ToolBox:
    chunks = [
        _chunk("w-contra-1", "Warfarin", "contraindications"),
        _chunk("w-contra-2", "Warfarin", "contraindications"),
        _chunk("w-contra-3", "Warfarin", "contraindications"),
        _chunk("a-contra", "Apixaban", "contraindications"),
        _chunk("w-inter", "Warfarin", "interactions"),
    ]
    index = VectorIndex(chunks, np.stack([_unit(1, 0)] * len(chunks)))
    retriever = TaskRetriever(index, _FakeEmbedder(), top_k_per_task=3, safety_net_top_k=3, max_chunks=10)
    return ToolBox(retriever, GLOSSARY, max_results)


# ---------- glossary file ----------

def test_glossary_file_is_parsed_into_citable_entries() -> None:
    chunks = load_glossary(DEFAULT_CORPUS_DIR / DEFAULT_GLOSSARY_FILENAME)
    by_term = {c.section_path[-1]: c for c in chunks}

    assert len(chunks) >= 20
    inr = by_term["INR (International Normalized Ratio)"]
    assert inr.breadcrumb == "Glossary > Safety and monitoring > INR (International Normalized Ratio)"
    assert "inhaled" in by_term["Route"].text  # indented continuation line was joined


# ---------- tools ----------

def test_schemas_offer_the_three_tools_with_corpus_values() -> None:
    toolbox = _toolbox()
    assert [s["function"]["name"] for s in toolbox.schemas] == ["search_corpus", "get_section", "lookup_glossary"]
    assert toolbox.schemas[1]["function"]["parameters"]["properties"]["drug"]["enum"] == ["apixaban", "warfarin"]


def test_search_corpus_keeps_the_strict_drug_filter() -> None:
    _, result = _toolbox().execute("search_corpus", json.dumps({"query": "contraindications", "drug": "apixaban"}))
    assert [r.chunk.drug for r in result.chunks] == ["Apixaban"]
    assert result.chunks[0].found_by == ("tool:search_corpus",)


def test_get_section_returns_the_section_in_order_with_a_cap() -> None:
    _, result = _toolbox(max_results=2).execute(
        "get_section", json.dumps({"drug": "warfarin", "section_type": "contraindications"})
    )
    assert [r.chunk.id for r in result.chunks] == ["w-contra-1", "w-contra-2"]
    assert "Showing 2 of 3" in result.text


def test_lookup_glossary_matches_terms_aliases_and_definitions() -> None:
    toolbox = _toolbox()
    inr = "INR (International Normalized Ratio)"
    assert toolbox.lookup_glossary("inr").chunks[0].chunk.section_path[-1] == inr
    assert toolbox.lookup_glossary("International Normalized Ratio").chunks[0].chunk.section_path[-1] == inr
    assert toolbox.lookup_glossary("anticoagulant").chunks[0].chunk.section_path[-1] == "DOAC"
    assert "No glossary entry" in toolbox.lookup_glossary("xyz").text


@pytest.mark.parametrize(
    ("name", "raw_arguments"),
    [
        ("get_section", json.dumps({"drug": "naproxen", "section_type": "contraindications"})),
        ("search_corpus", "this is not json"),
        ("delete_everything", "{}"),
        ("search_corpus", json.dumps({"query": "x", "unexpected": 1})),
    ],
)
def test_invalid_calls_return_an_error_to_the_agent_instead_of_raising(name: str, raw_arguments: str) -> None:
    _, result = _toolbox().execute(name, raw_arguments)
    assert result.text.startswith("Error")
    assert result.chunks == ()


# ---------- investigation loop ----------

class _ScriptedCompletions:
    """Returns the given assistant messages in order, then 'no tool call'."""

    def __init__(self, *messages: SimpleNamespace, error: Exception | None = None) -> None:
        self._messages = list(messages)
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = self._messages.pop(0) if self._messages else STOP
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _wants(*calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(content=None, tool_calls=list(calls))


STOP = SimpleNamespace(content="The evidence is sufficient.", tool_calls=None)
PLAN = Plan(sub_questions=(SubQuestion("Warfarin contraindications?", "warfarin contraindications", "warfarin"),))
INITIAL = [RetrievedChunk(_chunk("w-inter", "Warfarin", "interactions"), 0.8, ("task",))]
GET_CONTRA = {"drug": "warfarin", "section_type": "contraindications"}


def _investigator(completions: _ScriptedCompletions, max_rounds: int = 3, max_new_chunks: int = 8) -> Investigator:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return Investigator(client, "fake-model", _toolbox(), max_rounds, max_new_chunks)


def test_agent_may_decide_that_no_tool_is_needed() -> None:
    completions = _ScriptedCompletions(STOP)
    result = _investigator(completions).run("question", PLAN, INITIAL)

    assert result.tool_calls == ()
    assert [r.chunk.id for r in result.retrieved] == ["w-inter"]
    assert len(completions.calls) == 1
    offered = [t["function"]["name"] for t in completions.calls[0]["tools"]]
    assert offered == ["search_corpus", "get_section", "lookup_glossary"]


def test_tool_results_become_sources_and_are_sent_back_to_the_agent() -> None:
    completions = _ScriptedCompletions(_wants(_tool_call("c1", "get_section", GET_CONTRA)), STOP)
    result = _investigator(completions).run("question", PLAN, INITIAL)

    assert [r.chunk.id for r in result.retrieved] == ["w-inter", "w-contra-1", "w-contra-2"]
    assert result.tool_calls[0].name == "get_section"
    assert result.tool_calls[0].new_chunk_ids == ("w-contra-1", "w-contra-2")
    assert result.retrieved[1].found_by == ("tool:get_section",)
    tool_messages = [m for m in completions.calls[-1]["messages"] if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "c1"


def test_tool_rounds_are_bounded() -> None:
    endless = [_wants(_tool_call(f"c{i}", "lookup_glossary", {"term": "INR"})) for i in range(5)]
    completions = _ScriptedCompletions(*endless)
    result = _investigator(completions, max_rounds=2).run("question", PLAN, INITIAL)

    assert len(completions.calls) == 2
    assert any("maximum" in note for note in result.notes)


def test_new_evidence_is_capped() -> None:
    completions = _ScriptedCompletions(_wants(_tool_call("c1", "get_section", GET_CONTRA)))
    result = _investigator(completions, max_new_chunks=1).run("question", PLAN, INITIAL)

    assert len(result.retrieved) == 2  # 1 initial + 1 allowed by the budget
    assert any("budget" in note for note in result.notes)


def test_llm_failure_keeps_the_evidence_gathered_so_far() -> None:
    completions = _ScriptedCompletions(error=OpenAIError("proxy unavailable"))
    result = _investigator(completions).run("question", PLAN, INITIAL)

    assert [r.chunk.id for r in result.retrieved] == ["w-inter"]
    assert any("stopped early" in note for note in result.notes)

def test_agent_is_told_when_a_tool_returns_passages_it_already_has() -> None:
    initial = [RetrievedChunk(_chunk("w-contra-1", "Warfarin", "contraindications"), 0.9, ("task",))]
    completions = _ScriptedCompletions(_wants(_tool_call("c1", "get_section", GET_CONTRA)), STOP)
    _investigator(completions).run("question", PLAN, initial)

    tool_message = next(m for m in completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "Already in the evidence: 1 of these passages" in tool_message["content"]