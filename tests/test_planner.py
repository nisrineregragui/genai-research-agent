"""Planner tests with a fake chat client returning canned JSON (no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from openai import OpenAIError

from app.agent import Planner
from app.retrieval import RetrievalTask

DRUGS = ["apixaban", "metformin", "sertraline", "warfarin"]
SECTIONS = ["contraindications", "interactions", "warnings"]


class _FakeCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.kwargs: dict = {}

    def create(self, **kwargs) -> SimpleNamespace:
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _planner(content: str | None = None, error: Exception | None = None, max_tasks: int = 6):
    completions = _FakeCompletions(content, error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return Planner(client, "fake-model", DRUGS, SECTIONS, max_tasks), completions


def _sq(query: str, drug: str | None, sections: tuple[str, ...] = ()) -> dict:
    return {"sub_question": f"Q: {query}", "search_query": query, "drug": drug, "section_types": list(sections)}


def _response(*sub_questions: dict, unknown: tuple[str, ...] = ()) -> str:
    return json.dumps({"sub_questions": list(sub_questions), "unknown_drugs": list(unknown)})


def test_plan_is_parsed_into_retrieval_tasks() -> None:
    planner, _ = _planner(_response(
        _sq("warfarin contraindications", "warfarin", ("contraindications",)),
        _sq("apixaban contraindications", "apixaban", ("contraindications",)),
    ))
    plan = planner.plan("Compare warfarin and apixaban contraindications")

    assert plan.to_tasks() == [
        RetrievalTask("warfarin contraindications", "warfarin", ("contraindications",)),
        RetrievalTask("apixaban contraindications", "apixaban", ("contraindications",)),
    ]
    assert plan.notes == ()


def test_schema_constrains_drugs_and_sections_to_the_corpus() -> None:
    planner, completions = _planner(_response(_sq("q", "warfarin")))
    planner.plan("warfarin?")

    response_format = completions.kwargs["response_format"]["json_schema"]
    item = response_format["schema"]["properties"]["sub_questions"]["items"]["properties"]
    assert response_format["strict"] is True
    assert item["drug"]["anyOf"][0]["enum"] == DRUGS
    assert item["section_types"]["items"]["enum"] == SECTIONS


def test_guard_adds_a_drug_named_in_the_question_but_missing_from_the_plan() -> None:
    planner, _ = _planner(_response(_sq("warfarin bleeding risk", "warfarin")))
    plan = planner.plan("Is WARFARIN safe with Sertraline?")

    assert [s.drug for s in plan.sub_questions] == ["warfarin", "sertraline"]
    assert any("sertraline" in note for note in plan.notes)


def test_unknown_drug_from_the_llm_is_dropped() -> None:
    planner, _ = _planner(_response(_sq("aspirin dose", "not-a-corpus-drug")))
    plan = planner.plan("What is the usual dose?")

    # Dropped, and no corpus drug is named: fall back to one unfiltered search.
    assert plan.to_tasks() == [RetrievalTask("What is the usual dose?", None, ())]
    assert any("unknown drug" in note for note in plan.notes)


def test_unknown_section_type_is_dropped_but_the_task_is_kept() -> None:
    planner, _ = _planner(_response(_sq("warfarin dosing", "warfarin", ("dosage_tables",))))
    plan = planner.plan("warfarin dosing")
    assert plan.to_tasks() == [RetrievalTask("warfarin dosing", "warfarin", ())]


def test_llm_error_degrades_to_the_raw_question_plus_guard() -> None:
    planner, _ = _planner(error=OpenAIError("proxy unavailable"))
    question = "What are the contraindications of warfarin?"
    plan = planner.plan(question)

    assert plan.to_tasks() == [RetrievalTask(question, "warfarin", ())]
    assert any("planner failed" in note for note in plan.notes)


def test_malformed_json_degrades_gracefully() -> None:
    planner, _ = _planner("this is not json")
    plan = planner.plan("What is a DOAC?")
    assert plan.to_tasks() == [RetrievalTask("What is a DOAC?", None, ())]


def test_too_many_tasks_are_truncated() -> None:
    planner, _ = _planner(_response(_sq("a", None), _sq("b", None), _sq("c", None)), max_tasks=2)
    plan = planner.plan("general question")
    assert len(plan.sub_questions) == 2
    assert any("truncated" in note for note in plan.notes)


def test_unknown_drugs_are_reported() -> None:
    planner, _ = _planner(_response(_sq("warfarin interactions", "warfarin"), unknown=("ozempic",)))
    plan = planner.plan("Can I take warfarin with Ozempic?")
    assert plan.unknown_drugs == ("ozempic",)