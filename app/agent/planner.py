"""Planner: turns a user question into retrieval tasks with one LLM call.

Safeguards around the LLM:
- strict JSON schema whose drug and section enums come from the corpus;
- invalid values are dropped (and noted) instead of reaching the retriever;
- a deterministic guard adds a task for any corpus drug named in the question
  but missing from the plan;
- any LLM or parsing failure degrades to searching the question as-is.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from app.agent.prompts import PLANNER_SYSTEM_PROMPT
from app.retrieval import RetrievalTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubQuestion:
    sub_question: str                # readable sub-question, used later to structure the answer
    search_query: str                # short English keyword query, used for retrieval
    drug: str | None                 # strict retrieval filter
    section_types: tuple[str, ...] = ()  # retrieval preference


@dataclass(frozen=True)
class Plan:
    sub_questions: tuple[SubQuestion, ...]
    unknown_drugs: tuple[str, ...] = ()  # mentioned by the user but not in the corpus
    notes: tuple[str, ...] = ()          # what the safeguards changed (shown in traces/UI)

    def to_tasks(self) -> list[RetrievalTask]:
        return [
            RetrievalTask(query=s.search_query, drug=s.drug, section_types=s.section_types)
            for s in self.sub_questions
        ]


class Planner:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        available_drugs: Sequence[str],
        available_section_types: Sequence[str],
        max_tasks: int,
    ) -> None:
        self._client = client
        self._model = model
        self._drugs = list(available_drugs)
        self._section_types = list(available_section_types)
        self._max_tasks = max_tasks
        self._system_prompt = PLANNER_SYSTEM_PROMPT.format(
            drugs=", ".join(self._drugs),
            section_types=", ".join(self._section_types),
            max_tasks=max_tasks,
        )
        self._schema = _plan_schema(self._drugs, self._section_types)

    def plan(self, question: str) -> Plan:
        notes: list[str] = []
        try:
            sub_questions, unknown = self._parse(self._call_llm(question), notes)
        except (OpenAIError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Planner failed (%s: %s); falling back to the raw question", type(exc).__name__, exc)
            notes.append(f"planner failed ({type(exc).__name__}); searched the question as-is")
            sub_questions, unknown = [], []

        sub_questions = self._add_missing_drugs(question, sub_questions, notes)
        if not sub_questions:
            sub_questions = [SubQuestion(question, question, None)]
        return Plan(tuple(sub_questions), tuple(unknown), tuple(notes))

    def _call_llm(self, question: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": question},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "retrieval_plan", "strict": True, "schema": self._schema},
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty planner response")
        return content

    def _parse(self, raw: str, notes: list[str]) -> tuple[list[SubQuestion], list[str]]:
        data = json.loads(raw)
        result: list[SubQuestion] = []
        for item in data["sub_questions"]:
            drug = item["drug"].strip().lower() if item["drug"] else None
            if drug is not None and drug not in self._drugs:
                notes.append(f"dropped a sub-question about unknown drug {item['drug']!r}")
                continue
            sections = tuple(s for s in item["section_types"] if s in self._section_types)
            sub_question = item["sub_question"].strip()
            query = item["search_query"].strip() or sub_question
            if query:
                result.append(SubQuestion(sub_question or query, query, drug, sections))

        if len(result) > self._max_tasks:
            notes.append(f"plan truncated from {len(result)} to {self._max_tasks} tasks")
            result = result[: self._max_tasks]
        return result, [str(d) for d in data.get("unknown_drugs", [])]

    def _add_missing_drugs(
        self, question: str, sub_questions: list[SubQuestion], notes: list[str]
    ) -> list[SubQuestion]:
        """Coverage guarantee: every corpus drug named in the question gets a task.
        Runs after truncation on purpose: coverage wins over the task limit."""
        planned = {s.drug for s in sub_questions}
        for drug in self._drugs:
            if drug not in planned and re.search(rf"\b{re.escape(drug)}\b", question, re.IGNORECASE):
                notes.append(f"added a search for {drug}: named in the question but missing from the plan")
                sub_questions = [*sub_questions, SubQuestion(question, question, drug)]
        return sub_questions


def _plan_schema(drugs: Sequence[str], section_types: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sub_question": {"type": "string"},
                        "search_query": {"type": "string"},
                        "drug": {"anyOf": [{"type": "string", "enum": list(drugs)}, {"type": "null"}]},
                        "section_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(section_types)},
                        },
                    },
                    "required": ["sub_question", "search_query", "drug", "section_types"],
                    "additionalProperties": False,
                },
            },
            "unknown_drugs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sub_questions", "unknown_drugs"],
        "additionalProperties": False,
    }