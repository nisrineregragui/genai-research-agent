"""Answer generation: a cited answer built only from the retrieved passages.

Protection against unsourced claims, in three layers:
1. Prompt rules: sources only, cite every factual sentence, no transfer between drugs.
2. Deterministic checks on the output: every citation must match a provided source,
   the answer must cite something, and a cited sentence about a drug that is not in
   the corpus must cite at least one source that actually names that drug.
3. Guaranteed by code, never left to the LLM: drugs outside the corpus are always
   reported as not covered, the disclaimer is always appended, and with no retrieved
   passage the LLM is not called at all.
The Critic (next step) adds semantic, claim-by-claim verification on top of this.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI

from app.agent.planner import Plan
from app.agent.prompts import ANSWER_SYSTEM_PROMPT, MEDICAL_DISCLAIMER
from app.retrieval import RetrievedChunk

NO_SOURCES_MESSAGE = (
    "The corpus contains no passage relevant to this question, so no answer can be "
    "given from the provided sources."
)

_CITATION_GROUP_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")  # [S1] or [S1, S3]
_CITATION_SPLIT_RE = re.compile(r"\s*[,;]\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# "...bleeding. [S1]" -> "...bleeding [S1]." so the citations stay with their sentence.
_TRAILING_CITATIONS_RE = re.compile(r"([.!?])((?:\s*\[S\d+(?:\s*[,;]\s*S\d+)*\])+)")

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "not_covered": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "not_covered"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Citation:
    source_id: str      # "S1", as used in the answer text
    file: str
    breadcrumb: str
    url: str | None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    not_covered: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()    # deterministic check failures, shown to the user and the Critic
    disclaimer: str = MEDICAL_DISCLAIMER

    def to_markdown(self) -> str:
        parts = [self.text]
        if self.not_covered:
            parts.append("**Not covered by the provided sources:**\n" + "\n".join(f"- {item}" for item in self.not_covered))
        if self.citations:
            parts.append("**Sources:**\n" + "\n".join(
                f"- [{c.source_id}] {c.breadcrumb} ({c.file})" for c in self.citations
            ))
        parts.append(f"_{self.disclaimer}_")
        return "\n\n".join(parts)


class AnswerGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, question: str, plan: Plan, retrieved: Sequence[RetrievedChunk]) -> Answer:
        # Guaranteed by code: drugs outside the corpus are always reported.
        not_covered = [f"The corpus contains no drug label for {drug}." for drug in plan.unknown_drugs]
        if not retrieved:
            return Answer(NO_SOURCES_MESSAGE, (), tuple(not_covered))

        sources = {f"S{i}": r for i, r in enumerate(retrieved, start=1)}
        data = json.loads(self._call_llm(question, plan, sources))
        text = str(data["answer"]).strip()
        not_covered += [str(item).strip() for item in data["not_covered"] if str(item).strip()]

        cited_ids, warnings = _validate_citations(text, sources)
        warnings += _check_claims_about_unknown_drugs(text, plan.unknown_drugs, sources)
        citations = tuple(
            Citation(sid, sources[sid].chunk.file, sources[sid].chunk.breadcrumb, sources[sid].chunk.source_url)
            for sid in cited_ids
        )
        return Answer(text, citations, tuple(not_covered), tuple(dict.fromkeys(warnings)))

    def _call_llm(self, question: str, plan: Plan, sources: dict[str, RetrievedChunk]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(question, plan, sources)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "cited_answer", "strict": True, "schema": _ANSWER_SCHEMA},
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty answer from the model")
        return content


def _user_message(question: str, plan: Plan, sources: dict[str, RetrievedChunk]) -> str:
    lines = [f"Question: {question}", "", "Sub-questions to cover:"]
    lines += [f"- {s.sub_question}" for s in plan.sub_questions]
    lines += ["", f"Drugs mentioned but NOT in the corpus: {', '.join(plan.unknown_drugs) or 'none'}", "", "Sources:"]
    for sid, retrieved in sources.items():
        chunk = retrieved.chunk
        lines += [f"[{sid}] {chunk.breadcrumb} | file: {chunk.file}", chunk.text, ""]
    return "\n".join(lines)


def _cited_ids(text: str) -> list[str]:
    return [sid for group in _CITATION_GROUP_RE.findall(text) for sid in _CITATION_SPLIT_RE.split(group)]


def _validate_citations(text: str, sources: dict[str, RetrievedChunk]) -> tuple[list[str], list[str]]:
    cited: list[str] = []
    warnings: list[str] = []
    for sid in _cited_ids(text):
        if sid not in sources:
            warnings.append(f"citation [{sid}] does not match any provided source")
        elif sid not in cited:
            cited.append(sid)
    if not cited:
        warnings.append("the answer cites no source")
    return cited, warnings


def _check_claims_about_unknown_drugs(
    text: str, unknown_drugs: Sequence[str], sources: dict[str, RetrievedChunk]
) -> list[str]:
    """A cited sentence about a drug outside the corpus must cite a source naming that drug."""
    warnings: list[str] = []
    # The model sometimes puts citations after the final punctuation; move them back
    # before splitting, otherwise they would be attached to the next sentence.
    normalized = _TRAILING_CITATIONS_RE.sub(lambda m: f"{m.group(2)}{m.group(1)}", text)
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        ids = [sid for sid in _cited_ids(sentence) if sid in sources]
        if not ids:
            continue  # e.g. "The corpus has no label for naproxen." needs no citation
        for drug in unknown_drugs:
            pattern = rf"\b{re.escape(drug)}\b"
            if re.search(pattern, sentence, re.IGNORECASE) and not any(
                re.search(pattern, sources[sid].chunk.text, re.IGNORECASE) for sid in ids
            ):
                warnings.append(f"a statement about {drug} cites sources that never mention it: {sentence[:150]}")
    return warnings