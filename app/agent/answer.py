"""Answer generation: a cited answer built only from the retrieved passages.

Protection against unsourced claims, in three layers:
1. Prompt rules: sources only, cite every factual sentence, no transfer between drugs.
2. Deterministic checks on the output, reported as typed major ReviewIssues that force a
   revision: citations that match no source, an answer with no citation, and a cited
   sentence about a drug outside the corpus whose sources never name that drug.
3. Guaranteed by code, never left to the LLM: drugs outside the corpus are always
   reported as not covered, the disclaimer is always appended, and with no retrieved
   passage the LLM is not called at all.
The Critic adds semantic, claim-by-claim verification on top of this.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI

from app.agent.issues import ReviewIssue
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

# A number followed by a unit token, e.g. "25 mL/min", "500 mg", "4.0 mg/dL".
# The unit must start with a letter or "%", so "1 to 4" or "2.0-3.0" are ignored.
_VALUE_WITH_UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([A-Za-zµμ%][^\s,;:)\]]*)")

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
    check_issues: tuple[ReviewIssue, ...] = ()   # found by the deterministic checks
    disclaimer: str = MEDICAL_DISCLAIMER

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable messages of the deterministic checks."""
        return tuple(issue.explanation for issue in self.check_issues)

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


def number_sources(retrieved: Sequence[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    """Source IDs shared by the answer and the Critic: S1..Sn in retrieval order."""
    return {f"S{i}": r for i, r in enumerate(retrieved, start=1)}


def format_sources(sources: dict[str, RetrievedChunk]) -> str:
    lines: list[str] = []
    for sid, retrieved in sources.items():
        chunk = retrieved.chunk
        lines += [f"[{sid}] {chunk.breadcrumb} | file: {chunk.file}", chunk.text, ""]
    return "\n".join(lines)


class AnswerGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def generate(
        self,
        question: str,
        plan: Plan,
        retrieved: Sequence[RetrievedChunk],
        previous_draft: str | None = None,
        reviewer_feedback: str | None = None,
    ) -> Answer:
        """Draft an answer, or a corrected one when a previous draft and reviewer feedback are given."""
        # Guaranteed by code: drugs outside the corpus are always reported.
        not_covered = [f"The corpus contains no drug label for {drug}." for drug in plan.unknown_drugs]
        if not retrieved:
            return Answer(NO_SOURCES_MESSAGE, (), tuple(not_covered))

        sources = number_sources(retrieved)
        user_message = _user_message(question, plan, sources, previous_draft, reviewer_feedback)
        data = json.loads(self._call_llm(user_message))
        text = str(data["answer"]).strip()
        not_covered += [str(item).strip() for item in data["not_covered"] if str(item).strip()]

        cited_ids, issues = _validate_citations(text, sources)
        issues += _check_claims_about_unknown_drugs(text, plan.unknown_drugs, sources)
        issues += _check_question_values(question, text)
        citations = tuple(
            Citation(sid, sources[sid].chunk.file, sources[sid].chunk.breadcrumb, sources[sid].chunk.source_url)
            for sid in cited_ids
        )
        return Answer(text, citations, tuple(not_covered), _dedupe(issues))

    def _call_llm(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
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


def _user_message(
    question: str,
    plan: Plan,
    sources: dict[str, RetrievedChunk],
    previous_draft: str | None,
    reviewer_feedback: str | None,
) -> str:
    lines = [f"Question: {question}"]
    if plan.language:
        lines.append(f"Answer language: {plan.language}")
    lines += ["", "Sub-questions to cover:"]
    lines += [f"- {s.sub_question}" for s in plan.sub_questions]
    lines += ["", f"Drugs mentioned but NOT in the corpus: {', '.join(plan.unknown_drugs) or 'none'}", "", "Sources:"]
    lines.append(format_sources(sources))
    if previous_draft and reviewer_feedback:
        lines += ["Previous draft (to correct):", previous_draft, "", "Reviewer issues to fix:", reviewer_feedback]
    return "\n".join(lines)


def _cited_ids(text: str) -> list[str]:
    return [sid for group in _CITATION_GROUP_RE.findall(text) for sid in _CITATION_SPLIT_RE.split(group)]


def _validate_citations(text: str, sources: dict[str, RetrievedChunk]) -> tuple[list[str], list[ReviewIssue]]:
    cited: list[str] = []
    issues: list[ReviewIssue] = []
    for sid in _cited_ids(text):
        if sid not in sources:
            issues.append(ReviewIssue(
                kind="invalid_citation", severity="major", excerpt=f"[{sid}]",
                explanation=f"citation [{sid}] does not match any provided source",
                suggestion=f"cite only the provided source IDs (S1 to S{len(sources)})",
                detected_by="check",
            ))
        elif sid not in cited:
            cited.append(sid)
    if not cited:
        issues.append(ReviewIssue(
            kind="unsupported", severity="major", excerpt=text[:150],
            explanation="the answer cites no source",
            suggestion="support every factual sentence with a citation to the provided sources",
            detected_by="check",
        ))
    return cited, issues


def _check_claims_about_unknown_drugs(
    text: str, unknown_drugs: Sequence[str], sources: dict[str, RetrievedChunk]
) -> list[ReviewIssue]:
    """A cited sentence about a drug outside the corpus must cite a source naming that drug."""
    issues: list[ReviewIssue] = []
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
                issues.append(ReviewIssue(
                    kind="drug_transfer", severity="major", excerpt=sentence[:150],
                    explanation=f"a statement about {drug} cites sources that never mention it",
                    suggestion=f"only report what sources naming {drug} say, or state that the corpus does not cover it",
                    detected_by="check",
                ))
    return issues


def _values_with_units(text: str) -> dict[str, dict[str, str]]:
    """{number: {lower-cased unit: unit as written}} for every value written with a unit."""
    found: dict[str, dict[str, str]] = {}
    for number, unit in _VALUE_WITH_UNIT_RE.findall(text):
        unit = unit.rstrip(".")
        found.setdefault(number, {})[unit.lower()] = unit
    return found


def _check_question_values(question: str, text: str) -> list[ReviewIssue]:
    """A value the user gave with a unit must not reappear only with a different unit.

    Keeping the user's value and stating an interpretation alongside it is accepted;
    silently replacing the unit (e.g. "25 mL/min" -> "25 mL/min/1.73 m2") is not.
    """
    asked = _values_with_units(question)
    answered = _values_with_units(text)
    issues: list[ReviewIssue] = []
    for number, asked_units in asked.items():
        used_units = answered.get(number, {})
        if used_units and not set(used_units) & set(asked_units):
            given = f"{number} {next(iter(asked_units.values()))}"
            restated = ", ".join(f"{number} {u}" for u in used_units.values())
            issues.append(ReviewIssue(
                kind="clinically_ambiguous", severity="major", excerpt=restated,
                explanation=f"the question gives {given} but the answer only restates it as {restated}",
                suggestion=f"keep the user's value as given ({given}) and state explicitly any assumption "
                           "relating it to the unit used in the sources",
                detected_by="check",
            ))
    return issues

def _dedupe(issues: list[ReviewIssue]) -> tuple[ReviewIssue, ...]:
    """The same invalid citation repeated in the text is reported once."""
    return tuple({(i.kind, i.explanation): i for i in issues}.values())