"""Evidence gathering: the agent decides which local tools to call, if any.

The deterministic plan + retrieval guarantees per-drug coverage. This step lets the
LLM inspect what was found and fill specific gaps by choosing tools itself, through
native tool calling. Nothing in the sequence of calls is hardcoded.

Bounded: at most `max_rounds` rounds of tool calls and `max_new_chunks` added passages.
Fail-safe: an LLM error stops the investigation and keeps the evidence gathered so far.
Efficiency: the agent is told when a tool returns passages it already has.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from app.agent.planner import Plan
from app.agent.prompts import INVESTIGATOR_SYSTEM_PROMPT
from app.agent.tools import ToolBox, snippet
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict
    summary: str                     # first line of the tool output, for traces and the UI
    new_chunk_ids: tuple[str, ...]   # passages this call added to the evidence


@dataclass(frozen=True)
class Investigation:
    retrieved: tuple[RetrievedChunk, ...]       # initial passages + passages added by tools
    tool_calls: tuple[ToolCallRecord, ...] = ()
    notes: tuple[str, ...] = ()


class Investigator:
    def __init__(self, client: OpenAI, model: str, toolbox: ToolBox, max_rounds: int, max_new_chunks: int) -> None:
        self._client = client
        self._model = model
        self._toolbox = toolbox
        self._max_rounds = max_rounds
        self._max_new_chunks = max_new_chunks
        self._system_prompt = INVESTIGATOR_SYSTEM_PROMPT.format(
            drugs=", ".join(toolbox.available_drugs),
            section_types=", ".join(toolbox.available_section_types),
        )

    def run(self, question: str, plan: Plan, retrieved: Sequence[RetrievedChunk]) -> Investigation:
        pool: dict[str, RetrievedChunk] = {r.chunk.id: r for r in retrieved}
        records: list[ToolCallRecord] = []
        notes: list[str] = []
        added_total = 0
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": _user_message(question, plan, retrieved)},
        ]

        for round_number in range(1, self._max_rounds + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model, messages=messages, tools=self._toolbox.schemas
                )
            except OpenAIError as exc:
                logger.warning("Investigation stopped (%s: %s)", type(exc).__name__, exc)
                notes.append(f"investigation stopped early ({type(exc).__name__}); kept the evidence gathered so far")
                break

            message = response.choices[0].message
            if not message.tool_calls:
                break  # the agent judged the evidence sufficient

            messages.append(_assistant_message(message))
            for call in message.tool_calls:
                arguments, result = self._toolbox.execute(call.function.name, call.function.arguments)
                added: list[str] = []
                already_known = 0
                for r in result.chunks:
                    if r.chunk.id in pool:
                        already_known += 1
                    elif added_total < self._max_new_chunks:
                        pool[r.chunk.id] = r
                        added.append(r.chunk.id)
                        added_total += 1
                content = result.text
                if already_known:
                    content += f"\n(Already in the evidence: {already_known} of these passages. Do not fetch them again.)"
                summary = result.text.splitlines()[0] if result.text else ""
                records.append(ToolCallRecord(call.function.name, arguments, summary, tuple(added)))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

            if added_total >= self._max_new_chunks:
                notes.append(f"evidence budget reached ({self._max_new_chunks} passages added by tools)")
                break
            if round_number == self._max_rounds:
                notes.append(f"stopped after the maximum of {self._max_rounds} tool rounds")

        return Investigation(tuple(pool.values()), tuple(records), tuple(notes))


def _user_message(question: str, plan: Plan, retrieved: Sequence[RetrievedChunk]) -> str:
    lines = [f"Question: {question}", "", "Sub-questions:"]
    lines += [f"- {s.sub_question}" for s in plan.sub_questions]
    lines += ["", f"Drugs mentioned but NOT in the corpus: {', '.join(plan.unknown_drugs) or 'none'}", ""]
    lines.append("Passages already retrieved (previews only; each is available in full to the answering step):")
    lines += [f"- {r.chunk.breadcrumb}: {snippet(r.chunk.text, 200)}" for r in retrieved]
    return "\n".join(lines)


def _assistant_message(message) -> dict:
    """Echo the assistant's tool calls back, as the chat API requires before the tool results."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ],
    }