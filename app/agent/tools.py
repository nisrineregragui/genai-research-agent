"""Local tools the agent can choose to call while gathering evidence.

Every passage a tool returns becomes a numbered, citable source, so nothing a tool
finds can reach the answer without traceability. Invalid arguments are reported back
to the agent as an error message (so it can correct itself), never raised.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.ingestion import Chunk
from app.retrieval import RetrievalTask, RetrievedChunk, TaskRetriever

# get_section and glossary results are exact lookups, not similarity hits.
EXACT_FETCH_SCORE = 1.0


@dataclass(frozen=True)
class ToolResult:
    text: str                                  # what the agent reads
    chunks: tuple[RetrievedChunk, ...] = ()    # passages that become citable sources


def snippet(text: str, limit: int = 300) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


class ToolBox:
    def __init__(self, retriever: TaskRetriever, glossary: Sequence[Chunk], max_results: int) -> None:
        self._retriever = retriever
        self._glossary = list(glossary)
        self._max_results = max_results
        self.available_drugs = retriever.available_drugs
        self.available_section_types = retriever.available_section_types
        self.schemas = _tool_schemas(self.available_drugs, self.available_section_types)
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "search_corpus": self.search_corpus,
            "get_section": self.get_section,
            "lookup_glossary": self.lookup_glossary,
        }

    def execute(self, name: str, raw_arguments: str | None) -> tuple[dict, ToolResult]:
        """Run one tool call requested by the agent. Never raises."""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except ValueError:
            return {}, ToolResult("Error: arguments are not valid JSON.")
        if not isinstance(arguments, dict):
            return {}, ToolResult("Error: arguments must be a JSON object.")
        handler = self._handlers.get(name)
        if handler is None:
            return arguments, ToolResult(f"Error: unknown tool {name!r}. Available: {', '.join(self._handlers)}.")
        try:
            return arguments, handler(**arguments)
        except (ValueError, TypeError) as exc:  # unknown drug, unexpected argument, ...
            return arguments, ToolResult(f"Error: {exc}")

    def search_corpus(
        self, query: str, drug: str | None = None, section_types: Sequence[str] | None = None
    ) -> ToolResult:
        task = RetrievalTask(query=query, drug=drug or None, section_types=tuple(section_types or ()))
        results = [RetrievedChunk(r.chunk, r.score, ("tool:search_corpus",)) for r in self._retriever.run([task])]
        return ToolResult(_describe(results, "No passage found."), tuple(results))

    def get_section(self, drug: str, section_type: str) -> ToolResult:
        name = drug.strip().lower()
        if name not in self.available_drugs:
            raise ValueError(f"unknown drug {drug!r}; available: {', '.join(self.available_drugs)}")
        if section_type not in self.available_section_types:
            raise ValueError(f"unknown section type {section_type!r}")
        matches = [c for c in self._retriever.chunks if c.drug.lower() == name and c.section_type == section_type]
        shown = [RetrievedChunk(c, EXACT_FETCH_SCORE, ("tool:get_section",)) for c in matches[: self._max_results]]
        text = _describe(shown, f"No '{section_type}' section found for {drug}.")
        if len(matches) > len(shown):
            text += f"\n(Showing {len(shown)} of {len(matches)} passages; use search_corpus to target the rest.)"
        return ToolResult(text, tuple(shown))

    def lookup_glossary(self, term: str) -> ToolResult:
        query = term.strip().lower()
        if not query:
            raise ValueError("empty term")

        def name(c: Chunk) -> str:
            return c.section_path[-1].lower()           # "inr (international normalized ratio)"

        def base(c: Chunk) -> str:
            return name(c).split("(")[0].strip()        # "inr"

        exact = [c for c in self._glossary if query in (name(c), base(c))]
        named = [c for c in self._glossary if c not in exact
                 and (query in name(c) or re.search(rf"\b{re.escape(base(c))}\b", query))]
        defined = [c for c in self._glossary if c not in exact and c not in named and query in c.text.lower()]
        hits = [
            RetrievedChunk(c, EXACT_FETCH_SCORE, ("tool:lookup_glossary",))
            for c in (exact + named + defined)[: self._max_results]
        ]
        return ToolResult(_describe(hits, f"No glossary entry for {term!r}."), tuple(hits))


def _describe(results: Sequence[RetrievedChunk], empty_message: str) -> str:
    if not results:
        return empty_message
    return "\n".join(f"- {r.chunk.breadcrumb} ({r.chunk.file}): {snippet(r.chunk.text)}" for r in results)


def _tool_schemas(drugs: Sequence[str], section_types: Sequence[str]) -> list[dict]:
    drug_enum, section_enum = list(drugs), list(section_types)
    return [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": "Targeted search over the drug documents, to find one specific missing "
                               "piece of evidence. Set drug to restrict results to one corpus drug.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "short English keyword query"},
                        "drug": {"type": "string", "enum": drug_enum, "description": "optional drug filter"},
                        "section_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": section_enum},
                            "description": "optional preferred section types",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_section",
                "description": "Fetch a complete section of one drug's documents, for example its full "
                               "contraindications list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug": {"type": "string", "enum": drug_enum},
                        "section_type": {"type": "string", "enum": section_enum},
                    },
                    "required": ["drug", "section_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_glossary",
                "description": "Definition of a pharmacology term from the corpus glossary (for example "
                               "INR or DOAC). Definitions only, not evidence about a specific drug.",
                "parameters": {
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                },
            },
        },
    ]