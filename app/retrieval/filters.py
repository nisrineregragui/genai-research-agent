"""Optional search filters. An empty field means "no restriction"."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.ingestion import Chunk


@dataclass(frozen=True)
class ChunkFilter:
    drugs: frozenset[str] = frozenset()          # lower-case drug names
    section_types: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()

    @classmethod
    def build(
        cls,
        drugs: Iterable[str] | None = None,
        section_types: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
    ) -> ChunkFilter:
        return cls(
            drugs=frozenset(d.strip().lower() for d in drugs or ()),
            section_types=frozenset(section_types or ()),
            sources=frozenset(sources or ()),
        )

    def accepts(self, chunk: Chunk) -> bool:
        return (
            (not self.drugs or chunk.drug.lower() in self.drugs)
            and (not self.section_types or chunk.section_type in self.section_types)
            and (not self.sources or chunk.source in self.sources)
        )