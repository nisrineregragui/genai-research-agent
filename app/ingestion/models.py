"""Data model shared by ingestion, retrieval and the agent."""

from __future__ import annotations

from dataclasses import dataclass

# Corpus sources. For drug documents the key is the file-name prefix
# ("dailymed-warfarin.md" -> "dailymed"); "glossary" is GLOSSARY.md (term definitions).
SOURCE_LABELS: dict[str, str] = {
    "dailymed": "DailyMed",        # professional FDA prescribing / OTC labels
    "medlineplus": "MedlinePlus",  # consumer-oriented monographs
    "glossary": "Glossary",        # definitions only, not evidence about a drug
}


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a section (or part of a long section) of a drug document."""

    id: str                         # unique and stable, e.g. "dailymed-warfarin.md#7"
    file: str                       # used for citations
    drug: str                       # e.g. "Warfarin"
    source: str                     # key of SOURCE_LABELS
    source_url: str | None
    section_path: tuple[str, ...]   # e.g. ("7 DRUG INTERACTIONS", "7.3 Drugs that Increase Bleeding Risk")
    section_type: str               # normalized, e.g. "interactions"
    text: str

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS[self.source]

    @property
    def breadcrumb(self) -> str:
        head = f"{self.drug} ({self.source_label})" if self.drug else self.source_label
        return " > ".join((head, *self.section_path))

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model. The breadcrumb tells the embedding which
        drug and section this is, since many sections look alike across drugs."""
        return f"{self.breadcrumb}\n\n{self.text}"