"""Parse GLOSSARY.md into one citable chunk per term.

Format observed in the corpus: '## Heading' groups and '- **Term**: definition'
entries, with occasional indented continuation lines.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.models import Chunk

_ENTRY_RE = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")


def load_glossary(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"Glossary not found: {path}")

    entries: list[list[str]] = []  # [heading, term, definition]
    heading = ""
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading, current = line[3:].strip(), None
            continue
        match = _ENTRY_RE.match(line)
        if match:
            current = [heading, match.group(1).strip(), match.group(2).strip()]
            entries.append(current)
        elif current is not None and line[:1] in (" ", "\t") and line.strip():
            current[2] = f"{current[2]} {line.strip()}"  # indented continuation line
        else:
            current = None

    return [
        Chunk(
            id=f"{path.name}#{i}",
            file=path.name,
            drug="",
            source="glossary",
            source_url=None,
            section_path=(entry_heading, term),
            section_type="glossary",
            text=f"{term}: {definition}",
        )
        for i, (entry_heading, term, definition) in enumerate(entries)
    ]