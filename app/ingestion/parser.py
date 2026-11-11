"""Turn corpus markdown files into section-based chunks.

Design decisions, each based on inspecting the real corpus:
- The first "# " line is the title. Any later "# " line is body text
  (dailymed-atorvastatin.md has a table footnote that starts with "#").
- One chunk per section with body text (parent headings are often empty).
- Sections longer than `max_chars` are split on blank lines, so table rows stay together.
- Files are read as UTF-8 explicitly (Windows would otherwise default to cp1252).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.models import SOURCE_LABELS, Chunk
from app.ingestion.sections import classify_section

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
_SOURCE_URL_RE = re.compile(r"^Source:\s*(\S+)")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_CORPUS_SUFFIXES = {".md", ".txt"}


def load_corpus(corpus_dir: Path, max_chars: int) -> list[Chunk]:
    """Parse every drug document in `corpus_dir`. Non-drug files (GLOSSARY.md) are skipped."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() in _CORPUS_SUFFIXES and _source_of(path) is not None:
            chunks.extend(parse_document(path, max_chars))

    if not chunks:
        raise ValueError(f"No drug documents found in {corpus_dir}")
    return chunks


def parse_document(path: Path, max_chars: int) -> list[Chunk]:
    source = _source_of(path)
    if source is None:
        raise ValueError(f"Not a drug document (expected '<source>-<drug>' file name): {path.name}")

    lines = path.read_text(encoding="utf-8").splitlines()
    drug = _drug_name(lines, path)
    source_url = next((m.group(1) for line in lines if (m := _SOURCE_URL_RE.match(line))), None)

    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (heading level, title) of the current section path
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        body.clear()
        if not text or not stack:
            return  # empty parent heading, or the preamble (title + Source line)
        section_path = tuple(title for _, title in stack)
        for piece in split_text(text, max_chars):
            chunks.append(
                Chunk(
                    id=f"{path.name}#{len(chunks)}",
                    file=path.name,
                    drug=drug,
                    source=source,
                    source_url=source_url,
                    section_path=section_path,
                    section_type=classify_section(section_path[0]),
                    text=piece,
                )
            )

    title_seen = False
    for line in lines:
        if not title_seen and line.startswith("# "):
            title_seen = True
            continue
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
        else:
            body.append(line)
    flush()
    return chunks


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into pieces of at most `max_chars`, preferring paragraph boundaries.

    Paragraphs are packed greedily. A paragraph that is itself too long (e.g. a large
    table with no blank lines) is packed line by line. A single line longer than
    `max_chars` is kept whole rather than cut mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    blocks: list[str] = []
    for paragraph in _PARAGRAPH_BREAK_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            blocks.append(paragraph)
        else:
            blocks.extend(_pack(paragraph.splitlines(), "\n", max_chars))
    return _pack(blocks, "\n\n", max_chars)


def _pack(parts: list[str], separator: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = part
    if current:
        pieces.append(current)
    return pieces


def _source_of(path: Path) -> str | None:
    """'dailymed-warfarin.md' -> 'dailymed'. None for anything else, such as GLOSSARY.md."""
    if "-" not in path.stem:
        return None
    prefix = path.stem.split("-", 1)[0].lower()
    return prefix if prefix in SOURCE_LABELS else None


def _drug_name(lines: list[str], path: Path) -> str:
    """'# Warfarin (DailyMed label)' -> 'Warfarin'. Falls back to the file name."""
    for line in lines:
        if line.startswith("# "):
            return line[2:].split("(", 1)[0].strip()
    return path.stem.split("-", 1)[1].replace("-", " ").title()