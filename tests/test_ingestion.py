"""Ingestion tests: synthetic edge cases, plus checks against the real corpus
for the sections the public evaluation questions depend on."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_CORPUS_DIR
from app.ingestion import Chunk, load_corpus, parse_document, split_text
from app.ingestion.sections import classify_section


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_corpus(DEFAULT_CORPUS_DIR, DEFAULT_CHUNK_MAX_CHARS)


def _sections(chunks: list[Chunk], file: str, section_type: str) -> list[Chunk]:
    return [c for c in chunks if c.file == file and c.section_type == section_type]


# ---------- unit tests (no corpus needed) ----------

@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("4 CONTRAINDICATIONS", "contraindications"),
        ("7 DRUG INTERACTIONS", "interactions"),
        ("12 CLINICAL PHARMACOLOGY", "clinical_pharmacology"),
        ("WARNING: BLEEDING RISK", "boxed_warning"),
        ("IMPORTANT WARNING:", "boxed_warning"),
        ("Uses", "indications"),
        ("Warnings", "warnings"),
        ("Why is this medication prescribed?", "indications"),
        ("What side effects can this medication cause?", "adverse_reactions"),
        ("CLINICAL PHARMACOLOGY", "clinical_pharmacology"),
        ("Pregnancy", "specific_populations"),
        ("Nursing Mothers", "specific_populations"),
        ("Information for Patients", "patient_counseling"),
        ("PATIENT INFORMATION", "patient_counseling"),
        ("Notice:", "warnings"),
        ("What other information should I know?", "other"),
        ("OVERDOSAGE", "overdosage"),
        ("Something unexpected", "other"),
    ],
)
def test_classify_section(title: str, expected: str) -> None:
    assert classify_section(title) == expected


def test_split_text_respects_limit_and_keeps_content() -> None:
    text = "\n\n".join(["a" * 100] * 10)
    pieces = split_text(text, max_chars=250)
    assert all(len(p) <= 250 for p in pieces)
    assert sum(p.count("a") for p in pieces) == 1000


def test_split_text_keeps_table_rows_whole() -> None:
    rows = [f"Drug{i} | some value" for i in range(100)]
    pieces = split_text("\n".join(rows), max_chars=200)
    assert all(len(p) <= 200 for p in pieces)
    assert [line for p in pieces for line in p.splitlines() if line] == rows


def test_parse_document_edge_cases(tmp_path: Path) -> None:
    doc = tmp_path / "dailymed-testdrug.md"
    doc.write_text(
        "# Testdrug (DailyMed label)\n\n"
        "Source: https://example.org/testdrug\n\n"
        "## 7 DRUG INTERACTIONS\n\n"
        "### 7.1 General\n\nInteracts with things.\n\n"
        "# A stray footnote line\n\n"
        "## 4 CONTRAINDICATIONS\n\n- Pregnancy\n",
        encoding="utf-8",
    )
    result = parse_document(doc, max_chars=1000)

    # The empty parent "7 DRUG INTERACTIONS" produces no chunk of its own.
    assert [c.section_path for c in result] == [
        ("7 DRUG INTERACTIONS", "7.1 General"),
        ("4 CONTRAINDICATIONS",),
    ]
    assert "A stray footnote line" in result[0].text  # stray "# " line kept as body text
    assert [c.section_type for c in result] == ["interactions", "contraindications"]
    assert all(c.drug == "Testdrug" for c in result)
    assert all(c.source_url == "https://example.org/testdrug" for c in result)


# ---------- real corpus ----------

def test_corpus_coverage(chunks: list[Chunk]) -> None:
    files = {c.file for c in chunks}
    assert len(files) == 20
    assert "GLOSSARY.md" not in files
    assert len({c.drug for c in chunks}) == 10


def test_chunk_invariants(chunks: list[Chunk]) -> None:
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"
    for c in chunks:
        assert c.source_url, f"{c.id} has no source URL"
        assert "â€" not in c.text, f"{c.id} looks mis-decoded (not UTF-8)"
        # Over the limit is only allowed for a single unbreakable line.
        assert len(c.text) <= DEFAULT_CHUNK_MAX_CHARS or "\n" not in c.text, c.id


@pytest.mark.parametrize(
    "file",
    ["dailymed-warfarin.md", "dailymed-apixaban.md", "dailymed-metformin.md"],
)
def test_public_questions_have_contraindication_sections(chunks: list[Chunk], file: str) -> None:
    assert _sections(chunks, file, "contraindications")


def test_warfarin_contraindications_include_pregnancy(chunks: list[Chunk]) -> None:
    assert any("Pregnancy" in c.text for c in _sections(chunks, "dailymed-warfarin.md", "contraindications"))


def test_warfarin_lists_sertraline_as_bleeding_risk(chunks: list[Chunk]) -> None:
    # Public question 3 (warfarin + sertraline) depends on this section.
    assert any("sertraline" in c.text for c in _sections(chunks, "dailymed-warfarin.md", "interactions"))


def test_metformin_contraindications_mention_egfr(chunks: list[Chunk]) -> None:
    # Public question 4 (eGFR threshold) depends on this section.
    assert any("eGFR" in c.text for c in _sections(chunks, "dailymed-metformin.md", "contraindications"))


def test_atorvastatin_stray_heading_kept_as_text(chunks: list[Chunk]) -> None:
    atorvastatin = [c for c in chunks if c.file == "dailymed-atorvastatin.md"]
    assert any("prothrombin time" in c.text for c in atorvastatin)
    assert not any(t.startswith("See Section 7") for c in atorvastatin for t in c.section_path)


def test_otc_label_is_classified(chunks: list[Chunk]) -> None:
    assert _sections(chunks, "dailymed-acetaminophen.md", "indications")
    assert any("4,000 mg" in c.text for c in _sections(chunks, "dailymed-acetaminophen.md", "warnings"))


def test_only_generic_sections_are_unclassified(chunks: list[Chunk]) -> None:
    # Guards against regressions: anything new landing in "other" fails this test.
    leftovers = {c.section_path[0] for c in chunks if c.section_type == "other"}
    assert leftovers == {"What other information should I know?"}