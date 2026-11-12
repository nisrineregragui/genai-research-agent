"""Corpus ingestion: markdown drug documents -> section-based chunks, plus the glossary."""

from app.ingestion.glossary import load_glossary
from app.ingestion.models import Chunk
from app.ingestion.parser import load_corpus, parse_document, split_text

__all__ = ["Chunk", "load_corpus", "load_glossary", "parse_document", "split_text"]