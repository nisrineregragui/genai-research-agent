"""Summarize how the corpus is chunked, and optionally show matching chunks.

Examples (from the project root):
  python -m scripts.inspect_corpus
  python -m scripts.inspect_corpus --drug warfarin --section interactions
"""

from __future__ import annotations

import argparse
from collections import Counter

import argparse
import sys
from collections import Counter

from app.config import load_settings
from app.ingestion import load_corpus


def main() -> None:
    # Piped output on Windows defaults to cp1252, which cannot encode characters like "β".
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", help="show chunks for this drug (case-insensitive)")
    parser.add_argument("--section", help="show chunks of this section type, e.g. contraindications")
    args = parser.parse_args()

    settings = load_settings()
    chunks = load_corpus(settings.corpus_dir, settings.chunk_max_chars)

    print(f"Corpus: {settings.corpus_dir}")
    print(
        f"{len(chunks)} chunks from {len({c.file for c in chunks})} files, "
        f"{len({c.drug for c in chunks})} drugs (max {settings.chunk_max_chars} chars)\n"
    )
    print("Chunks per section type:")
    for section_type, count in Counter(c.section_type for c in chunks).most_common():
        print(f"  {section_type:<24}{count}")
    largest = max(chunks, key=lambda c: len(c.text))
    print(f"\nLargest chunk: {len(largest.text)} chars  [{largest.breadcrumb}]")

    if args.drug or args.section:
        matches = [
            c for c in chunks
            if (not args.drug or c.drug.lower() == args.drug.lower())
            and (not args.section or c.section_type == args.section)
        ]
        print(f"\n{len(matches)} matching chunks:")
        for c in matches:
            print(f"\n--- {c.id}  [{c.breadcrumb}]  ({len(c.text)} chars)")
            print(c.text[:300] + ("..." if len(c.text) > 300 else ""))


if __name__ == "__main__":
    main()