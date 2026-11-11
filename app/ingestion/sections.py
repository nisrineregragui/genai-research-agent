"""Map a document's top-level section title to a normalized section type.

These tables are domain knowledge, not configuration: FDA prescribing labels
number their sections identically for every drug (see GLOSSARY.md), while OTC
"Drug Facts" labels and MedlinePlus monographs use plain-language titles.
"""

from __future__ import annotations

import re

# Standard FDA prescribing-information section numbers.
FDA_SECTION_NUMBERS: dict[str, str] = {
    "1": "indications",
    "2": "dosage",
    "3": "dosage_forms",
    "4": "contraindications",
    "5": "warnings",
    "6": "adverse_reactions",
    "7": "interactions",
    "8": "specific_populations",
    "9": "abuse_dependence",
    "10": "overdosage",
    "11": "description",
    "12": "clinical_pharmacology",
    "13": "nonclinical_toxicology",
    "14": "clinical_studies",
    "15": "references",
    "16": "how_supplied",
    "17": "patient_counseling",
}

# Fallback for unnumbered titles. Ordered: the first matching keyword wins,
# so specific phrases come before generic ones ("warning:" before "warning").
TITLE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("contraindication", "contraindications"),
    ("warning:", "boxed_warning"),          # "WARNING: BLEEDING RISK", "IMPORTANT WARNING:"
    ("interaction", "interactions"),
    # Old-format (pre-2006) FDA labels (albuterol, ibuprofen): unnumbered headings,
    # with PRECAUTIONS subsections flattened to the top level.
    ("clinical pharmacology", "clinical_pharmacology"),
    ("pregnancy", "specific_populations"),
    ("labor and delivery", "specific_populations"),
    ("nursing mothers", "specific_populations"),
    ("pediatric use", "specific_populations"),
    ("geriatric use", "specific_populations"),
    ("information for patients", "patient_counseling"),
    ("patient information", "patient_counseling"),
    ("notice", "warnings"),                 # MedlinePlus FDA safety notice
    ("why is this medication prescribed", "indications"),
    ("uses", "indications"),                # OTC "Uses", MedlinePlus "Other uses ..."
    ("indication", "indications"),
    ("overdos", "overdosage"), 
    ("how should this medicine be used", "dosage"),
    ("directions", "dosage"),               # OTC "Directions"
    ("dosage", "dosage"),
    ("forget a dose", "dosage"),
    ("side effect", "adverse_reactions"),
    ("adverse", "adverse_reactions"),
    ("precaution", "precautions"),
    ("warning", "warnings"),                # OTC "Warnings"
    ("dietary", "diet"),
    ("storage", "storage"),
    ("brand name", "names"),
    ("other names", "names"),
)
_SECTION_NUMBER_RE = re.compile(r"^(\d+)(?:\.\d+)*\b")


def classify_section(top_level_title: str) -> str:
    """Classify by the top-level heading only.

    Deeper MedlinePlus headings are long sentences that mention other sections
    ("... or those listed in the IMPORTANT WARNING section ..."), which would
    mislead keyword matching. DailyMed subsections (7.3) belong to their parent (7) anyway.
    """
    title = top_level_title.strip()
    match = _SECTION_NUMBER_RE.match(title)
    if match and match.group(1) in FDA_SECTION_NUMBERS:
        return FDA_SECTION_NUMBERS[match.group(1)]

    lowered = title.lower()
    for keyword, section_type in TITLE_KEYWORDS:
        if keyword in lowered:
            return section_type
    return "other"