"""Review issues shared by the deterministic checks (answer.py) and the Critic.

Every problem found in a draft, whoever found it, is a ReviewIssue, so a single
rule decides whether to revise: any major issue.
"""

from __future__ import annotations

from dataclasses import dataclass

# Detected by code only: a citation that matches no provided source. Never offered to the LLM.
CODE_ONLY_KINDS = ("invalid_citation",)
# Kinds the Critic (LLM) may report.
CRITIC_KINDS = ("unsupported", "wrong_citation", "drug_transfer", "clinically_ambiguous", "incomplete")
ISSUE_KINDS = CODE_ONLY_KINDS + CRITIC_KINDS
SEVERITIES = ("major", "minor")


@dataclass(frozen=True)
class ReviewIssue:
    kind: str
    severity: str
    excerpt: str
    explanation: str
    suggestion: str
    detected_by: str = "critic"   # "check" (deterministic code) or "critic" (LLM)