"""AnswerGenerator tests with a fake chat client (no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent import AnswerGenerator, Plan, SubQuestion
from app.agent.answer import NOT_COVERED_MESSAGE
from app.agent.prompts import MEDICAL_DISCLAIMER
from app.ingestion import Chunk
from app.retrieval import RetrievedChunk


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0
        self.kwargs: dict = {}

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _generator(answer: str, not_covered: tuple[str, ...] = ()):
    completions = _FakeCompletions(json.dumps({"answer": answer, "not_covered": list(not_covered)}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return AnswerGenerator(client, "fake-model"), completions


def _retrieved(drug: str, section: str, text: str) -> RetrievedChunk:
    chunk = Chunk(
        id=f"dailymed-{drug.lower()}.md#0", file=f"dailymed-{drug.lower()}.md", drug=drug, source="dailymed",
        source_url="https://example.org", section_path=(section,), section_type="other", text=text,
    )
    return RetrievedChunk(chunk, 0.9, ("task",))


PLAN = Plan(sub_questions=(SubQuestion("What are the contraindications?", "contraindications", "warfarin"),))
WARFARIN = _retrieved("Warfarin", "4 CONTRAINDICATIONS", "Warfarin is contraindicated in pregnancy.")
TABLE = _retrieved("Warfarin", "7.3 Drugs that Increase Bleeding Risk", "NSAIDs: ibuprofen, naproxen, celecoxib")
IBUPROFEN = _retrieved("Ibuprofen", "WARNINGS", "Ibuprofen can cause gastrointestinal bleeding.")


def test_sources_are_numbered_with_their_breadcrumb_in_the_prompt() -> None:
    generator, completions = _generator("Contraindicated in pregnancy [S1].")
    generator.generate("question", PLAN, [WARFARIN])

    user_message = completions.kwargs["messages"][1]["content"]
    assert "[S1] Warfarin (DailyMed) > 4 CONTRAINDICATIONS" in user_message
    assert "Warfarin is contraindicated in pregnancy." in user_message


def test_citations_are_mapped_to_sources_in_order_of_first_use() -> None:
    generator, _ = _generator("Bleeding risk [S2]. Pregnancy [S1, S2].")
    answer = generator.generate("question", PLAN, [WARFARIN, TABLE])

    assert [c.source_id for c in answer.citations] == ["S2", "S1"]
    assert answer.citations[1].file == "dailymed-warfarin.md"
    assert answer.warnings == ()


def test_citation_to_a_missing_source_is_flagged() -> None:
    generator, _ = _generator("Contraindicated in pregnancy [S1]. Also something [S7].")
    answer = generator.generate("question", PLAN, [WARFARIN])
    assert any("[S7]" in w for w in answer.warnings)


def test_answer_without_citation_is_flagged() -> None:
    generator, _ = _generator("Warfarin is contraindicated in pregnancy.")
    answer = generator.generate("question", PLAN, [WARFARIN])
    assert "the answer cites no source" in answer.warnings


def test_drugs_outside_the_corpus_are_always_reported() -> None:
    plan = Plan(PLAN.sub_questions, unknown_drugs=("naproxen",))
    generator, _ = _generator("Contraindicated in pregnancy [S1].")  # the LLM forgot to mention naproxen
    answer = generator.generate("question", plan, [WARFARIN])
    assert "The corpus contains no drug label for naproxen." in answer.not_covered


def test_claim_about_unknown_drug_citing_a_source_that_names_it_is_accepted() -> None:
    plan = Plan(PLAN.sub_questions, unknown_drugs=("naproxen",))
    generator, _ = _generator("The warfarin label lists naproxen among drugs that increase bleeding risk [S1].")
    answer = generator.generate("question", plan, [TABLE])
    assert answer.warnings == ()


def test_claim_about_unknown_drug_transferred_from_another_drug_is_flagged() -> None:
    plan = Plan(PLAN.sub_questions, unknown_drugs=("naproxen",))
    generator, _ = _generator("Naproxen can cause gastrointestinal bleeding [S1].")
    answer = generator.generate("question", plan, [IBUPROFEN])
    assert any("naproxen" in w and "never mention it" in w for w in answer.warnings)


def test_no_retrieved_passage_means_no_llm_call() -> None:
    generator, completions = _generator("unused")
    plan = Plan(PLAN.sub_questions, unknown_drugs=("ozempic",))
    answer = generator.generate("question", plan, [])

    assert completions.calls == 0
    assert answer.citations == ()
    assert "The corpus contains no drug label for ozempic." in answer.not_covered


def test_disclaimer_is_always_in_the_rendered_answer() -> None:
    generator, _ = _generator("Contraindicated in pregnancy [S1].")
    markdown = generator.generate("question", PLAN, [WARFARIN]).to_markdown()
    assert MEDICAL_DISCLAIMER in markdown
    assert "[S1] Warfarin (DailyMed) > 4 CONTRAINDICATIONS (dailymed-warfarin.md)" in markdown

def test_transfer_check_still_works_when_citations_follow_the_period() -> None:
    # Observed in practice: "...m². [S1][S2]". Without normalization, the citation
    # would be attached to the next sentence and this transfer would go unnoticed.
    plan = Plan(PLAN.sub_questions, unknown_drugs=("naproxen",))
    generator, _ = _generator("Naproxen can cause gastrointestinal bleeding. [S1] The end.")
    answer = generator.generate("question", plan, [IBUPROFEN])
    assert any("naproxen" in w and "never mention it" in w for w in answer.warnings)

def test_answer_language_is_given_explicitly_to_the_model() -> None:
    plan = Plan(PLAN.sub_questions, language="English")
    generator, completions = _generator("Contraindicated in pregnancy [S1].")
    generator.generate("question", plan, [WARFARIN])
    assert "Answer language: English" in completions.kwargs["messages"][1]["content"]

def test_deterministic_checks_produce_typed_major_issues() -> None:
    generator, _ = _generator("Contraindicated in pregnancy [S1]. Also something [S7]. Again [S7].")
    answer = generator.generate("question", PLAN, [WARFARIN])
    # The repeated invalid citation is reported once.
    assert [(i.kind, i.severity, i.detected_by) for i in answer.check_issues] == [
        ("invalid_citation", "major", "check")
    ]


def test_revision_prompt_includes_previous_draft_and_reviewer_feedback() -> None:
    generator, completions = _generator("Corrected answer [S1].")
    generator.generate(
        "question", PLAN, [WARFARIN],
        previous_draft="Old draft [S1].",
        reviewer_feedback='- [major unsupported] "Old draft": not in the source. Fix: remove it.',
    )
    user_message = completions.kwargs["messages"][1]["content"]
    assert "Previous draft (to correct):\nOld draft [S1]." in user_message
    assert "Reviewer issues to fix:" in user_message and "not in the source" in user_message

def test_value_from_the_question_silently_restated_in_another_unit_is_flagged() -> None:
    generator, _ = _generator("An eGFR of 25 mL/min/1.73 m² is below the threshold [S1].")
    answer = generator.generate("Patient with an eGFR of 25 mL/min: contraindicated?", PLAN, [WARFARIN])
    assert [(i.kind, i.severity, i.detected_by) for i in answer.check_issues] == [
        ("clinically_ambiguous", "major", "check")
    ]


def test_value_kept_as_given_with_a_stated_assumption_is_accepted() -> None:
    generator, _ = _generator(
        "The question gives 25 mL/min; assuming an eGFR normalized to 25 mL/min/1.73 m², "
        "it is below the threshold [S1]."
    )
    answer = generator.generate("Patient with an eGFR of 25 mL/min: contraindicated?", PLAN, [WARFARIN])
    assert answer.check_issues == ()


def test_values_in_the_same_unit_or_without_unit_are_not_flagged() -> None:
    generator, _ = _generator("A dose of 5 mg is described; checks every 1 to 4 weeks [S1].")
    answer = generator.generate("Is 5 mg of warfarin a usual dose?", PLAN, [WARFARIN])
    assert answer.check_issues == ()


def test_question_outside_the_sources_gets_the_standard_message_without_citations() -> None:
    content = json.dumps({
        "answer": "Ozempic causes nausea [S1].",  # must never reach the user
        "not_covered": ["Side effects of Ozempic."],
        "covered": False,
    })
    completions = _FakeCompletions(content)
    generator = AnswerGenerator(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "fake-model")
    plan = Plan(PLAN.sub_questions, unknown_drugs=("Ozempic",))

    answer = generator.generate("What are the side effects of Ozempic?", plan, [IBUPROFEN])

    assert answer.text == NOT_COVERED_MESSAGE
    assert answer.citations == () and answer.check_issues == ()
    assert "The corpus contains no drug label for Ozempic." in answer.not_covered