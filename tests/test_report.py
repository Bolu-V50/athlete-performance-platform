"""Tests for the athlete report.

The numeric guard is only one of four checks, because it turned out to catch
only one class of error. Every figure a model quoted could be real while the
sentence around it was still wrong: a metric described as rising when the two
numbers in the same clause show it falling, a change called "within noise" when
it is three times the athlete's variation, a female athlete referred to as
"his". These tests cover each check independently.
"""

from __future__ import annotations

import os

import pytest

from src.analytics.briefing import (
    contains_prescription,
    direction_contradictions,
    gendered_pronouns,
    guard_text,
)

needs_db = pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="no database configured")


# ---------------------------------------------------------------------------
# the four independent checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "RSI-modified rose from 0.360 to 0.324 m/s.",
        "The Yo-Yo distance increased from 1482 to 1405 m.",
        "IMTP fell from 29.4 to 34.7 N/kg.",
        "Skinfolds dropped from 52.9 to 60.2 mm.",
    ],
)
def test_a_claim_that_contradicts_its_own_numbers_is_caught(text):
    """No knowledge of the metric is needed: if the sentence says something rose
    and then names a smaller second number, the sentence contradicts itself."""
    assert direction_contradictions(text), f"missed a contradiction in: {text}"


@pytest.mark.parametrize(
    "text",
    [
        "Skinfolds fell from 60.2 to 52.9 mm, a 12.9% improvement.",
        "Jump height rose from 0.24 to 0.29 m.",
        "IMTP dropped from 34.7 to 29.4 N/kg.",
        "The athlete was tested 45 times.",
        "Neuromuscular power is declining.",
    ],
)
def test_consistent_statements_are_not_flagged(text):
    assert not direction_contradictions(text), f"false positive on: {text}"


def test_gendered_pronouns_are_rejected():
    """The athletes table is de-identified and carries no sex into the facts, so
    a pronoun is always a guess. The model called a woman in the football squad
    'his'."""
    assert gendered_pronouns("This exceeds his repeat variation.") == ["his"]
    assert gendered_pronouns("She improved.") == ["she"]
    assert not gendered_pronouns("This exceeds the athlete's repeat variation.")
    assert not gendered_pronouns("They were tested nine times.")


def test_prescription_language_is_rejected():
    assert contains_prescription("The athlete should add a deload week.")
    assert contains_prescription("Reduce the weekly volume.")
    assert not contains_prescription("Maximal strength is the area the data points to.")


def test_numeric_guard_traces_every_figure_to_the_facts():
    facts = "IMTP fell from 34.7 to 29.4 N/kg, a 12.3% worsening. Repeat variation 6.4%."
    assert guard_text("IMTP is down 12.3%, beyond the 6.4% variation.", facts)[0]
    ok, offenders = guard_text("IMTP is down 19.8%.", facts)
    assert not ok and 19.8 in offenders


# ---------------------------------------------------------------------------
# slot behaviour, with stub backends
# ---------------------------------------------------------------------------
class _Stub:
    name = "stub"
    available = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, system, user):  # noqa: ARG002
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


@pytest.fixture(scope="module")
def rep():
    # pytest refuses marks on fixtures, so the skip is raised from the body.
    if not os.getenv("SUPABASE_DB_URL"):
        pytest.skip("no database configured")
    from src.analytics.report import collect_report_data

    return collect_report_data("ATH-009")


@needs_db
def test_a_bad_generation_is_retried_before_falling_back(rep):
    """Sampling means a slot can fail one draw and pass the next on identical
    input. Demoting it to the template for one unlucky draw throws away a good
    section for no reason."""
    from src.analytics.report import facts_summary, generate_slot

    stub = _Stub("The athlete jumped 9.99 m.", "Neuromuscular power is declining.")
    slot = generate_slot("summary", rep, facts_summary(rep), stub)
    assert stub.calls == 2
    assert slot.source == "stub" and slot.guard_passed is True


@needs_db
def test_two_bad_generations_fall_back_with_the_reason_recorded(rep):
    from src.analytics.report import facts_summary, generate_slot

    stub = _Stub("The athlete jumped 9.99 m.")
    slot = generate_slot("summary", rep, facts_summary(rep), stub)
    assert stub.calls == 2
    assert slot.source == "template" and slot.guard_passed is False
    assert "9.99" in (slot.fallback_reason or "")


@needs_db
def test_a_transport_failure_is_not_retried(rep):
    """A guard rejection may fix itself on a resample; a dead connection will not."""
    from src.analytics.report import facts_summary, generate_slot

    class Broken:
        name = "broken"
        available = True
        calls = 0

        def generate(self, system, user):  # noqa: ARG002
            Broken.calls += 1
            raise TimeoutError("upstream timed out")

    slot = generate_slot("summary", rep, facts_summary(rep), Broken())
    assert Broken.calls == 1
    assert slot.source == "template" and "timed out" in (slot.fallback_reason or "")


# ---------------------------------------------------------------------------
# the deterministic scaffold
# ---------------------------------------------------------------------------
@needs_db
def test_facts_state_which_way_the_raw_number_moved(rep):
    """Handed only a sign-corrected percentage, the model described a skinfold
    sum that had fallen as 'rising'. The polarity is resolved in the facts."""
    from src.analytics.report import facts_qualities

    facts = facts_qualities(rep)
    assert " fell from " in facts or " rose from " in facts
    for line in facts.splitlines():
        if "Skinfold" in line:
            assert "fell from" in line and "improvement" in line


@needs_db
def test_facts_decide_the_noise_verdict_rather_than_the_model(rep):
    """Left to the model this produced errors in both directions in one
    paragraph. The comparison is made in code and stated in capitals."""
    from src.analytics.report import facts_qualities

    facts = facts_qualities(rep)
    assert "LARGER than that variation" in facts or "SMALLER than that variation" in facts


@needs_db
def test_report_renders_every_section_without_a_model():
    from src.analytics.briefing import TemplateBackend
    from src.analytics.report import build_report, render_markdown

    md = render_markdown(build_report("ATH-009", TemplateBackend()))
    for heading in ("## 1. Summary", "## 2. Physical qualities", "## 3. Current readiness",
                    "## 4. Training load", "## 5. What the data points to",
                    "## 6. How to read this"):
        assert heading in md, f"missing {heading}"
    assert "| Quality | Measure |" in md
    assert "does not prescribe training" in md


@needs_db
def test_report_prose_is_clean_on_a_live_model_run():
    """End-to-end: whatever the model writes must clear all four checks."""
    from src.analytics.report import build_report

    rep = build_report("ATH-009")
    for name, slot in rep.slots.items():
        assert not gendered_pronouns(slot.text), f"{name}: {gendered_pronouns(slot.text)}"
        assert not direction_contradictions(slot.text), f"{name}: {direction_contradictions(slot.text)}"
        assert not contains_prescription(slot.text), f"{name}: prescribed training"
