"""Tests for the LLM briefing layer.

These run without a database and without an API key: the snapshot is built by
hand so the governance mechanisms can be tested in isolation. That is the point
of the design -- the guard and the fallback are testable properties, not
promises made in a README.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.analytics.briefing import (
    AthleteRow,
    BriefingResult,
    Snapshot,
    TemplateBackend,
    daily_briefing,
    numeric_guard,
    render_facts,
    template_briefing,
)


def make_snapshot() -> Snapshot:
    rows = [
        AthleteRow("ATH-009", "Netball-Senior", "flag", "caution",
                   0.247, 0.284, -2.79, 13.0, 1.40, 1),
        AthleteRow("ATH-006", "Jumps", "flag", "sweet_spot",
                   0.438, 0.461, -1.89, 5.0, 1.01, 1),
        AthleteRow("ATH-001", "Sprints", "normal", "sweet_spot",
                   0.437, 0.451, -0.31, 3.1, 1.09, 5),
    ]
    return Snapshot(
        as_of=date(2026, 8, 24), rows=rows, n_athletes=3,
        n_flag=2, n_watch=0, n_load_concern=1, rejected_today=6,
        notable=[r for r in rows if r.attention_rank <= 4],
    )


# ---------------------------------------------------------------------------
# the numeric guard is the governance mechanism, so test it hardest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "ATH-009 is 13.0% below baseline with an ACWR of 1.40 (caution).",
        "ATH-009 jumped 0.247 m against a baseline of 0.284 m, z-score -2.79.",
        "2 of 3 athletes are flagged as of 2026-08-24.",
    ],
)
def test_guard_accepts_values_traceable_to_the_facts(text):
    ok, offenders = numeric_guard(text, make_snapshot())
    assert ok, f"guard wrongly rejected {offenders}"


@pytest.mark.parametrize(
    "text,bad",
    [
        ("ATH-009 jumped 0.612 m today.", 0.612),              # invented a measurement
        ("ATH-009 dropped 0.037 m from baseline.", 0.037),     # did its own subtraction
        ("ATH-009 is 41.7% below baseline.", 41.7),            # invented a percentage
        ("ATH-009 has an ACWR of 2.31.", 2.31),                # invented a ratio
    ],
)
def test_guard_rejects_numbers_the_model_made_up(text, bad):
    """The model is told not to calculate. The guard is what makes that
    enforceable rather than aspirational."""
    ok, offenders = numeric_guard(text, make_snapshot())
    assert not ok
    assert any(abs(o - bad) < 1e-9 for o in offenders)


def test_guard_is_not_fooled_by_unicode_dashes_in_athlete_codes():
    """Models routinely render ATH-009 with a non-breaking hyphen. Without
    normalisation the code regex misses it and '009' is scored as a measurement."""
    ok, offenders = numeric_guard("Ath‑009 is 13.0% below baseline, ACWR 1.40.", make_snapshot())
    assert ok, f"guard tripped on an athlete code: {offenders}"


def test_guard_ignores_iso_dates():
    ok, _ = numeric_guard("As of 2026-08-24 two athletes are flagged.", make_snapshot())
    assert ok


# ---------------------------------------------------------------------------
# the deterministic path must work with no model at all
# ---------------------------------------------------------------------------
def test_template_briefing_needs_no_model_and_names_the_priority_athlete():
    s = make_snapshot()
    txt = template_briefing(s)
    assert "ATH-009" in txt                       # highest attention rank leads
    assert txt.index("ATH-009") < txt.index("ATH-006")
    assert "1.40" in txt                          # the load concern is surfaced


def test_template_briefing_survives_an_all_clear_day():
    s = make_snapshot()
    s.notable = []
    s.n_flag = s.n_load_concern = 0
    txt = template_briefing(s)
    assert "no" in txt.lower() and "ATH-" not in txt


def test_template_output_passes_its_own_guard():
    """The fallback must not be able to fail the check the model must pass."""
    s = make_snapshot()
    ok, offenders = numeric_guard(template_briefing(s), s)
    assert ok, f"the deterministic template emitted unverifiable numbers: {offenders}"


def test_daily_briefing_falls_back_when_no_backend_is_available():
    class Unavailable:
        name = "unavailable"
        available = False

        def generate(self, system, user):  # noqa: ARG002
            raise AssertionError("must not be called when unavailable")

    r = daily_briefing(make_snapshot(), Unavailable())
    assert isinstance(r, BriefingResult)
    assert r.source == "template" and not r.is_model_generated


def test_daily_briefing_discards_model_output_that_fails_the_guard():
    class Hallucinating:
        name = "fake"
        available = True

        def generate(self, system, user):  # noqa: ARG002
            return "ATH-009 jumped 0.612 m, which is 41.7% below baseline."

    r = daily_briefing(make_snapshot(), Hallucinating())
    assert r.guard_passed is False
    assert r.source == "template"
    assert "0.612" not in r.text
    assert r.guard_offenders


def test_daily_briefing_falls_back_when_the_model_errors():
    class Broken:
        name = "broken"
        available = True

        def generate(self, system, user):  # noqa: ARG002
            raise TimeoutError("upstream timed out")

    r = daily_briefing(make_snapshot(), Broken())
    assert r.source == "template" and "timed out" in (r.fallback_reason or "")


def test_daily_briefing_accepts_clean_model_output():
    class Good:
        name = "good"
        available = True

        def generate(self, system, user):  # noqa: ARG002
            return "ATH-009 is 13.0% below baseline with an ACWR of 1.40."

    r = daily_briefing(make_snapshot(), Good())
    assert r.guard_passed is True and r.source == "good" and r.is_model_generated


# ---------------------------------------------------------------------------
# the fact block is the model's entire world, so it must be complete
# ---------------------------------------------------------------------------
def test_fact_block_contains_every_notable_athlete_and_no_raw_data():
    facts = render_facts(make_snapshot())
    assert "ATH-009" in facts and "ATH-006" in facts
    assert "0.247" in facts and "1.40" in facts
    assert "13.0" in facts
    # provenance and raw signal must not leak into the model's context
    assert "force_plate" not in facts and "session_id" not in facts
