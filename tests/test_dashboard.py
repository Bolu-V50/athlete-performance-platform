"""Smoke tests for the dashboard.

A dashboard that raises on load is worse than no dashboard, and the failure only
shows up when someone opens it. Streamlit's AppTest executes the whole script in
process, so a broken chart spec or a renamed column fails here instead of in
front of a coach.

Requires a database, so these skip when SUPABASE_DB_URL is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

needs_db = pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="no database configured")

pytestmark = needs_db


APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def all_text(at) -> str:
    """Every string the page renders, whatever widget carried it.

    These assertions used to name the widget -- at.info[0] for the briefing, a
    caption starting "Briefing source". Moving the briefing from an alert box
    into a card broke them without anything being wrong, so they check what the
    page says rather than which container said it.
    """
    parts = []
    for group in (at.markdown, at.caption, at.info, at.success, at.warning, at.error):
        parts += [e.value for e in group]
    for h in (at.title, at.header, at.subheader):
        parts += [e.value for e in h]
    return " ".join(parts)


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=180).run()


def test_app_renders_without_raising(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_first_screen_answers_who_needs_attention(app):
    """The design constraint is that a coach gets the answer without scrolling,
    so the headline elements must actually be on the page."""
    labels = {m.label for m in app.metric}
    assert {"Athletes monitored", "Flagged", "On watch", "Load concerns"} <= labels
    text = all_text(app)
    assert "Who needs attention" in text
    assert "Today" in text, "the squad briefing is not on the first screen"


def test_briefing_declares_its_own_provenance(app):
    """A coach must be able to tell a model-written sentence from a template one."""
    text = all_text(app)
    assert ("numeric guard passed" in text) or ("Deterministic template" in text), (
        "the briefing does not say whether a model wrote it and whether it was checked"
    )


def test_dashboard_reads_metrics_rather_than_recomputing_them(app):
    """The jump height on screen must be the value the database returns."""
    from src.analytics.queries import squad_status

    status = squad_status()
    on_screen = {m.label: m.value for m in app.metric}
    assert on_screen["Athletes monitored"] == str(len(status))
    assert on_screen["Flagged"] == str((status["baseline_status"] == "flag").sum())


def test_attention_ordering_puts_the_worst_case_first():
    """Within an attention rank, ties were broken by athlete_code, which put a
    -1.89 SD athlete ahead of a -2.79 SD one. The worst case must lead."""
    from src.analytics.queries import squad_status

    s = squad_status()
    top = s[s["attention_rank"] == s["attention_rank"].min()]
    if len(top) > 1 and top["z_score"].notna().all():
        z = top["z_score"].astype(float).tolist()
        assert z == sorted(z), f"attention queue is not ordered worst-first: {z}"


def test_a_rejected_trial_is_never_presented_as_a_measurement():
    """A trial the pipeline threw away must not appear on screen as a result.
    The synthetic data plants a trial with a mis-set amplifier gain that reads
    1.548 m; before this fix the dashboard showed '154.8 cm' as a headline
    metric for an athlete whose real jumps are around 30 cm."""
    from streamlit.testing.v1 import AppTest

    # Find the rejected trial from the database rather than hard-coding an
    # athlete. A previous version named ATH-007; when the roster changed sport
    # the seeded fault moved to another athlete and this test skipped silently,
    # which is the failure mode it exists to catch.
    from src.analytics.queries import _df

    rejected = _df(
        "select athlete_code, source_ref from data_quality_log "
        "where rule = 'cmj_rejected' and detail like '%outside 0.05-1.20%' "
        "order by issue_id desc limit 1"
    )
    assert not rejected.empty, "the synthetic data no longer seeds an impossible jump"
    code = rejected["athlete_code"].iloc[0]
    bad = rejected["source_ref"].iloc[0].rsplit("_", 1)[1].removesuffix(".csv")

    at = AppTest.from_file(str(APP), default_timeout=240).run()
    boxes = {s.label: s for s in at.selectbox}
    assert code in boxes["Athlete detail"].options, f"{code} is not selectable"
    boxes["Athlete detail"].set_value(code).run()

    trial = {s.label: s for s in at.selectbox}["Trial"]
    assert any(o.startswith(bad) for o in trial.options), (
        f"the rejected trial {bad} is not offered in the picker"
    )
    trial.set_value(bad).run()

    errors = " ".join(e.value for e in at.error)
    assert "rejected by pipeline validation" in errors
    assert "outside 0.05-1.20" in errors
    # and the impossible value must not be sitting in the headline metric row
    headline = [m for m in at.metric if m.label in ("RSI-mod", "Peak power", "Contraction")]
    assert not headline, "headline metrics are still rendered for a rejected trial"


def test_trial_picker_offers_no_malformed_dates():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=200).run()
    trial = [s for s in at.selectbox if s.label == "Trial"]
    if not trial:
        pytest.skip("no raw traces available")
    for option in trial[0].options:
        assert len(option.split("  ·")[0]) == 10, f"malformed trial date offered: {option!r}"
