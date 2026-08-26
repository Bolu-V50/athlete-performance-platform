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
    assert {"Athletes monitored", "Flagged today", "On watch", "Load concerns"} <= labels
    assert any("Who needs attention" in s.value for s in app.subheader)
    assert app.info and "briefing" in app.info[0].value.lower()


def test_briefing_declares_its_own_provenance(app):
    """A coach must be able to tell a model-written sentence from a template one."""
    badges = [c.value for c in app.caption if "Briefing source" in c.value]
    assert badges, "the briefing does not say where it came from"
    badge = badges[0]
    assert ("numeric guard passed" in badge) or ("deterministic template" in badge)


def test_dashboard_reads_metrics_rather_than_recomputing_them(app):
    """The jump height on screen must be the value the database returns."""
    from src.analytics.queries import squad_status

    status = squad_status()
    on_screen = {m.label: m.value for m in app.metric}
    assert on_screen["Athletes monitored"] == str(len(status))
    assert on_screen["Flagged today"] == str((status["baseline_status"] == "flag").sum())


def test_attention_ordering_puts_the_worst_case_first():
    """Within an attention rank, ties were broken by athlete_code, which put a
    -1.89 SD athlete ahead of a -2.79 SD one. The worst case must lead."""
    from src.analytics.queries import squad_status

    s = squad_status()
    top = s[s["attention_rank"] == s["attention_rank"].min()]
    if len(top) > 1 and top["z_score"].notna().all():
        z = top["z_score"].astype(float).tolist()
        assert z == sorted(z), f"attention queue is not ordered worst-first: {z}"
