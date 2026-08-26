"""Tests that the migrations can be applied more than once.

Both times a view teardown went stale, the SQL applied perfectly to a fresh
database and failed on the second run, so nothing local caught it and CI found
it after the push. A migration that only works once is not a migration.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

needs_db = pytest.mark.skipif(not os.getenv("SUPABASE_DB_URL"), reason="no database configured")
pytestmark = needs_db


def _views(conn) -> set[str]:
    return set(
        conn.execute(
            text("select table_name from information_schema.views where table_schema = 'public'")
        ).scalars()
    )


def _tables(conn) -> set[str]:
    return set(
        conn.execute(
            text(
                "select table_name from information_schema.tables "
                "where table_schema = 'public' and table_type = 'BASE TABLE'"
            )
        ).scalars()
    )


def test_views_can_be_applied_twice_in_a_row():
    """The exact failure that reached CI twice: a new view depending on
    v_metric_history blocked its drop on the second application."""
    from src.db.apply_views import main as apply_views
    from src.db.connection import get_engine

    apply_views()
    with get_engine().connect() as conn:
        first = _views(conn)
    apply_views()
    with get_engine().connect() as conn:
        second = _views(conn)

    assert first == second, f"the view set changed on re-application: {first ^ second}"
    assert len(second) >= 12


def test_schema_and_catalogue_can_be_applied_twice_in_a_row():
    from src.db.apply_schema import main as apply_schema
    from src.db.connection import get_engine

    apply_schema()
    with get_engine().connect() as conn:
        first = _tables(conn)
        n_metrics = conn.execute(text("select count(*) from metric_catalog")).scalar()
        n_norms = conn.execute(text("select count(*) from normative_values")).scalar()
    apply_schema()
    with get_engine().connect() as conn:
        assert _tables(conn) == first
        # The catalogue and reference library upsert; re-applying must not
        # duplicate them.
        assert conn.execute(text("select count(*) from metric_catalog")).scalar() == n_metrics
        assert conn.execute(text("select count(*) from normative_values")).scalar() == n_norms


def test_the_teardown_is_derived_from_the_database_not_hand_written():
    """A hand-maintained drop list is only correct until someone adds a view.
    Both stale-teardown failures came from exactly that."""
    from pathlib import Path

    sql_dir = Path(__file__).resolve().parents[1] / "src" / "db"
    for f in sql_dir.glob("views*.sql"):
        body = f.read_text().lower()
        assert "drop view" not in body, (
            f"{f.name} contains a hand-written DROP; teardown belongs in apply_views.py, "
            "which derives it from the database"
        )


def test_every_view_file_is_registered_for_application():
    """A view file that exists but is not in VIEW_FILES is dead SQL: it passes
    review, never runs, and its views are silently missing."""
    from pathlib import Path

    from src.db.apply_views import VIEW_FILES

    sql_dir = Path(__file__).resolve().parents[1] / "src" / "db"
    on_disk = {f.name for f in sql_dir.glob("views*.sql")}
    assert on_disk == set(VIEW_FILES), f"unregistered: {on_disk - set(VIEW_FILES)}"
