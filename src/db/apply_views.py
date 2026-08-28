"""Apply every view definition. Idempotent.

Executed with exec_driver_sql rather than text(): these files contain no bind
parameters, and SQLAlchemy's `:name` parsing would misread ordinary SQL.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url
from src.db.migrate import run_sql_file

HERE = Path(__file__).parent
VIEW_FILES = ["views.sql", "views_qualities.sql", "views_report.sql", "views_normative.sql"]


def drop_existing_views(conn) -> list[str]:
    """Drop every view in the public schema, in one CASCADE statement.

    The teardown used to be hand-written DROP lines at the top of the view files,
    and it went stale twice: a new view built on v_metric_history blocked its
    drop, and both times the file applied cleanly on a fresh database and failed
    on the second run. A list that has to be updated whenever a view gains a
    dependant will eventually not be.

    Postgres already knows the dependency graph, so ask it for the views and let
    CASCADE resolve the order. This is safe precisely because this function is
    only ever called immediately before recreating the complete set: a view in
    the database but not in these files is an orphan from a previous schema and
    should not survive.
    """
    names = (
        conn.execute(
            text(
                "select table_name from information_schema.views "
                "where table_schema = 'public' order by table_name"
            )
        )
        .scalars()
        .all()
    )
    if names:
        joined = ", ".join(f'public."{n}"' for n in names)
        conn.exec_driver_sql(f"drop view if exists {joined} cascade")

    # Materialised views live in pg_matviews, not information_schema.views, and
    # need their own DROP. Missing them leaves a stale copy that blocks the
    # rebuild.
    mats = (
        conn.execute(
            text("select matviewname from pg_matviews where schemaname = 'public' order by 1")
        )
        .scalars()
        .all()
    )
    for m in mats:
        conn.exec_driver_sql(f'drop materialized view if exists public."{m}" cascade')
    return list(names) + list(mats)


def main() -> None:
    print(f"applying {', '.join(VIEW_FILES)} to {redacted_url()}")
    with get_engine().begin() as conn:
        dropped = drop_existing_views(conn)
        if dropped:
            print(f"dropped {len(dropped)} existing view(s) before rebuilding")
        for name in VIEW_FILES:
            run_sql_file(conn, HERE / name)
        rows = (
            conn.execute(
                text(
                    "select table_name from information_schema.views "
                    "where table_schema = 'public' "
                    "union all select matviewname from pg_matviews "
                    "where schemaname = 'public' order by 1"
                )
            )
            .scalars()
            .all()
        )
    print(f"views ({len(rows)}):", ", ".join(rows))


if __name__ == "__main__":
    main()
