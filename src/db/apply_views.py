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


def main() -> None:
    print(f"applying {', '.join(VIEW_FILES)} to {redacted_url()}")
    with get_engine().begin() as conn:
        for name in VIEW_FILES:
            run_sql_file(conn, HERE / name)
        rows = (
            conn.execute(
                text(
                    "select table_name from information_schema.views "
                    "where table_schema = 'public' order by table_name"
                )
            )
            .scalars()
            .all()
        )
    print("views:", ", ".join(rows))


if __name__ == "__main__":
    main()
