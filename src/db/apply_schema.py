"""Apply the schema, catalogue and normative reference data. Idempotent.

Migration files are executed with exec_driver_sql, not text(). SQLAlchemy reads
`:name` in a text() string as a bind parameter, and a literature citation like
"Sports (Basel) 2018;6(4):174" contains one. A migration file has no parameters
by definition, so the parsing is not just unnecessary, it is actively wrong.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url
from src.db.migrate import run_sql_file

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
CATALOG_PATH = Path(__file__).with_name("catalog.sql")
NORMATIVE_PATH = Path(__file__).with_name("normative.sql")


def main() -> None:
    print(f"applying schema.sql + catalog.sql + normative.sql to {redacted_url()}")
    with get_engine().begin() as conn:
        for path in (SCHEMA_PATH, CATALOG_PATH, NORMATIVE_PATH):
            run_sql_file(conn, path)
        rows = (
            conn.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_schema = 'public' order by table_name"
                )
            )
            .scalars()
            .all()
        )
    print("public tables:", ", ".join(rows) or "(none)")


if __name__ == "__main__":
    main()
