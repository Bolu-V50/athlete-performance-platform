"""Apply src/db/schema.sql to the configured database. Idempotent."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def main() -> None:
    sql = SCHEMA_PATH.read_text()
    print(f"applying {SCHEMA_PATH.name} to {redacted_url()}")
    with get_engine().begin() as conn:
        conn.execute(text(sql))
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
