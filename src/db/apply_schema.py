"""Apply src/db/schema.sql to the configured database. Idempotent."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
CATALOG_PATH = Path(__file__).with_name("catalog.sql")


def main() -> None:
    print(f"applying schema.sql + catalog.sql to {redacted_url()}")
    with get_engine().begin() as conn:
        conn.execute(text(SCHEMA_PATH.read_text()))
        conn.execute(text(CATALOG_PATH.read_text()))
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
