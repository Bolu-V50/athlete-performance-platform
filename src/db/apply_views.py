"""Apply src/db/views.sql. Idempotent (CREATE OR REPLACE)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url

VIEWS_PATH = Path(__file__).with_name("views.sql")


def main() -> None:
    print(f"applying {VIEWS_PATH.name} to {redacted_url()}")
    with get_engine().begin() as conn:
        conn.execute(text(VIEWS_PATH.read_text()))
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
    print("views:", ", ".join(rows) or "(none)")


if __name__ == "__main__":
    main()
