"""Apply every view definition. Idempotent (CREATE OR REPLACE)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url

HERE = Path(__file__).parent
VIEW_FILES = ["views.sql", "views_qualities.sql"]


def main() -> None:
    print(f"applying {', '.join(VIEW_FILES)} to {redacted_url()}")
    with get_engine().begin() as conn:
        for name in VIEW_FILES:
            conn.execute(text((HERE / name).read_text()))
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
