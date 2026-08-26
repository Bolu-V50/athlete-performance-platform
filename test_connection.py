"""Phase 0 smoke test: prove we can reach the Supabase Postgres instance."""
from sqlalchemy import text

from src.db.connection import get_engine, redacted_url


def main() -> None:
    print(f"connecting to: {redacted_url()}")
    with get_engine().connect() as conn:
        print(conn.execute(text("select version()")).scalar())
        print("current_database:", conn.execute(text("select current_database()")).scalar())
        print("current_user    :", conn.execute(text("select current_user")).scalar())


if __name__ == "__main__":
    main()
