"""Shared test setup.

Database-backed tests skip when SUPABASE_DB_URL is unset. Without this file that
check depended on whether some earlier test module happened to have imported
src.db.connection, which calls load_dotenv() as a side effect -- so the full
suite ran the database tests while running a single file silently skipped them.
A test that skips when you did not intend it to is worse than one that fails.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
