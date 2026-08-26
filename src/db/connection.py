"""Single source of truth for the database engine.

The .env holds a plain `postgresql://` URL so the same string works with psql,
alembic and any client. The psycopg3 driver prefix is applied here instead of
being baked into the credential, which keeps the secret portable.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def get_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Copy .env.example to .env and fill it in."
        )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Pooled engine. pre_ping guards against Supavisor dropping idle sessions."""
    return create_engine(
        get_db_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
    )


def redacted_url() -> str:
    """URL safe to print in logs / CI output."""
    import re

    return re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", get_db_url())
