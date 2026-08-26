"""Execute a .sql migration file exactly as written.

Neither of the obvious ways to run a SQL file is safe here, and both failed on
this project's own files:

  * ``conn.execute(text(sql))`` treats ``:name`` as a bind parameter. A
    literature citation reading "Sports (Basel) 2018;6(4):174" is not a
    parameter, and SQLAlchemy raised asking for a value for ":174".
  * ``conn.exec_driver_sql(sql)`` hands the string to the driver with an empty
    parameter set, which makes psycopg apply %-interpolation. A unit column
    containing '%' then fails as an unknown placeholder.

A migration file has no parameters by construction, so the correct thing is to
pass it to the driver with the parameter argument omitted entirely -- psycopg
skips interpolation only when it is absent, not when it is empty. Going through
the DBAPI cursor keeps the statement inside the SQLAlchemy transaction.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Connection


def run_sql_file(conn: Connection, path: Path) -> None:
    """Run every statement in `path` verbatim, inside the caller's transaction."""
    sql = Path(path).read_text()
    cursor = conn.connection.cursor()
    try:
        cursor.execute(sql)          # no parameter argument: nothing is interpolated
    finally:
        cursor.close()
