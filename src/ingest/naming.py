"""The force-plate filename contract, in one place.

A raw trial is identified by its filename: ``<ATHLETE_CODE>_<YYYY-MM-DD>.csv``.
That contract was previously implemented twice -- a strict regex in the ingest
pipeline and a loose ``split("_")`` in the dashboard's query layer -- which is
how a file called ``ATH-007_2026-08-14 2.csv`` came to be silently skipped by
the pipeline and simultaneously offered to a coach as a trial dated
"2026-08-14 2". One parser, one behaviour.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

TRACE_NAME = re.compile(r"^(?P<code>[A-Za-z]+-\d+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")

__all__ = ["TRACE_NAME", "parse_trace_name", "iter_trace_files"]


def parse_trace_name(name: str) -> tuple[str, date] | None:
    """Return (athlete_code, session_date), or None if the name is malformed."""
    m = TRACE_NAME.match(name)
    if not m:
        return None
    try:
        return m["code"], date.fromisoformat(m["date"])
    except ValueError:
        return None


def iter_trace_files(directory: Path):
    """Yield (path, athlete_code, session_date) for well-formed files, and
    (path, None, None) for malformed ones so callers can report them rather
    than pretend they were not there."""
    if not directory.exists():
        return
    for p in sorted(directory.glob("*.csv")):
        parsed = parse_trace_name(p.name)
        if parsed is None:
            yield p, None, None
        else:
            yield p, parsed[0], parsed[1]
