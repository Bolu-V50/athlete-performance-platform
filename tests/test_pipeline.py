"""Tests for the ingest layer.

The validation rules and the file-naming contract are tested without a database
so they run in CI without credentials. The idempotency guarantee is tested
against a real database and skips when SUPABASE_DB_URL is absent.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from src.ingest.pipeline import (
    DURATION_RANGE,
    METRIC_KEYS,
    SRPE_RANGE,
    RunStats,
    TRACE_NAME,
    extract_force_files,
)

needs_db = pytest.mark.skipif(
    not os.getenv("SUPABASE_DB_URL"), reason="no database configured"
)


# ---------------------------------------------------------------------------
# file-naming contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,code,day",
    [
        ("ATH-001_2026-08-24.csv", "ATH-001", "2026-08-24"),
        ("ATH-999_2026-01-01.csv", "ATH-999", "2026-01-01"),
    ],
)
def test_trace_filename_parses(name, code, day):
    m = TRACE_NAME.match(name)
    assert m and m["code"] == code and m["date"] == day


@pytest.mark.parametrize(
    "name",
    ["notes.txt", "ATH-001.csv", "ATH-001_24-08-2026.csv", "ATH-001_2026-08-24.CSV.bak"],
)
def test_malformed_filenames_are_skipped(name):
    assert TRACE_NAME.match(name) is None


def test_extract_ignores_unparseable_files(tmp_path: Path):
    for n in ["ATH-001_2026-08-24.csv", "README.md", "scratch.csv"]:
        (tmp_path / n).write_text("x")
    found = list(extract_force_files(tmp_path))
    assert len(found) == 1
    assert found[0][1] == "ATH-001" and found[0][2] == date(2026, 8, 24)


# ---------------------------------------------------------------------------
# acceptance thresholds are physiology, so pin them
# ---------------------------------------------------------------------------
def test_srpe_range_is_the_borg_cr10_scale():
    assert SRPE_RANGE == (0.0, 10.0)


def test_duration_range_excludes_zero_and_negatives():
    lo, hi = DURATION_RANGE
    assert lo == 0.0 and hi > 0
    assert not (lo < -45.0 <= hi)   # the injected negative duration must fail
    assert not (lo < 0.0)           # a zero-minute session is not a session


def test_metric_keys_exclude_provenance_fields():
    """Sample rate and filter cutoff describe how a number was produced, not the
    athlete, so they must not be stored as performance metrics."""
    assert "sample_rate_hz" not in METRIC_KEYS
    assert "filter_cutoff_hz" not in METRIC_KEYS
    assert "jump_height_m" in METRIC_KEYS and "rsi_mod" in METRIC_KEYS


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------
def test_rejects_and_warns_are_counted_differently():
    """A warning keeps the row; a rejection drops it. Only rejections reduce
    the loaded count, so they must not share a counter."""
    st = RunStats(source="test")
    st.warn("f.csv", "ATH-001", "cmj_warning", "unstable weighing")
    st.warn("f.csv", "ATH-001", "duplicate_athlete_day", "kept last")
    st.reject("g.csv", "ATH-999", "unknown_athlete_code", "not on roster")
    assert st.rejected == 1
    assert len(st.issues) == 3
    assert sum(1 for i in st.issues if i["severity"] == "warn") == 2


# ---------------------------------------------------------------------------
# idempotency, against a real database
# ---------------------------------------------------------------------------
@needs_db
def test_rerunning_the_pipeline_does_not_duplicate_rows():
    from sqlalchemy import text

    from src.db.connection import get_engine
    from src.ingest.pipeline import run_pipeline

    def counts():
        with get_engine().connect() as c:
            return tuple(
                c.execute(text(f"select count(*) from {t}")).scalar()
                for t in ("athletes", "sessions", "performance_metrics", "training_load")
            )

    run_pipeline(verbose=False)
    before = counts()
    run_pipeline(verbose=False)
    assert counts() == before, "re-running the pipeline changed row counts"
