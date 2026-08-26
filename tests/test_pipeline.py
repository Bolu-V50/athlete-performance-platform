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


def test_extract_yields_csvs_and_identifies_the_parseable_ones(tmp_path: Path):
    """Non-CSV files are not the pipeline's business, but a CSV whose name does
    not parse is: it is yielded with no code/date so the caller can record it
    rather than let a trial vanish without trace."""
    for n in ["ATH-001_2026-08-24.csv", "README.md", "scratch.csv"]:
        (tmp_path / n).write_text("x")
    found = list(extract_force_files(tmp_path))
    assert len(found) == 2, "both CSVs should be seen; README.md should not"
    parsed = [(c, d) for _p, c, d in found if c is not None]
    assert parsed == [("ATH-001", date(2026, 8, 24))]


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


# ---------------------------------------------------------------------------
# one filename parser, shared by ingest and presentation
# ---------------------------------------------------------------------------
def test_ingest_and_dashboard_share_one_filename_parser():
    """These were two implementations -- a strict regex in the pipeline and a
    split('_') in the query layer. The result was a file the pipeline silently
    skipped being offered to a coach as a trial dated '2026-08-14 2'."""
    from src.analytics import queries
    from src.ingest import pipeline
    from src.ingest.naming import TRACE_NAME, parse_trace_name

    assert pipeline.TRACE_NAME is TRACE_NAME
    assert queries.parse_trace_name is parse_trace_name


@pytest.mark.parametrize(
    "name",
    [
        "ATH-007_2026-08-14 2.csv",   # an iCloud conflict copy
        "ATH-007_2026-08-14.csv.bak",
        "ATH-007_2026-13-01.csv",     # month 13
        "ATH-007_20260814.csv",
    ],
)
def test_malformed_filenames_do_not_parse(name):
    from src.ingest.naming import parse_trace_name

    assert parse_trace_name(name) is None


def test_unparseable_filenames_are_reported_not_skipped(tmp_path: Path):
    """Silently ignoring a file means a trial can disappear between the
    collection laptop and the database with nothing to show for it."""
    from src.ingest.pipeline import extract_force_files

    (tmp_path / "ATH-001_2026-08-24.csv").write_text("x")
    (tmp_path / "ATH-001_2026-08-24 2.csv").write_text("x")
    (tmp_path / "junk.csv").write_text("x")

    found = list(extract_force_files(tmp_path))
    assert len(found) == 3, "malformed files must still be yielded so they can be logged"
    unparsed = [p.name for p, code, d in found if code is None]
    assert sorted(unparsed) == ["ATH-001_2026-08-24 2.csv", "junk.csv"]
