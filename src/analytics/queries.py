"""Read-only query layer for the dashboard.

The dashboard never computes a metric. Every number it shows comes from a view
in ``src/db/views.sql``, so what a coach sees on screen and what a SQL user gets
from the database are the same number by construction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine
from src.ingest.naming import parse_trace_name

SAMPLE_TRACES = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "sample_traces"
ALL_TRACES = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "force_plate"


def _df(sql: str, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.DataFrame(conn.execute(text(sql), params).mappings().all())


ATTENTION_ORDER = (
    # Within an attention rank the worst case must come first: the most negative
    # z-score, then the highest workload ratio. Falling back to athlete_code put
    # a -1.89 SD athlete ahead of a -2.79 SD one purely because 006 sorts before
    # 009, which makes "the first to look at" wrong.
    "order by attention_rank, z_score asc nulls last, acwr desc nulls last, athlete_code"
)


def squad_status() -> pd.DataFrame:
    return _df(f"select * from v_athlete_status {ATTENTION_ORDER}")


def squads() -> list[str]:
    return sorted(_df("select distinct squad from athletes where squad is not null")["squad"])


def data_window() -> tuple[date, date]:
    r = _df("select min(session_date) lo, max(session_date) hi from sessions").iloc[0]
    return r.lo, r.hi


def cmj_series(athlete_code: str, start: date, end: date) -> pd.DataFrame:
    return _df(
        """
        select session_date, jump_height_m, baseline_mean, baseline_sd,
               baseline_n, z_score, baseline_status
        from v_cmj_flags
        where athlete_code = :code and session_date between :lo and :hi
        order by session_date
        """,
        code=athlete_code, lo=start, hi=end,
    )


def acwr_series(athlete_code: str, start: date, end: date) -> pd.DataFrame:
    return _df(
        """
        select v.date, v.session_load, v.acute_load, v.chronic_load, v.acwr, v.acwr_zone
        from v_acwr v join athletes a using (athlete_id)
        where a.athlete_code = :code and v.date between :lo and :hi
        order by v.date
        """,
        code=athlete_code, lo=start, hi=end,
    )


def trial_metrics(athlete_code: str, session_date: date) -> pd.DataFrame:
    return _df(
        """
        select m.metric_name, m.metric_value, m.source
        from performance_metrics m
        join sessions s using (session_id)
        join athletes a using (athlete_id)
        where a.athlete_code = :code and s.session_date = :d
        order by m.metric_name
        """,
        code=athlete_code, d=session_date,
    )


def recent_runs(limit: int = 10) -> pd.DataFrame:
    return _df(
        "select run_id, source, started_at, status, rows_read, rows_loaded, rows_rejected "
        "from pipeline_runs order by run_id desc limit :n",
        n=limit,
    )


def recent_rejections(limit: int = 25) -> pd.DataFrame:
    return _df(
        "select logged_at, severity, rule, athlete_code, source_ref, detail "
        "from data_quality_log order by issue_id desc limit :n",
        n=limit,
    )


def find_trace(athlete_code: str, session_date: date) -> Path | None:
    """Locate a raw force-plate file.

    Only the most recent trial per athlete is committed; the full set is
    regenerated locally. The dashboard degrades to 'no raw trace available'
    rather than failing when a file is absent.
    """
    name = f"{athlete_code}_{session_date}.csv"
    for folder in (ALL_TRACES, SAMPLE_TRACES):
        p = folder / name
        if p.exists():
            return p
    return None


def available_trace_dates(athlete_code: str) -> list[str]:
    """Dates with a readable raw file. Malformed filenames are excluded here for
    the same reason the pipeline excludes them -- offering "2026-08-14 2" as a
    trial date is worse than offering nothing."""
    seen: set[str] = set()
    for folder in (ALL_TRACES, SAMPLE_TRACES):
        if not folder.exists():
            continue
        for p in folder.glob(f"{athlete_code}_*.csv"):
            parsed = parse_trace_name(p.name)
            if parsed is not None:
                seen.add(parsed[1].isoformat())
    return sorted(seen, reverse=True)


def ingested_session_dates(athlete_code: str) -> set[str]:
    """Dates that actually made it into the database. A raw file on disk is not
    proof the trial passed validation."""
    df = _df(
        "select s.session_date from sessions s join athletes a using (athlete_id) "
        "where a.athlete_code = :c",
        c=athlete_code,
    )
    return set() if df.empty else {d.isoformat() for d in df["session_date"]}


# ---------------------------------------------------------------------------
# physical qualities
# ---------------------------------------------------------------------------
def quality_profile(athlete_code: str) -> pd.DataFrame:
    """One row per physical quality: headline metric, fitted trend, direction."""
    return _df(
        "select * from v_quality_profile where athlete_code = :c order by quality_order, display_name",
        c=athlete_code,
    )


def metric_trends(athlete_code: str) -> pd.DataFrame:
    return _df(
        "select * from v_metric_trend where athlete_code = :c "
        "order by quality_order, is_headline desc, display_name",
        c=athlete_code,
    )


def test_days(athlete_code: str) -> pd.DataFrame:
    """Dates on which this athlete was tested, and what was measured."""
    return _df(
        """
        select session_date,
               count(distinct session_type) n_tests,
               count(*)                     n_metrics,
               string_agg(distinct session_type, ', ' order by session_type) tests
        from v_test_day where athlete_code = :c
        group by session_date order by session_date desc
        """,
        c=athlete_code,
    )


def test_day_detail(athlete_code: str, session_date) -> pd.DataFrame:
    return _df(
        "select * from v_test_day where athlete_code = :c and session_date = :d "
        "order by quality_order, is_headline desc, display_name",
        c=athlete_code, d=session_date,
    )


def metric_series(athlete_code: str, metric_name: str) -> pd.DataFrame:
    return _df(
        """
        select session_date, metric_value, display_name, unit, higher_is_better, quality_name
        from v_metric_history
        where athlete_code = :c and metric_name = :m
        order by session_date
        """,
        c=athlete_code, m=metric_name,
    )


def qualities() -> pd.DataFrame:
    return _df("select * from quality_catalog order by sort_order")


def headline_history(athlete_code: str) -> pd.DataFrame:
    """Every measurement of every headline metric, for the small-multiple trends."""
    return _df(
        """
        select h.session_date, h.metric_value, h.metric_name, h.display_name,
               h.unit, h.quality_name, h.quality_order, h.higher_is_better
        from v_metric_history h
        where h.athlete_code = :c and h.is_headline
        order by h.quality_order, h.display_name, h.session_date
        """,
        c=athlete_code,
    )
