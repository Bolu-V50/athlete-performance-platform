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
    seen: set[str] = set()
    for folder in (ALL_TRACES, SAMPLE_TRACES):
        if folder.exists():
            seen.update(
                p.stem.split("_", 1)[1] for p in folder.glob(f"{athlete_code}_*.csv")
            )
    return sorted(seen, reverse=True)
