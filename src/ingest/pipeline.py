"""Ingest pipeline: raw force-plate exports and sRPE diaries -> Postgres.

Three properties separate a pipeline you can put on a schedule from a script
that happens to load data once:

1. **Idempotent writes.** Every insert is ``ON CONFLICT ... DO UPDATE`` against
   a real unique constraint, so re-running the pipeline over the same files
   converges to the same rows instead of duplicating them. This is what makes
   an unattended nightly run safe: if it half-fails, you just run it again.

2. **Validation with domain thresholds.** The limits below are physiology, not
   arbitrary guard rails -- an adult CMJ outside 5-120 cm is a measurement
   fault, and sRPE is a 0-10 Borg CR-10 scale, so 14 is a typing error. Every
   rejected row is written to ``data_quality_log`` with the rule that caught
   it, so exclusions are auditable rather than silent.

3. **Run provenance.** Each execution opens a ``pipeline_runs`` row and closes
   it with counts and status, so "is today's data actually in?" is a SQL
   question.
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db.connection import get_engine
from src.ingest.naming import TRACE_NAME, iter_trace_files  # noqa: F401
from src.signal_processing.cmj import analyse_cmj

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"

# Physiological acceptance limits. Stated here rather than buried inline so a
# practitioner can see and challenge them.
SRPE_RANGE = (0.0, 10.0)          # Borg CR-10
DURATION_RANGE = (0.0, 400.0)     # minutes; a 7-hour session is a typo

# Which CMJ outputs are stored as performance metrics. Provenance fields
# (sample rate, filter cutoff) stay on the result object, not in the metrics
# table -- they describe how the number was made, not the athlete.
METRIC_KEYS = [
    "jump_height_m",
    "jump_height_flight_time_m",
    "takeoff_velocity_ms",
    "rsi_mod",
    "peak_force_n",
    "peak_force_bw",
    "peak_power_w",
    "peak_power_w_kg",
    "net_impulse_ns",
    "body_weight_n",
    "body_mass_kg",
    "unweighting_duration_s",
    "ecc_duration_s",
    "con_duration_s",
    "contraction_time_s",
    "flight_time_s",
    "countermovement_depth_m",
]

# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RunStats:
    source: str
    run_id: int | None = None
    read: int = 0
    loaded: int = 0
    rejected: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def reject(self, ref: str, code: str | None, rule: str, detail: str) -> None:
        self.rejected += 1
        self.issues.append(
            dict(source_ref=ref, athlete_code=code, rule=rule, detail=detail, severity="reject")
        )

    def warn(self, ref: str, code: str | None, rule: str, detail: str) -> None:
        self.issues.append(
            dict(source_ref=ref, athlete_code=code, rule=rule, detail=detail, severity="warn")
        )


def _open_run(conn: Connection, source: str) -> int:
    return conn.execute(
        text("insert into pipeline_runs (source) values (:s) returning run_id"),
        {"s": source},
    ).scalar_one()


def _close_run(conn: Connection, st: RunStats, status: str, error: str | None) -> None:
    conn.execute(
        text(
            "update pipeline_runs set finished_at = now(), status = :st, rows_read = :r, "
            "rows_loaded = :l, rows_rejected = :x, error_summary = :e where run_id = :id"
        ),
        dict(st=status, r=st.read, l=st.loaded, x=st.rejected, e=error, id=st.run_id),
    )
    if st.issues:
        conn.execute(
            text(
                "insert into data_quality_log (run_id, source_ref, athlete_code, rule, detail, severity) "
                "values (:run_id, :source_ref, :athlete_code, :rule, :detail, :severity)"
            ),
            [dict(run_id=st.run_id, **i) for i in st.issues],
        )


# ---------------------------------------------------------------------------
# roster
# ---------------------------------------------------------------------------
def load_roster(conn: Connection, path: Path) -> dict[str, int]:
    """Upsert the athlete roster and return code -> athlete_id.

    Idempotent on athlete_code, so a re-sent roster updates squad/sport rather
    than creating a second athlete.
    """
    df = pd.read_csv(path)
    ids: dict[str, int] = {}
    for row in df.itertuples(index=False):
        ids[row.athlete_code] = conn.execute(
            text(
                "insert into athletes (athlete_code, sport, sex, squad) "
                "values (:c, :sp, :sx, :sq) "
                "on conflict (athlete_code) do update set "
                "  sport = excluded.sport, sex = excluded.sex, squad = excluded.squad "
                "returning athlete_id"
            ),
            dict(c=row.athlete_code, sp=row.sport, sx=row.sex, sq=row.squad),
        ).scalar_one()
    return ids


def athlete_index(conn: Connection) -> dict[str, int]:
    return {
        r.athlete_code: r.athlete_id
        for r in conn.execute(text("select athlete_code, athlete_id from athletes"))
    }


# ---------------------------------------------------------------------------
# force-plate branch
# ---------------------------------------------------------------------------
def extract_force_files(directory: Path) -> Iterator[tuple[Path, str | None, date | None]]:
    """Yield every CSV, including the ones whose names do not parse.

    Skipping a malformed filename without saying so means a trial can vanish
    between the collection laptop and the database with nothing to show for it.
    The caller records these as a data-quality issue.
    """
    yield from iter_trace_files(directory)


def _upsert_session(conn: Connection, athlete_id: int, d: date, kind: str) -> int:
    # DO UPDATE rather than DO NOTHING: a no-op update still produces a row for
    # RETURNING, so a re-run gets the existing session_id instead of NULL.
    return conn.execute(
        text(
            "insert into sessions (athlete_id, session_date, session_type) "
            "values (:a, :d, :k) "
            "on conflict (athlete_id, session_date, session_type) do update set "
            "  session_type = excluded.session_type "
            "returning session_id"
        ),
        dict(a=athlete_id, d=d, k=kind),
    ).scalar_one()


def ingest_force_plate(conn: Connection, st: RunStats, directory: Path, known: dict[str, int]) -> None:
    metric_rows: list[dict[str, Any]] = []

    for path, code, d in extract_force_files(directory):
        st.read += 1
        ref = path.name

        if code is None or d is None:
            st.reject(ref, None, "unparseable_filename",
                      "expected <ATHLETE_CODE>_<YYYY-MM-DD>.csv; file not ingested")
            continue

        if code not in known:
            st.reject(ref, code, "unknown_athlete_code",
                      "athlete_code is not on the roster; trial cannot be attributed")
            continue

        try:
            df = pd.read_csv(path)
            result = analyse_cmj(df, time_col="time_s",
                                 force_cols=[c for c in df.columns if c.startswith("Fz")])
        except Exception as exc:
            st.reject(ref, code, "unreadable_trace", f"{type(exc).__name__}: {exc}")
            continue

        if not result.is_valid:
            st.reject(ref, code, "cmj_rejected",
                      " | ".join(f for f in result.quality_flags if f.startswith("REJECT")))
            continue
        for flag in result.quality_flags:
            st.warn(ref, code, "cmj_warning", flag)

        session_id = _upsert_session(conn, known[code], d, "CMJ_test")
        values = result.to_dict()
        for key in METRIC_KEYS:
            metric_rows.append(
                dict(sid=session_id, name=key, value=float(values[key]), src="force_plate")
            )
        st.loaded += 1

    if metric_rows:
        conn.execute(
            text(
                "insert into performance_metrics (session_id, metric_name, metric_value, source) "
                "values (:sid, :name, :value, :src) "
                "on conflict (session_id, metric_name, source) do update set "
                "  metric_value = excluded.metric_value, ingested_at = now()"
            ),
            metric_rows,
        )


# ---------------------------------------------------------------------------
# laboratory / field test batteries
# ---------------------------------------------------------------------------
def load_catalog(conn: Connection) -> dict[str, dict[str, Any]]:
    """The metric catalogue drives this whole branch.

    Which test a column belongs to, which device produced it, and what counts as
    a physiologically possible value all come from the catalogue rather than
    from code. Adding a new test means inserting catalogue rows -- the pipeline,
    the validation and the dashboard all pick it up with no change.
    """
    return {
        r.metric_name: dict(
            session_type=r.session_type, source=r.source,
            lo=float(r.typical_min) if r.typical_min is not None else None,
            hi=float(r.typical_max) if r.typical_max is not None else None,
            display=r.display_name, unit=r.unit,
        )
        for r in conn.execute(
            text("select metric_name, session_type, source, typical_min, typical_max, "
                 "display_name, unit from metric_catalog")
        )
    }


def ingest_battery(
    conn: Connection, st: RunStats, path: Path, known: dict[str, int],
    catalog: dict[str, dict[str, Any]],
) -> None:
    """Melt a wide test export into the long metrics table.

    Athlete management systems export one wide, sparse row per athlete-day, with
    a column per measurement and blanks where that test was not run. The long
    table wants one row per measurement. Reshaping is this function's job, and
    it is the reason the schema is long: sprint, Wingate and skinfold data land
    in the same table as jump data without a single new column.
    """
    df = pd.read_csv(path)
    value_cols = [c for c in df.columns if c not in ("athlete_code", "date")]
    unknown_cols = [c for c in value_cols if c not in catalog]
    for c in unknown_cols:
        st.warn(path.name, None, "uncatalogued_metric",
                f"column '{c}' has no metric_catalog entry; not ingested")

    metric_rows: list[dict[str, Any]] = []
    sessions_seen: set[tuple[int, str, str]] = set()
    st.read = 0

    for i, row in enumerate(df.itertuples(index=False), start=2):
        code = str(row.athlete_code)
        ref = f"{path.name}:{i}"
        measured = [c for c in value_cols
                    if c in catalog and pd.notna(getattr(row, c)) and str(getattr(row, c)) != ""]
        st.read += len(measured)

        if code not in known:
            for c in measured:
                st.reject(ref, code, "unknown_athlete_code", f"{c}: athlete not on the roster")
            continue

        for c in measured:
            spec = catalog[c]
            try:
                value = float(getattr(row, c))
            except (TypeError, ValueError):
                st.reject(ref, code, "non_numeric_value", f"{c}={getattr(row, c)!r}")
                continue
            lo, hi = spec["lo"], spec["hi"]
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                st.reject(ref, code, "value_out_of_range",
                          f"{spec['display']} {value} {spec['unit']} outside {lo}-{hi}")
                continue

            key = (known[code], str(row.date), spec["session_type"])
            if key not in sessions_seen:
                _upsert_session(conn, known[code], row.date, spec["session_type"])
                sessions_seen.add(key)
            metric_rows.append(dict(code=known[code], d=row.date,
                                    kind=spec["session_type"], name=c,
                                    value=value, src=spec["source"]))

    if metric_rows:
        conn.execute(
            text(
                "insert into performance_metrics (session_id, metric_name, metric_value, source) "
                "select s.session_id, :name, :value, :src from sessions s "
                "where s.athlete_id = :code and s.session_date = :d and s.session_type = :kind "
                "on conflict (session_id, metric_name, source) do update set "
                "  metric_value = excluded.metric_value, ingested_at = now()"
            ),
            metric_rows,
        )
    st.loaded = len(metric_rows)


# ---------------------------------------------------------------------------
# sRPE branch
# ---------------------------------------------------------------------------
def ingest_srpe(conn: Connection, st: RunStats, path: Path, known: dict[str, int]) -> None:
    df = pd.read_csv(path)
    st.read = len(df)
    clean: dict[tuple[str, str], dict[str, Any]] = {}

    for i, row in enumerate(df.itertuples(index=False), start=2):  # +2: header + 1-indexed
        ref = f"{path.name}:{i}"
        code = str(row.athlete_code)

        if code not in known:
            st.reject(ref, code, "unknown_athlete_code", "athlete_code is not on the roster")
            continue
        if pd.isna(row.srpe) or not (SRPE_RANGE[0] <= float(row.srpe) <= SRPE_RANGE[1]):
            st.reject(ref, code, "srpe_out_of_range",
                      f"sRPE {row.srpe} outside the Borg CR-10 scale {SRPE_RANGE}")
            continue
        if pd.isna(row.duration_min) or not (DURATION_RANGE[0] < float(row.duration_min) <= DURATION_RANGE[1]):
            st.reject(ref, code, "duration_out_of_range",
                      f"duration {row.duration_min} min outside {DURATION_RANGE}")
            continue

        key = (code, str(row.date))
        if key in clean:
            # One athlete-day is one load entry. Two submissions means the
            # athlete re-entered the session; the later row is the correction.
            st.warn(ref, code, "duplicate_athlete_day",
                    f"{row.date} submitted more than once; keeping the last value")
        clean[key] = dict(
            a=known[code], d=row.date,
            dur=float(row.duration_min), srpe=float(row.srpe),
        )

    if clean:
        conn.execute(
            text(
                "insert into training_load (athlete_id, date, duration_min, srpe) "
                "values (:a, :d, :dur, :srpe) "
                "on conflict (athlete_id, date) do update set "
                "  duration_min = excluded.duration_min, srpe = excluded.srpe"
            ),
            list(clean.values()),
        )
    st.loaded = len(clean)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def refresh_derived(verbose: bool = True) -> None:
    """Rebuild anything materialised from the tables the pipeline just wrote.

    v_acwr holds a day-by-day EWMA recursion. Leaving it as a plain view made
    every dashboard query re-run the whole recursion; materialising it moves
    that cost here, where it happens once per ingest instead of once per page
    load.
    """
    with get_engine().begin() as conn:
        conn.exec_driver_sql("refresh materialized view v_acwr")
    if verbose:
        print(f"[{datetime.now():%H:%M:%S}] refreshed v_acwr")


def run_pipeline(data_dir: Path = DATA, verbose: bool = True) -> list[RunStats]:
    engine = get_engine()
    results: list[RunStats] = []

    for source, fn in (
        ("force_plate_csv", "force"),
        ("lab_tests", "battery"),
        ("field_tests", "battery"),
        ("srpe_diary", "srpe"),
    ):
        st = RunStats(source=source)
        # The run record is committed in its own transaction so that a failure
        # in the body still leaves a durable record that the run was attempted.
        with engine.begin() as conn:
            st.run_id = _open_run(conn, source)

        status, error = "success", None
        try:
            with engine.begin() as conn:
                known = (
                    load_roster(conn, data_dir / "athletes.csv")
                    if fn == "force"
                    else athlete_index(conn)
                )
                if fn == "force":
                    ingest_force_plate(conn, st, data_dir / "force_plate", known)
                elif fn == "battery":
                    ingest_battery(conn, st, data_dir / f"{source}.csv",
                                   known, load_catalog(conn))
                else:
                    ingest_srpe(conn, st, data_dir / "srpe_diary.csv", known)
        except Exception:
            status, error = "failed", traceback.format_exc(limit=4)

        with engine.begin() as conn:
            _close_run(conn, st, status, error)

        results.append(st)
        if verbose:
            warns = sum(1 for i in st.issues if i["severity"] == "warn")
            print(
                f"[{datetime.now():%H:%M:%S}] {source:16} run={st.run_id} {status:7} "
                f"read={st.read:5} loaded={st.loaded:5} rejected={st.rejected:3} warn={warns:3}"
            )
            if error:
                print(error)

    refresh_derived(verbose)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the athlete-data ingest pipeline.")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()
    run_pipeline(args.data_dir)


if __name__ == "__main__":
    main()
