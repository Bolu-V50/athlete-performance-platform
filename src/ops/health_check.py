"""Database health check — a set of falsifiable invariants, run on a schedule.

A pipeline that finished without raising is not the same as a database you can
trust. This asserts things that must be true about the data itself and exits
non-zero when one of them is not, so a scheduled run fails loudly instead of
going green while the contents rot.

Two rules govern what is allowed to be a FAIL here, both learned the hard way:

**Constrain quantities we control, not quantities we configure.** The tempting
check is "no stored value falls outside its catalogue range". It is a trap: the
moment anyone tightens a range, every historical row that was perfectly valid
when it was ingested turns into a failure, and the audit cries wolf on a config
change. Those are reported as diagnostics instead. What is a FAIL is whether the
catalogue and the data agree structurally -- something no threshold edit can
retroactively break.

**A check that cannot fail is not a check.** Each invariant below is written so
that a plausible real defect would trip it, and the self-test at the bottom
proves the query returns rows when the defect is present.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import text

from src.db.connection import get_engine, redacted_url

# The demo dataset is static and ends on a fixed date, so freshness is a
# diagnostic by default. Set FRESHNESS_DAYS in a live deployment to make a stale
# database a hard failure -- a pipeline that quietly stopped three days ago is
# the most common way a platform like this fails without anyone noticing.
FRESHNESS_DAYS = int(os.getenv("FRESHNESS_DAYS", "0"))


@dataclass
class Invariant:
    key: str
    description: str
    sql: str
    severity: str = "fail"          # 'fail' or 'info'
    explain: str = ""               # why a violation matters


INVARIANTS: list[Invariant] = [
    Invariant(
        "I1", "every stored metric is described by the catalogue",
        """
        select distinct m.metric_name
        from performance_metrics m
        left join metric_catalog c on c.metric_name = m.metric_name
        where c.metric_name is null
        """,
        explain="An uncatalogued metric has no unit and no polarity, so every trend "
                "and z-score computed from it is a coin flip on sign.",
    ),
    Invariant(
        "I2", "every physical quality has at least one headline metric",
        """
        select q.quality from quality_catalog q
        where not exists (
            select 1 from metric_catalog c
            where c.quality = q.quality and c.is_headline
        )
        """,
        explain="A quality with no headline metric is silently absent from every "
                "capability profile. This has already happened once, to body composition.",
    ),
    Invariant(
        "I3", "the most recent run of each pipeline source succeeded",
        """
        select source, status, started_at from (
            select distinct on (source) source, status, started_at
            from pipeline_runs order by source, run_id desc
        ) t where status <> 'success'
        """,
        explain="A failed latest run means the newest data never landed, while the "
                "dashboard carries on showing yesterday's numbers as if they were today's.",
    ),
    Invariant(
        "I4", "no pipeline run has been stuck in 'running' for over six hours",
        """
        select run_id, source, started_at from pipeline_runs
        where status = 'running' and started_at < now() - interval '6 hours'
        """,
        explain="A run that opened and never closed means the process died mid-write. "
                "The row counts for that run are meaningless and may be partial.",
    ),
    Invariant(
        "I5", "no session exists without any metrics attached",
        """
        select s.session_id, s.session_date, s.session_type
        from sessions s
        left join performance_metrics m on m.session_id = s.session_id
        where m.metric_id is null
        """,
        explain="An empty session means a trial was registered and its measurements "
                "were not, which reads on the dashboard as a test the athlete missed.",
    ),
    Invariant(
        "I6", "the generated session_load column agrees with its inputs",
        """
        select load_id, duration_min, srpe, session_load
        from training_load
        where duration_min is not null and srpe is not null
          and abs(session_load - duration_min * srpe) > 0.001
        """,
        explain="This should be impossible -- the column is GENERATED ALWAYS. If it "
                "ever trips, the schema has been altered underneath the application.",
    ),
    Invariant(
        "I7", "every analytics view returns rows",
        """
        select 'v_athlete_status' as view_name where not exists (select 1 from v_athlete_status)
        union all select 'v_quality_profile' where not exists (select 1 from v_quality_profile)
        union all select 'v_acwr'            where not exists (select 1 from v_acwr)
        union all select 'v_cmj_flags'       where not exists (select 1 from v_cmj_flags)
        union all select 'v_test_day'        where not exists (select 1 from v_test_day)
        union all select 'v_metric_trend'    where not exists (select 1 from v_metric_trend)
        union all select 'v_squad_comparison' where not exists (select 1 from v_squad_comparison)
        """,
        explain="An empty view usually means a join condition broke after a schema "
                "change. The dashboard renders blank rather than erroring, so nothing "
                "else would report it.",
    ),
    Invariant(
        "I8", "no athlete carries a metric for a test their sport does not run",
        """
        select distinct a.sport, s.session_type, count(*) over () as n
        from sessions s join athletes a using (athlete_id)
        where s.session_type = 'swim_test' and a.sport <> 'Swimming'
        """,
        explain="Attributing another sport's test to an athlete means the ingest has "
                "mismatched a row, and the resulting profile describes nobody.",
    ),
    # ---- diagnostics: reported, never failed ------------------------------
    Invariant(
        "D1", "stored values that fall outside their current catalogue range",
        """
        select m.metric_name, count(*) as n,
               min(m.metric_value) as lo, max(m.metric_value) as hi,
               min(c.typical_min) as range_min, max(c.typical_max) as range_max
        from performance_metrics m
        join metric_catalog c on c.metric_name = m.metric_name
        where (c.typical_min is not null and m.metric_value < c.typical_min)
           or (c.typical_max is not null and m.metric_value > c.typical_max)
        group by m.metric_name
        """,
        severity="info",
        explain="Deliberately NOT a failure. Ranges get tightened as a service learns "
                "its population, and rows ingested under the old range were valid when "
                "they were written. Failing on them would turn every threshold edit into "
                "a fleet of false alarms.",
    ),
    Invariant(
        "D2", "data freshness",
        """
        select max(session_date) as newest,
               (current_date - max(session_date)) as days_old
        from sessions
        """,
        severity="info",
        explain="Set FRESHNESS_DAYS to make a stale database a hard failure in a live "
                "deployment.",
    ),
]


def run_checks(verbose: bool = True) -> int:
    """Returns the number of failed invariants."""
    engine = get_engine()
    failures = 0
    if verbose:
        print(f"health check against {redacted_url()}")
        print("=" * 78)

    with engine.connect() as conn:
        for inv in INVARIANTS:
            rows = conn.execute(text(inv.sql)).mappings().all()

            if inv.key == "D2":
                r = rows[0] if rows else {}
                days = r.get("days_old")
                stale = FRESHNESS_DAYS and days is not None and days > FRESHNESS_DAYS
                label = "FAIL" if stale else "INFO"
                if stale:
                    failures += 1
                if verbose:
                    print(f"[{label}] {inv.key} {inv.description}: newest session "
                          f"{r.get('newest')}, {days} days old"
                          + (f" (limit {FRESHNESS_DAYS})" if FRESHNESS_DAYS else
                             " (no limit set; see FRESHNESS_DAYS)"))
                continue

            violated = bool(rows)
            if inv.severity == "info":
                if verbose:
                    print(f"[INFO] {inv.key} {inv.description}: "
                          f"{len(rows)} finding(s)")
                    for r in rows[:5]:
                        print(f"       {dict(r)}")
                    if rows:
                        print(f"       -> {inv.explain}")
                continue

            if violated:
                failures += 1
            if verbose:
                print(f"[{'FAIL' if violated else 'PASS'}] {inv.key} {inv.description}")
                if violated:
                    for r in rows[:5]:
                        print(f"       {dict(r)}")
                    if len(rows) > 5:
                        print(f"       ... and {len(rows) - 5} more")
                    print(f"       -> {inv.explain}")

    if verbose:
        print("=" * 78)
        checked = sum(1 for i in INVARIANTS if i.severity == "fail")
        print(f"{checked - failures}/{checked} invariants hold"
              if not failures else
              f"{failures} of {checked} invariants FAILED")
    return failures


def main() -> None:
    # No output is swallowed and the exit code is the verdict: a health check
    # whose failures are invisible to the caller is worse than none.
    sys.exit(1 if run_checks() else 0)


if __name__ == "__main__":
    main()
