"""Athlete development report.

A deterministic scaffold with model-written prose in three named slots.

Asking the model for a whole report gives it a long unconstrained output in
which to invent a number, and one bad sentence would force the entire document
back to a template. Here every table, every figure and every caveat is assembled in Python
from SQL; the model writes an executive summary, an interpretation of the
quality trends, and a "what the data points to" section, each against its own
facts block and each checked by its own guard. A slot that fails falls back on
its own, and the rest of the report is unaffected.

The report deliberately reports and does not prescribe. It will name the quality
that has progressed least, because that is a statement about the data. It will
not tell a coach what to program, because that depends on the competition
calendar, injury history and a hundred things this system cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.analytics.briefing import (
    Backend,
    GroqBackend,
    TemplateBackend,
    contains_prescription,
    direction_contradictions,
    gendered_pronouns,
    guard_text,
)
from src.analytics.queries import _df

# Test counts below which a direction should not be asserted at all.
ADEQUACY = [(5, "adequate"), (3, "limited"), (0, "insufficient")]
# Within-athlete CV above which a small change is not readable.
CV_NOISY = 6.0


def _adequacy(n: int) -> str:
    for threshold, label in ADEQUACY:
        if n >= threshold:
            return label
    return "insufficient"


# ---------------------------------------------------------------------------
# deterministic data collection
# ---------------------------------------------------------------------------
@dataclass
class ReportSlot:
    text: str
    source: str
    guard_passed: bool | None = None
    fallback_reason: str | None = None


@dataclass
class AthleteReport:
    athlete_code: str
    squad: str
    sport: str
    window_start: date | None
    window_end: date | None
    n_test_days: int
    profile: pd.DataFrame
    reliability: pd.DataFrame
    squad_rows: pd.DataFrame
    recency: pd.DataFrame
    normative: pd.DataFrame
    readiness: dict[str, Any]
    load: dict[str, Any]
    flags: dict[str, int]
    quality_flags: list[str] = field(default_factory=list)
    slots: dict[str, ReportSlot] = field(default_factory=dict)


def collect_report_data(athlete_code: str) -> AthleteReport:
    meta = _df(
        "select athlete_code, squad, sport from athletes where athlete_code = :c", c=athlete_code
    )
    if meta.empty:
        raise ValueError(f"unknown athlete {athlete_code}")

    profile = _df(
        "select * from v_quality_profile where athlete_code = :c order by quality_order, display_name",
        c=athlete_code,
    )
    reliability = _df(
        "select * from v_metric_reliability where athlete_code = :c and is_headline "
        "order by quality_order, display_name",
        c=athlete_code,
    )
    squad_rows = _df(
        "select * from v_squad_comparison where athlete_code = :c order by quality_order, display_name",
        c=athlete_code,
    )
    recency = _df(
        "select * from v_recent_vs_prior where athlete_code = :c and is_headline "
        "and n_recent > 0 and n_prior > 0 order by quality_order, display_name",
        c=athlete_code,
    )
    normative = _df(
        "select * from v_normative_comparison where athlete_code = :c "
        "order by quality_order, display_name, population",
        c=athlete_code,
    )
    days = _df(
        "select min(session_date) lo, max(session_date) hi, count(distinct session_date) n "
        "from v_test_day where athlete_code = :c",
        c=athlete_code,
    ).iloc[0]

    status = _df("select * from v_athlete_status where athlete_code = :c", c=athlete_code)
    r = status.iloc[0] if not status.empty else None

    flag_counts = _df(
        "select baseline_status, count(*) n from v_cmj_flags where athlete_code = :c "
        "and session_date > (select max(session_date) from v_cmj_flags where athlete_code = :c) "
        "- interval '90 days' group by 1",
        c=athlete_code,
    )
    flags = {row.baseline_status: int(row.n) for row in flag_counts.itertuples(index=False)}

    load = _df(
        """
        select round(sum(v.session_load)::numeric, 0)            as total_load,
               round(avg(v.session_load)::numeric * 7, 0)        as mean_weekly_load,
               round(max(v.acwr)::numeric, 2)                    as peak_acwr,
               count(*) filter (where v.acwr_zone = 'high_risk') as days_high_risk,
               count(*) filter (where v.acwr_zone = 'caution')   as days_caution,
               count(*) filter (where v.acwr_zone = 'sweet_spot')as days_sweet,
               count(*)                                          as days_total
        from v_acwr v join athletes a using (athlete_id)
        where a.athlete_code = :c
          and v.date > (select max(date) from v_acwr) - interval '90 days'
        """,
        c=athlete_code,
    ).iloc[0]

    rejected = _df(
        "select count(*) n from data_quality_log where athlete_code = :c", c=athlete_code
    ).iloc[0]

    return AthleteReport(
        athlete_code=athlete_code,
        squad=str(meta["squad"].iloc[0]),
        sport=str(meta["sport"].iloc[0]),
        window_start=days.lo,
        window_end=days.hi,
        n_test_days=int(days.n or 0),
        profile=profile,
        reliability=reliability,
        squad_rows=squad_rows,
        recency=recency,
        normative=normative,
        readiness={
            "baseline_status": str(r.baseline_status) if r is not None else "no_data",
            "z_score": float(r.z_score) if r is not None and r.z_score is not None else None,
            "jump_height_m": float(r.jump_height_m) if r is not None and r.jump_height_m is not None else None,
            "baseline_mean_m": float(r.baseline_mean_m) if r is not None and r.baseline_mean_m is not None else None,
            "acwr": float(r.acwr) if r is not None and r.acwr is not None else None,
            "acwr_zone": str(r.acwr_zone) if r is not None else "no_data",
            "last_cmj_date": r.last_cmj_date if r is not None else None,
        },
        load={k: (float(v) if v is not None else None) for k, v in load.items()},
        flags=flags,
        quality_flags=[f"{int(rejected.n)} rows from this athlete were rejected by validation"]
        if int(rejected.n or 0)
        else [],
    )


# ---------------------------------------------------------------------------
# fact blocks — one per prose slot, each as small as the slot needs
# ---------------------------------------------------------------------------
def facts_summary(rep: AthleteReport) -> str:
    lines = [
        f"ATHLETE: {rep.athlete_code}, {rep.squad}, {rep.sport}",
        f"TESTING: {rep.n_test_days} test days on record",
        "",
        "QUALITY DIRECTIONS (fitted across every test):",
    ]
    for r in rep.profile.itertuples(index=False):
        if r.direction == "insufficient_data":
            lines.append(f"  {r.quality_name}: only {r.n_tests} tests, no direction asserted")
        else:
            lines.append(
                f"  {r.quality_name} ({r.display_name}): {r.direction}, "
                f"{float(r.pct_improvement_fitted):+.1f}% in the improving direction"
            )
    if not rep.normative.empty:
        lines += ["", "AGAINST PUBLISHED NORMS (only where a study reports a standard deviation):"]
        for r in rep.normative.itertuples(index=False):
            if r.z_vs_reference is None or pd.isna(r.z_vs_reference):
                continue
            lines.append(
                f"  {r.display_name}: {float(r.z_vs_reference):+.2f} SD from the published mean "
                f"for {r.population} ({r.standing.replace('_', ' ')})"
            )

    rd = rep.readiness
    lines += [
        "",
        f"TODAY: neuromuscular status {rd['baseline_status']}"
        + (f", z-score {rd['z_score']:+.2f}" if rd["z_score"] is not None else ""),
        f"WORKLOAD RATIO: {rd['acwr'] if rd['acwr'] is not None else 'not available'} ({rd['acwr_zone']})",
    ]
    return "\n".join(lines)


def facts_qualities(rep: AthleteReport) -> str:
    lines = ["QUALITY DETAIL. Percentages are already sign-corrected: positive always means better.", ""]
    cv = {r.display_name: r.cv_pct for r in rep.reliability.itertuples(index=False)}
    rec = {r.display_name: r for r in rep.recency.itertuples(index=False)}
    sq = {r.display_name: r for r in rep.squad_rows.itertuples(index=False)}

    for r in rep.profile.itertuples(index=False):
        # Spell out which way the raw number actually moved. Handed only a
        # sign-corrected "+12.9%", the model described a sum of skinfolds that
        # had fallen from 60.2 to 52.9 mm as "rising" -- every figure it quoted
        # was real, so no numeric check could catch it. The polarity is resolved
        # here instead of being left for the model to infer.
        moved = "rose" if float(r.latest_value) > float(r.first_value) else "fell"
        bits = [f"  {r.quality_name} — {r.display_name}:"]
        bits.append(
            f"{moved} from {r.first_value} to {r.latest_value} {r.unit} over {r.n_tests} tests"
        )
        if r.pct_improvement_fitted is not None:
            bits.append(
                f"which is a {abs(float(r.pct_improvement_fitted)):.1f}% "
                f"{'improvement' if float(r.pct_improvement_fitted) >= 0 else 'worsening'} "
                f"for this metric"
            )
        bits.append(f"direction {r.direction}")
        # Whether a change clears this athlete's own noise is a judgement, and
        # judgements belong in code. Left to the model it produced both
        # directions of error in one paragraph: a -7.0% trend against 5.8%
        # variation called "within normal noise", and a 2.6% variation called
        # "large". The numeric guard cannot catch either, because every figure
        # quoted was real -- only the claim about them was wrong.
        c = cv.get(r.display_name)
        if c is not None:
            trend = abs(float(r.pct_improvement_fitted or 0))
            verdict = (
                "LARGER than that variation, so the change is readable"
                if trend > float(c)
                else "SMALLER than that variation, so it is NOT evidence of a real change"
            )
            bits.append(
                f"this athlete's repeat variation on this test is {float(c):.1f}% and the "
                f"season trend is {verdict}"
            )
        if r.display_name in rec:
            x = rec[r.display_name]
            bits.append(
                f"last six weeks vs the six before: {float(x.recent_pct_change):+.1f}% "
                f"({x.n_prior} then {x.n_recent} tests)"
            )
        if r.display_name in sq:
            x = sq[r.display_name]
            bits.append(f"ranks {x.squad_rank} of {x.squad_n} in the squad, {float(x.z_vs_squad):+.2f} SD")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def facts_discussion(rep: AthleteReport) -> str:
    declining = rep.profile[rep.profile["direction"] == "declining"]
    thin = rep.profile[rep.profile["n_tests"] < 5]
    lines = ["WHAT THE NUMBERS SHOW", ""]
    if declining.empty:
        lines.append("  No quality is declining.")
    for r in declining.itertuples(index=False):
        lines.append(
            f"  DECLINING: {r.quality_name} ({r.display_name}), "
            f"{float(r.pct_improvement_fitted):+.1f}% over {r.n_tests} tests"
        )
    for r in thin.itertuples(index=False):
        lines.append(f"  THIN EVIDENCE: {r.quality_name} rests on only {r.n_tests} tests")
    ld = rep.load
    lines += [
        "",
        "TRAINING LOAD, LAST 90 DAYS:",
        f"  peak workload ratio {ld['peak_acwr']}, "
        f"{int(ld['days_high_risk'] or 0)} days in the high-risk band, "
        f"{int(ld['days_caution'] or 0)} in caution, {int(ld['days_sweet'] or 0)} in the sweet spot",
        "",
        "NEUROMUSCULAR FLAGS, LAST 90 DAYS:",
        f"  {rep.flags.get('flag', 0)} flagged sessions, {rep.flags.get('watch', 0)} on watch, "
        f"{rep.flags.get('normal', 0)} normal",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# prose slots
# ---------------------------------------------------------------------------
SLOT_PROMPTS = {
    "summary": """You write the executive summary of an athlete review, for their coach.

- Use ONLY numbers that appear verbatim in the FACTS block. Copy them exactly.
- Do NOT calculate anything.
- Directions are already decided ('improving', 'stable', 'declining'). Never
  contradict one and never infer one that is not stated. Several metrics improve
  by getting smaller, so never say a value rose or fell unless the facts say so.
- Do NOT recommend training: no sets, sessions, loads, rest or deloads.
- Do NOT speculate about causes.
- Write about "the athlete", never "he" or "she". You have not been told their
  sex and it is not relevant to any of these numbers.
- Quote at most three numbers. This is a summary, not a listing: a coach reads
  it first and the detail is in the tables below.
- Two to three sentences. Plain English. No headings, no bullets.""",
    "qualities": """You interpret an athlete's physical testing results for their coach.

A table of every one of these numbers is already printed directly above your
paragraph. Restating it is worthless. Your job is to say what it means.

- Do NOT walk through the qualities one by one. Pick the two or three that
  actually matter and say why.
- Do NOT restate first-to-latest value pairs. They are in the table. Refer to a
  change as an improvement or a worsening of N%, using the word the facts use.
- Quote at most four numbers in total, and only ones that appear verbatim in the
  FACTS block. Copy them exactly. Do NOT calculate anything.
- Whether a change clears the athlete's own repeat variation is already decided
  for you in the facts, in capitals. Never contradict it and never make that
  judgement yourself.
- The facts say whether each measurement ROSE or FELL, and separately whether
  that was an improvement or a worsening. Use their words. Several metrics
  improve by getting smaller, so never infer the direction from the percentage.
- Where the season trend and the last six weeks disagree, that is worth a
  sentence.
- Do NOT recommend training. Do NOT speculate about causes.
- One paragraph, three to five sentences. No headings, no bullets.""",
    "discussion": """You write the closing section of an athlete review, for their coach.

- Use ONLY numbers that appear verbatim in the FACTS block. Copy them exactly.
- Do NOT calculate anything.
- Name the quality the data points to, and say why in one clause.
- Where evidence is thin, say that the honest answer is more testing rather than
  a conclusion.
- Do NOT recommend training: no sets, sessions, exercises, loads, rest or
  deloads. You are describing what the data shows, not what to do about it.
- Two to four sentences. No headings, no bullets.""",
}


def _template_slot(name: str, rep: AthleteReport) -> str:
    prof = rep.profile
    improving = prof[prof["direction"] == "improving"]
    declining = prof[prof["direction"] == "declining"]

    if name == "summary":
        parts = [f"{rep.athlete_code} has {rep.n_test_days} test days on record."]
        if not improving.empty:
            parts.append(
                f"{len(improving)} qualities are improving, led by "
                f"{improving.iloc[0]['quality_name'].lower()}."
            )
        if not declining.empty:
            parts.append(
                f"{len(declining)} are declining, including "
                f"{declining.iloc[0]['quality_name'].lower()}."
            )
        parts.append(
            f"Current neuromuscular status is {rep.readiness['baseline_status']} and the "
            f"workload ratio is {rep.readiness['acwr']} ({rep.readiness['acwr_zone']})."
        )
        return " ".join(parts)

    if name == "qualities":
        noisy = rep.reliability[
            rep.reliability["cv_pct"].astype("float", errors="ignore") > CV_NOISY
        ]
        s = (
            f"Across {len(prof)} measured qualities, {len(improving)} are improving, "
            f"{len(declining)} declining and {len(prof) - len(improving) - len(declining)} "
            "showing no direction."
        )
        if not noisy.empty:
            s += (
                f" {len(noisy)} metrics carry repeat variation above {CV_NOISY:.0f}%, so small "
                "changes in those are inside this athlete's normal noise."
            )
        return s

    parts = []
    if not declining.empty:
        w = declining.iloc[0]
        parts.append(
            f"The area the data points to is {w['quality_name'].lower()}, "
            f"{float(w['pct_improvement_fitted']):+.1f}% over {int(w['n_tests'])} tests."
        )
    thin = prof[prof["n_tests"] < 5]
    if not thin.empty:
        parts.append(
            f"{len(thin)} qualities rest on fewer than 5 tests; the honest answer there is "
            "more testing rather than a conclusion."
        )
    if not parts:
        parts.append("No quality is declining and no evidence gap stands out.")
    return " ".join(parts)


def generate_slot(
    name: str, rep: AthleteReport, facts: str, backend: Backend | None, attempts: int = 2
) -> ReportSlot:
    """Generate one prose slot, retrying once if the guard rejects the output.

    The guard is a filter on a single generation, not a verdict on the model.
    Sampling means a slot can fail one run and pass the next on identical input,
    and permanently demoting it to the template for one unlucky draw throws away
    a good section for no reason. A retry costs about a fifth of a second. Two
    consecutive rejections are treated as a real problem with the request and the
    template stands, with the reason recorded.
    """
    if backend is None:
        backend = GroqBackend()
    if isinstance(backend, TemplateBackend) or not getattr(backend, "available", True):
        return ReportSlot(_template_slot(name, rep), "template", None, "no API key configured")

    prompt = f"FACTS\n-----\n{facts}\n\nWrite it."
    reasons: list[str] = []

    for _ in range(max(attempts, 1)):
        try:
            body = backend.generate(SLOT_PROMPTS[name], prompt)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"model call failed: {type(exc).__name__}: {exc}")
            break                      # a transport failure will not fix itself on retry
        if not body:
            reasons.append("model returned empty content")
            continue

        strayed = contains_prescription(body)
        if strayed:
            reasons.append(f"strayed into training prescription: {strayed}")
            continue

        pronouns = gendered_pronouns(body)
        if pronouns:
            reasons.append(f"used gendered pronouns the system never supplied: {pronouns}")
            continue

        contradictions = direction_contradictions(body)
        if contradictions:
            reasons.append(f"self-contradictory direction: {contradictions}")
            continue

        ok, offenders = guard_text(body, facts)
        if not ok:
            reasons.append(f"numeric guard rejected: {offenders}")
            continue

        return ReportSlot(body, backend.name, True)

    guard_verdict = None if reasons and reasons[-1].startswith("model call failed") else False
    return ReportSlot(_template_slot(name, rep), "template", guard_verdict,
                      " ; then ".join(reasons))


def build_report(athlete_code: str, backend: Backend | None = None) -> AthleteReport:
    rep = collect_report_data(athlete_code)
    for name, facts_fn in (
        ("summary", facts_summary),
        ("qualities", facts_qualities),
        ("discussion", facts_discussion),
    ):
        rep.slots[name] = generate_slot(name, rep, facts_fn(rep), backend)
    return rep


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _fmt(v, spec="g", dash="—"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else format(float(v), spec)


def render_markdown(rep: AthleteReport) -> str:
    rd, ld = rep.readiness, rep.load
    out: list[str] = [
        f"# Athlete development report — {rep.athlete_code}",
        "",
        f"**Squad** {rep.squad}  |  **Sport** {rep.sport}  |  "
        f"**Window** {rep.window_start} to {rep.window_end}  |  "
        f"**Test days** {rep.n_test_days}",
        "",
        "## 1. Summary",
        "",
        rep.slots["summary"].text,
        "",
        "## 2. Physical qualities",
        "",
        "| Quality | Measure | First | Latest | Unit | Season trend | Last 6 wk | Own variation | Squad | Direction | Evidence |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    cv = {r.display_name: r.cv_pct for r in rep.reliability.itertuples(index=False)}
    rec = {r.display_name: r for r in rep.recency.itertuples(index=False)}
    sq = {r.display_name: r for r in rep.squad_rows.itertuples(index=False)}

    for r in rep.profile.itertuples(index=False):
        c = cv.get(r.display_name)
        x = rec.get(r.display_name)
        s = sq.get(r.display_name)
        out.append(
            f"| {r.quality_name} | {r.display_name} | {_fmt(r.first_value)} | "
            f"{_fmt(r.latest_value)} | {r.unit} | "
            f"{_fmt(r.pct_improvement_fitted, '+.1f')}% | "
            f"{(_fmt(x.recent_pct_change, '+.1f') + '%') if x is not None else '—'} | "
            f"{(_fmt(c, '.1f') + '%') if c is not None else '—'} | "
            f"{(str(s.squad_rank) + '/' + str(s.squad_n)) if s is not None else '—'} | "
            f"{r.direction} | {_adequacy(int(r.n_tests))} ({int(r.n_tests)}) |"
        )

    out += ["", rep.slots["qualities"].text, ""]

    if not rep.normative.empty:
        out += [
            "## 3. Against published normative data",
            "",
            "| Measure | This athlete | Reference mean | Spread | n | z | Standing | Reference population | Source |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rep.normative.itertuples(index=False):
            if r.spread_type == "sd":
                spread = f"± {float(r.reference_sd):g} SD"
            elif r.spread_type == "ci95":
                spread = f"95% CI {float(r.reference_low):g}–{float(r.reference_high):g}"
            elif r.spread_type == "range":
                spread = f"range {float(r.reference_low):g}–{float(r.reference_high):g}"
            else:
                spread = "—"
            z = "—" if r.z_vs_reference is None or pd.isna(r.z_vs_reference) \
                else f"{float(r.z_vs_reference):+.2f}"
            ident = f"[{r.study_key} {r.reference_year}](https://doi.org/{r.doi})" if r.doi \
                else f"{r.study_key} {r.reference_year}"
            out.append(
                f"| {r.display_name} | {_fmt(r.athlete_value)} {r.unit} | "
                f"{_fmt(r.reference_mean)} {r.unit} | {spread} | {r.reference_n} | {z} | "
                f"{str(r.standing).replace('_', ' ')} | {r.population} | {ident} |"
            )
        notes = [x for x in rep.normative["protocol_note"].dropna().unique() if x]
        if notes:
            out += ["", "**Protocol differences that affect these comparisons:**", ""]
            out += [f"- {x}" for x in notes]
        out += [
            "",
            "A z-score is shown only where the source published a standard deviation. A 95% "
            "confidence interval describes uncertainty about the mean rather than the spread of "
            "athletes, and dividing by it would place an athlete several standard deviations from "
            "normal when they are a fraction of one.",
            "",
        ]

    out += [
        "## 4. Current readiness",
        "",
        f"- Neuromuscular status: **{rd['baseline_status']}**"
        + (f" (z-score {rd['z_score']:+.2f} against the athlete's own 28-day baseline)"
           if rd["z_score"] is not None else ""),
        f"- Latest countermovement jump: {_fmt(rd['jump_height_m'], '.3f')} m against a baseline of "
        f"{_fmt(rd['baseline_mean_m'], '.3f')} m, tested {rd['last_cmj_date']}",
        f"- Acute:chronic workload ratio: **{_fmt(rd['acwr'], '.2f')}** ({rd['acwr_zone']})",
        "",
        "## 5. Training load, last 90 days",
        "",
        f"- Total session load: {_fmt(ld['total_load'], '.0f')} AU "
        f"(mean {_fmt(ld['mean_weekly_load'], '.0f')} AU per week)",
        f"- Peak workload ratio reached: {_fmt(ld['peak_acwr'], '.2f')}",
        f"- Days by band: {int(ld['days_sweet'] or 0)} sweet spot, "
        f"{int(ld['days_caution'] or 0)} caution, {int(ld['days_high_risk'] or 0)} high risk",
        f"- Neuromuscular flags in the window: {rep.flags.get('flag', 0)} flagged, "
        f"{rep.flags.get('watch', 0)} on watch, {rep.flags.get('normal', 0)} normal",
        "",
        "## 6. What the data points to",
        "",
        rep.slots["discussion"].text,
        "",
        "## 7. How to read this, and what it cannot tell you",
        "",
        "- **Trends are fitted across every test, not first versus latest.** Two endpoints carry "
        "the full test-retest error of both days and can invert the direction of a real trend.",
        "- **Percentages are sign-corrected**: positive always means the athlete got better, "
        "whether the metric improves by rising or by falling.",
        "- **Own variation is this athlete's coefficient of variation** across their own tests. "
        f"A change smaller than it is not readable. Above {CV_NOISY:.0f}% treat small movements "
        "as noise.",
        "- **Squad rank compares within the training group only.** A swimmer and a basketballer "
        "share no normative band, and a rank across both would describe the roster rather than "
        "the athlete.",
        "- **The workload ratio is a discussion starter, not a rule.** The 0.80–1.30 band comes "
        "from team-sport literature and its use as a threshold is contested.",
        "- **Session RPE is self-reported** and is not comparable between athletes.",
        "- **Every published comparison traces to a real paper**, recorded with its DOI or "
        "PMID and the date it was verified. Where no study in the library matches the athlete's "
        "sport and sex, the comparison is left empty rather than filled with a population that "
        "does not apply.",
        "- **Protocols must match before numbers do.** A countermovement jump without arm swing "
        "is several centimetres lower than one with; a 10 m sprint time depends on the starting "
        "position and gate height.",
        "- This report describes what was measured. It does not prescribe training.",
    ]
    if rep.quality_flags:
        out += ["", "**Data quality:** " + "; ".join(rep.quality_flags) + "."]

    prov = ", ".join(
        f"{k}: {v.source}"
        + ("" if v.guard_passed is None else f" (guard {'passed' if v.guard_passed else 'failed'})")
        for k, v in rep.slots.items()
    )
    out += ["", "---", "", f"*Prose sections — {prov}. Every number in this report is computed in "
            "SQL; the model selects and phrases them and is checked against the figures it was given.*"]
    return "\n".join(out)
