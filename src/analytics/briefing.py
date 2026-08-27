"""Daily coach briefing: turn today's alert table into two or three sentences.

How the numbers are kept out of the model's hands
------------------------------------------------
Every number in the briefing is computed by SQL in ``src/db/views.sql``. The
language model receives those numbers already computed and is permitted to do
exactly one thing: choose which of them matter and phrase them. It does not
calculate, it does not decide who trains, and it does not see raw data.

Three mechanisms enforce that rather than merely asking for it:

1. **A closed fact block.** The model's entire view of the world is a rendered
   list of pre-computed values. There is nothing else in the context to reason
   from.
2. **A numeric guard.** Every number in the generated text is checked back
   against the fact block. A number the model invented -- including one it
   derived by doing arithmetic it was told not to do -- fails the check and the
   output is discarded.
3. **A deterministic fallback.** If the guard fails, the API is down, or no key
   is configured, the briefing is rendered from a template instead. The system
   never has a state where a coach sees nothing, and never a state where a coach
   sees an unverified number.

The third point is the one that matters operationally: the LLM is an optional
presentation layer over a system that is fully functional without it.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from dotenv import load_dotenv
from sqlalchemy import text

from src.db.connection import get_engine

load_dotenv()

# Words that mean the model has stopped describing the data and started
# programming the athlete. Naming where the numbers are flat is analysis;
# deciding what to do about it depends on the competition calendar, injury
# history and a hundred things this system cannot see.
PRESCRIPTION_WORDS = (
    "should ", "recommend", "prescrib", "increase the", "reduce the", "add a",
    "sets", "reps", "sessions per week", "rest day", "deload", "must ",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You write one short daily briefing for a strength and conditioning coach.
The coach reads it in five seconds between sessions, so it must be a sentence they
can act on, not a table read aloud.

Rules, all of them absolute:
- Use ONLY numbers that appear verbatim in the FACTS block. Copy them exactly.
- Do NOT calculate anything. No percentages, differences, averages or totals that
  are not already given to you.
- Name AT MOST TWO athletes: the one who needs attention most, and the next one
  only if it adds something. Summarise the rest as a count.
- At most two numbers per athlete. Prefer the percentage below baseline and the
  ACWR. Do not list every field you were given.
- 'sweet_spot' means the workload ratio is FINE. Never present it as a concern.
  Only 'caution' and 'high_risk' are load concerns.
- Do NOT recommend training changes, rest, load reductions or medical action.
  Report what the monitoring system observed; the coach decides what to do.
- Do NOT speculate about causes (illness, sleep, travel, injury).
- Do NOT state the date. The coach knows what day it is and the screen shows it.
- Two to three sentences of plain English. No bullet points, no headings, no
  jargon the coach has not asked for."""


# ---------------------------------------------------------------------------
# snapshot: everything the briefing is allowed to know
# ---------------------------------------------------------------------------
@dataclass
class AthleteRow:
    athlete_code: str
    squad: str
    baseline_status: str
    acwr_zone: str
    jump_height_m: float | None
    baseline_mean_m: float | None
    z_score: float | None
    pct_below_baseline: float | None
    acwr: float | None
    attention_rank: int


@dataclass
class Snapshot:
    as_of: date
    rows: list[AthleteRow]
    n_athletes: int
    n_flag: int
    n_watch: int
    n_load_concern: int
    rejected_today: int = 0
    notable: list[AthleteRow] = field(default_factory=list)


def collect_snapshot(limit_notable: int = 4) -> Snapshot:
    """Read the squad overview. Pure SQL -- no model involvement."""
    with get_engine().connect() as conn:
        from src.analytics.queries import ATTENTION_ORDER

        rows = list(conn.execute(text(f"select * from v_athlete_status {ATTENTION_ORDER}")))
        as_of = conn.execute(text("select max(session_date) from sessions")).scalar()
        # The latest run of each source, not a 24-hour sum. Re-running the
        # pipeline over the same files is normal and idempotent, so adding up
        # every run's rejections counts the same bad row once per run: twenty
        # re-runs turned ten genuine rejections into 3704 and made a clean
        # database look like a data-quality emergency.
        rejected = conn.execute(
            text(
                "select coalesce(sum(rows_rejected), 0) from ("
                "  select distinct on (source) source, rows_rejected"
                "  from pipeline_runs order by source, run_id desc"
                ") latest"
            )
        ).scalar()

    out: list[AthleteRow] = []
    for r in rows:
        h = float(r.jump_height_m) if r.jump_height_m is not None else None
        b = float(r.baseline_mean_m) if r.baseline_mean_m is not None else None
        pct = round((b - h) / b * 100, 1) if (h is not None and b) else None
        out.append(
            AthleteRow(
                athlete_code=r.athlete_code,
                squad=r.squad,
                baseline_status=r.baseline_status,
                acwr_zone=r.acwr_zone,
                jump_height_m=h,
                baseline_mean_m=b,
                z_score=float(r.z_score) if r.z_score is not None else None,
                pct_below_baseline=pct,
                acwr=float(r.acwr) if r.acwr is not None else None,
                attention_rank=r.attention_rank,
            )
        )

    return Snapshot(
        as_of=as_of,
        rows=out,
        n_athletes=len(out),
        n_flag=sum(1 for r in out if r.baseline_status == "flag"),
        n_watch=sum(1 for r in out if r.baseline_status == "watch"),
        n_load_concern=sum(1 for r in out if r.acwr_zone in ("caution", "high_risk")),
        rejected_today=int(rejected or 0),
        notable=[r for r in out if r.attention_rank <= 4][:limit_notable],
    )


def render_facts(s: Snapshot) -> str:
    """The model's entire view of the world."""
    lines = [
        f"DATE: {s.as_of}",
        f"SQUAD SIZE: {s.n_athletes} athletes monitored",
        f"COUNTS: {s.n_flag} flagged, {s.n_watch} on watch, {s.n_load_concern} with a load-ratio concern",
        "",
        "ATHLETES NEEDING ATTENTION (most urgent first):",
    ]
    if not s.notable:
        lines.append("  none - every athlete is within their normal range today")
    for r in s.notable:
        bits = [f"  {r.athlete_code} ({r.squad}):"]
        if r.jump_height_m is not None:
            bits.append(f"jump height {r.jump_height_m:.3f} m")
        if r.baseline_mean_m is not None:
            bits.append(f"vs 28-day baseline {r.baseline_mean_m:.3f} m")
        if r.pct_below_baseline is not None and r.pct_below_baseline > 0:
            bits.append(f"({r.pct_below_baseline:.1f}% below baseline)")
        if r.z_score is not None:
            bits.append(f"z-score {r.z_score:.2f}")
        bits.append(f"status {r.baseline_status}")
        if r.acwr is not None:
            bits.append(f"ACWR {r.acwr:.2f} ({r.acwr_zone})")
        lines.append(" ".join(bits))
    if s.rejected_today:
        lines.append("")
        lines.append(f"DATA QUALITY: {s.rejected_today} rows rejected by validation on the most recent ingest")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# numeric guard
# ---------------------------------------------------------------------------
_CODE = re.compile(r"\bATH-\d+\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

# Models routinely render "ATH-009" with a non-breaking hyphen or an en dash and
# in mixed case. Without normalising first, the code regex misses it and the
# guard then evaluates "009" as if it were a measurement.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")

# Numbers the briefing may legitimately mention that are not athlete data:
# the sweet-spot bounds, the flag threshold, and the ACWR window lengths.
CONTEXT_NUMBERS = {0.8, 1.3, 1.5, 7.0, 28.0, 1.0}


def _matches_at_written_precision(written: str, allowed: set[float]) -> bool:
    """Does a number the model wrote correspond to one the system computed?

    A fixed absolute tolerance is the obvious approach and it leaks. With a
    tolerance of 0.005, a model that quietly subtracted two values and wrote
    "0.048" was waved through because an unrelated r-squared of 0.05 sat within
    the window. Matching at the precision the model actually wrote closes it:
    "0.048" must be some computed value rounded to three places, which 0.05 is
    not, while "0.05" still matches a computed 0.048 because rounding to two
    places is a legitimate thing to do to it.
    """
    decimals = len(written.split(".")[1]) if "." in written else 0
    target = float(written)
    return any(round(a, decimals) == target for a in allowed)


def _numbers_in_text(text_block: str) -> set[float]:
    """Every number that literally appears in the fact block.

    The guard's real rule is traceability: a number the coach reads must be one
    the system computed. Restricting the allowed set to *values* was too narrow,
    because measurement and test names carry numbers of their own -- "505 change
    of direction", "30-15 IFT", "sum of 7 skinfolds", "10 m sprint". Those were
    scored as fabricated measurements and perfectly good summaries were thrown
    away. Anything printed in the facts is by definition traceable to them.
    """
    return {float(m) for m in _NUM.findall(text_block)}


def allowed_numbers(s: Snapshot) -> set[float]:
    vals: set[float] = set(CONTEXT_NUMBERS)
    vals |= _numbers_in_text(render_facts(s))
    vals.update({float(s.n_athletes), float(s.n_flag), float(s.n_watch),
                 float(s.n_load_concern), float(s.rejected_today)})
    vals.update(float(i) for i in range(0, s.n_athletes + 1))  # counts
    # The reporting date is itself a fact. Written in prose ("on the 24th",
    # "Aug 24") it survives the ISO-date strip and would otherwise be scored as
    # an unverifiable measurement -- a false positive that silently discards
    # perfectly good output.
    if s.as_of:
        vals.update({float(s.as_of.day), float(s.as_of.month), float(s.as_of.year)})
    for r in s.rows:
        for v in (r.jump_height_m, r.baseline_mean_m, r.z_score, r.pct_below_baseline, r.acwr):
            if v is not None:
                vals.add(round(float(v), 4))
                vals.add(abs(round(float(v), 4)))
    return vals


def numeric_guard(body: str, s: Snapshot) -> tuple[bool, list[float]]:
    """Check that every number in the text traces back to the fact block.

    Athlete codes and ISO dates are stripped first -- 'ATH-009' is an identifier,
    not a measurement. What remains must match a pre-computed value, otherwise
    the model either hallucinated it or did arithmetic it was told not to do.
    """
    normalised = body.translate(_DASHES)
    stripped = _ISO_DATE.sub(" ", _CODE.sub(" ", normalised))
    allowed = allowed_numbers(s)
    offenders = [
        float(w) for w in _NUM.findall(stripped)
        if not _matches_at_written_precision(w, allowed)
    ]
    return (not offenders), offenders


def guard_text(body: str, facts_text: str) -> tuple[bool, list[float]]:
    """Generic traceability guard: every number in `body` must appear in `facts_text`.

    The report generates prose in several separate slots rather than as one long
    piece. That is deliberate: a single long generation gives the model far more
    opportunity to invent a number, and one bad sentence would force the whole
    report back to a template. Each slot carries its own facts and its own guard,
    so a failure is contained to the paragraph that caused it.
    """
    allowed = set(CONTEXT_NUMBERS) | _numbers_in_text(facts_text)
    stripped = _ISO_DATE.sub(" ", _CODE.sub(" ", body.translate(_DASHES)))
    offenders = [
        float(w) for w in _NUM.findall(stripped)
        if not _matches_at_written_precision(w, allowed)
    ]
    return (not offenders), offenders


_RISE = ("rose", "risen", "increased", "climbed", "grew", "gained", "went up", "up from")
_FALL = ("fell", "fallen", "dropped", "decreased", "declined", "reduced", "went down", "down from")
_MOVE = re.compile(
    r"\b(" + "|".join(_RISE + _FALL) + r")\b[^.;]{0,40}?"
    r"(-?\d+(?:\.\d+)?)\s*[^\d.;]{0,18}?\bto\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def direction_contradictions(body: str) -> list[str]:
    """Catch a claim that contradicts the two numbers in the same sentence.

    The numeric guard checks that a figure is real. It cannot check what is said
    about it, and that is where a smaller model actually slips: given a fact line
    reading "fell from 0.360 to 0.324 m/s" it wrote "rose from 0.360 to 0.324".
    Every number was correct and the sentence was still wrong.

    This needs no knowledge of the metric. If the text says something rose and
    then names a smaller second number, the sentence contradicts itself, and the
    same in reverse. Metrics that improve by getting smaller are exactly where a
    model is most likely to make this mistake.
    """
    bad: list[str] = []
    for verb, a, b in _MOVE.findall(body):
        lo, hi = float(a), float(b)
        rising = verb.lower() in _RISE
        if rising and hi <= lo:
            bad.append(f"'{verb} from {a} to {b}' but {b} is not greater than {a}")
        elif not rising and hi >= lo:
            bad.append(f"'{verb} from {a} to {b}' but {b} is not less than {a}")
    return bad


_GENDERED = re.compile(r"\b(he|him|his|she|her|hers|himself|herself)\b", re.IGNORECASE)


def gendered_pronouns(body: str) -> list[str]:
    """Reject gendered pronouns in generated prose.

    The facts blocks carry an athlete code and no sex, by design -- the athletes
    table is de-identified and nothing downstream needs to know. Given no signal,
    the model guessed, and called a woman in the football squad "his". A report
    about a real person is not a place to be wrong about that, and the fix is not
    to hand the model the sex but to have it write about "the athlete".
    """
    return sorted({m.lower() for m in _GENDERED.findall(body)})


def contains_prescription(body: str) -> list[str]:
    """Words that mean the model has stopped describing and started programming."""
    lowered = body.lower()
    return [w for w in PRESCRIPTION_WORDS if w in lowered]


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class Backend(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


class GroqBackend:
    """Groq free tier. OpenAI-compatible chat completions."""

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.name = f"groq:{self.model}"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, user: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 600,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model.startswith("openai/gpt-oss"):
            # A reasoning model: without this the whole token budget is spent
            # thinking and `content` comes back empty.
            body["reasoning_effort"] = "low"

        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Groq sits behind Cloudflare, which rejects urllib's default
                # User-Agent with an opaque 1010.
                "User-Agent": "athlete-performance-platform/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode(), strict=False)
        return (payload["choices"][0]["message"].get("content") or "").strip()


class TemplateBackend:
    """Deterministic fallback. No API, no key, no network, no surprises."""

    name = "template"
    available = True

    def generate(self, system: str, user: str) -> str:  # noqa: ARG002
        raise NotImplementedError("TemplateBackend is rendered directly, not generated")


def template_briefing(s: Snapshot) -> str:
    if not s.notable:
        return (
            f"All {s.n_athletes} monitored athletes are within their individual "
            f"normal range as of {s.as_of}. No neuromuscular or load-ratio alerts today."
        )
    lead = s.notable[0]
    parts: list[str] = []
    bits = [f"{lead.athlete_code} ({lead.squad}) is the first to look at"]
    if lead.jump_height_m is not None and lead.baseline_mean_m is not None:
        bits.append(
            f": jump height {lead.jump_height_m:.3f} m against a 28-day baseline of "
            f"{lead.baseline_mean_m:.3f} m"
        )
        if lead.z_score is not None:
            bits.append(f" (z-score {lead.z_score:.2f})")
    if lead.acwr is not None and lead.acwr_zone in ("caution", "high_risk"):
        bits.append(f", with an acute:chronic workload ratio of {lead.acwr:.2f}")
    parts.append("".join(bits) + ".")

    others = s.notable[1:]
    if others:
        names = ", ".join(r.athlete_code for r in others)
        parts.append(f"Also showing a change from baseline: {names}.")
    parts.append(
        f"{s.n_flag} of {s.n_athletes} athletes are flagged and "
        f"{s.n_load_concern} carry a load-ratio concern as of {s.as_of}."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
@dataclass
class BriefingResult:
    text: str
    source: str                    # 'groq:<model>' or 'template'
    guard_passed: bool | None      # None when no model was involved
    guard_offenders: list[float] = field(default_factory=list)
    fallback_reason: str | None = None

    @property
    def is_model_generated(self) -> bool:
        return self.source != "template"


def daily_briefing(snapshot: Snapshot | None = None, backend: Backend | None = None) -> BriefingResult:
    s = snapshot or collect_snapshot()
    facts = render_facts(s)

    if backend is None:
        backend = GroqBackend()
    if isinstance(backend, TemplateBackend) or not getattr(backend, "available", True):
        return BriefingResult(
            text=template_briefing(s),
            source="template",
            guard_passed=None,
            fallback_reason="no API key configured" if not getattr(backend, "available", True) else None,
        )

    try:
        body = backend.generate(SYSTEM_PROMPT, f"FACTS\n-----\n{facts}\n\nWrite the briefing.")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
        return BriefingResult(
            text=template_briefing(s),
            source="template",
            guard_passed=None,
            fallback_reason=f"model call failed: {type(exc).__name__}: {exc}",
        )

    if not body:
        return BriefingResult(text=template_briefing(s), source="template",
                              guard_passed=None, fallback_reason="model returned empty content")

    ok, offenders = numeric_guard(body, s)
    if not ok:
        # The model produced a number that is not traceable to the fact block.
        # It does not get shown to a coach.
        return BriefingResult(
            text=template_briefing(s),
            source="template",
            guard_passed=False,
            guard_offenders=offenders,
            fallback_reason=f"numeric guard rejected unverifiable values: {offenders}",
        )

    return BriefingResult(text=body, source=backend.name, guard_passed=True)


def main() -> None:
    s = collect_snapshot()
    print("=" * 74)
    print(render_facts(s))
    print("=" * 74)
    r = daily_briefing(s)
    print(f"\nBRIEFING  [source: {r.source}"
          + (f", numeric guard: {'passed' if r.guard_passed else 'FAILED'}" if r.guard_passed is not None else "")
          + "]")
    if r.fallback_reason:
        print(f"  fallback: {r.fallback_reason}")
    print()
    print(r.text)


if __name__ == "__main__":
    main()


# ===========================================================================
# per-athlete narrative
# ---------------------------------------------------------------------------
# The squad briefing answers "who today". This answers "how is this athlete
# developing", which is a different question over a different time scale, and it
# is the one a coach asks in a review meeting rather than on the training floor.
#
# The governance is unchanged and non-negotiable: every trend, percentage and
# direction is computed in SQL by v_quality_profile, the model receives them
# already decided, and the same numeric guard rejects anything it invents.
#
# One deliberate extension: the athlete narrative is allowed to name the quality
# that has progressed least, because that is a statement about the data. It is
# still forbidden from prescribing training. Naming where the numbers are flat
# is analysis; deciding what to do about it is the coach's job and depends on
# the competition calendar, the athlete's history and a hundred things this
# system cannot see.
# ===========================================================================

ATHLETE_SYSTEM_PROMPT = """You write a short development summary about one athlete for their coach.

Rules, all of them absolute:
- Use ONLY numbers that appear verbatim in the FACTS block. Copy them exactly.
- Do NOT calculate anything: no differences, averages, rankings or totals that
  are not already given to you.
- Directions are already decided for you ('improving', 'stable', 'declining').
  Never contradict one, and never infer a direction that is not stated.
- A quality marked 'insufficient_data' has not been measured enough to judge.
  Say that it has not been assessed; do not describe it as stable.
- You MAY name the quality that has progressed least or is declining, and say it
  is the area the data points to for discussion. Do NOT prescribe training:
  no sets, sessions, exercises, volumes, loads or rest.
- Do NOT speculate about causes (injury, illness, sleep, motivation).
- Do NOT state the date.
- Three to four sentences of plain English for a coach, not a data scientist.
  No bullet points, no headings."""


@dataclass
class AthleteSnapshot:
    athlete_code: str
    squad: str
    window_start: date | None
    window_end: date | None
    qualities: list[dict[str, Any]] = field(default_factory=list)
    n_test_days: int = 0
    baseline_status: str = "no_data"
    acwr: float | None = None
    acwr_zone: str = "no_data"


def collect_athlete_snapshot(athlete_code: str) -> AthleteSnapshot:
    from src.analytics.queries import quality_profile, squad_status, test_days

    prof = quality_profile(athlete_code)
    days = test_days(athlete_code)
    status = squad_status()
    row = status[status["athlete_code"] == athlete_code]

    qualities: list[dict[str, Any]] = []
    for r in prof.itertuples(index=False):
        qualities.append(
            dict(
                quality=r.quality_name,
                metric=r.display_name,
                unit=r.unit,
                n_tests=int(r.n_tests),
                first_value=float(r.first_value),
                latest_value=float(r.latest_value),
                pct=float(r.pct_improvement_fitted) if r.pct_improvement_fitted is not None else None,
                sd=float(r.fitted_change_in_sd) if r.fitted_change_in_sd is not None else None,
                r2=float(r.trend_r2) if r.trend_r2 is not None else None,
                direction=r.direction,
            )
        )

    return AthleteSnapshot(
        athlete_code=athlete_code,
        squad=str(row["squad"].iloc[0]) if not row.empty else "",
        window_start=days["session_date"].min() if not days.empty else None,
        window_end=days["session_date"].max() if not days.empty else None,
        qualities=qualities,
        n_test_days=len(days),
        baseline_status=str(row["baseline_status"].iloc[0]) if not row.empty else "no_data",
        acwr=float(row["acwr"].iloc[0]) if not row.empty and row["acwr"].iloc[0] is not None else None,
        acwr_zone=str(row["acwr_zone"].iloc[0]) if not row.empty else "no_data",
    )


def render_athlete_facts(s: AthleteSnapshot) -> str:
    lines = [
        f"ATHLETE: {s.athlete_code} ({s.squad})",
        f"TEST DAYS IN WINDOW: {s.n_test_days}",
        "",
        "PHYSICAL QUALITIES (trend fitted across every test, not first vs last):",
    ]
    if not s.qualities:
        lines.append("  no physical testing on record")
    for q in s.qualities:
        if q["direction"] == "insufficient_data":
            lines.append(f"  {q['quality']} — {q['metric']}: only {q['n_tests']} tests, "
                         "not enough to judge a direction")
            continue
        conf = ""
        if q["r2"] is not None:
            conf = (" (scattered — the line explains little of the variation)"
                    if q["r2"] < 0.2 else "")
        lines.append(
            f"  {q['quality']} — {q['metric']}: {q['first_value']} to {q['latest_value']} "
            f"{q['unit']} over {q['n_tests']} tests, {q['pct']:+.1f}% "
            f"(positive always means better, whichever way the metric runs), "
            f"{q['sd']:+.2f} of this athlete's own SD, "
            f"direction {q['direction']}{conf}"
        )
    lines += [
        "",
        f"TODAY'S NEUROMUSCULAR STATUS: {s.baseline_status}",
        f"WORKLOAD RATIO: {s.acwr if s.acwr is not None else 'not available'} ({s.acwr_zone})",
    ]
    return "\n".join(lines)


def athlete_allowed_numbers(s: AthleteSnapshot) -> set[float]:
    vals: set[float] = set(CONTEXT_NUMBERS)
    vals |= _numbers_in_text(render_athlete_facts(s))
    vals.update(float(i) for i in range(0, max(s.n_test_days, 12) + 1))
    if s.acwr is not None:
        vals.add(round(float(s.acwr), 4))
    for q in s.qualities:
        for v in (q["first_value"], q["latest_value"], q["pct"], q["sd"], q["r2"], q["n_tests"]):
            if v is not None:
                vals.add(round(float(v), 4))
                vals.add(abs(round(float(v), 4)))
    if s.window_end:
        vals.update({float(s.window_end.day), float(s.window_end.month), float(s.window_end.year)})
    return vals


def athlete_numeric_guard(body: str, s: AthleteSnapshot) -> tuple[bool, list[float]]:
    stripped = _ISO_DATE.sub(" ", _CODE.sub(" ", body.translate(_DASHES)))
    allowed = athlete_allowed_numbers(s)
    offenders = [
        float(w) for w in _NUM.findall(stripped)
        if not _matches_at_written_precision(w, allowed)
    ]
    return (not offenders), offenders


def template_athlete_narrative(s: AthleteSnapshot) -> str:
    improving = [q for q in s.qualities if q["direction"] == "improving"]
    declining = [q for q in s.qualities if q["direction"] == "declining"]
    stable = [q for q in s.qualities if q["direction"] == "stable"]
    unknown = [q for q in s.qualities if q["direction"] == "insufficient_data"]

    parts: list[str] = []
    if improving:
        best = max(improving, key=lambda q: q["pct"] or 0)
        parts.append(
            f"{s.athlete_code} is improving in {best['quality'].lower()}: "
            f"{best['metric']} moved from {best['first_value']} to {best['latest_value']} "
            f"{best['unit']} across {best['n_tests']} tests ({best['pct']:+.1f}%)."
        )
    if declining:
        worst = min(declining, key=lambda q: q["pct"] or 0)
        parts.append(
            f"The area the data points to is {worst['quality'].lower()}: "
            f"{worst['metric']} is {worst['pct']:+.1f}% over the same window."
        )
    if stable and not declining:
        parts.append(f"{len(stable)} of the measured qualities are holding steady.")
    if unknown:
        parts.append(f"{len(unknown)} qualities have too few tests to judge.")
    if not parts:
        parts.append(f"{s.athlete_code} has no physical testing on record in this window.")
    return " ".join(parts)


def athlete_narrative(athlete_code: str, backend: Backend | None = None) -> BriefingResult:
    s = collect_athlete_snapshot(athlete_code)
    facts = render_athlete_facts(s)

    if backend is None:
        backend = GroqBackend()
    if isinstance(backend, TemplateBackend) or not getattr(backend, "available", True):
        return BriefingResult(text=template_athlete_narrative(s), source="template",
                              guard_passed=None, fallback_reason="no API key configured")

    try:
        body = backend.generate(
            ATHLETE_SYSTEM_PROMPT, f"FACTS\n-----\n{facts}\n\nWrite the summary."
        )
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back
        return BriefingResult(text=template_athlete_narrative(s), source="template",
                              guard_passed=None,
                              fallback_reason=f"model call failed: {type(exc).__name__}: {exc}")

    if not body:
        return BriefingResult(text=template_athlete_narrative(s), source="template",
                              guard_passed=None, fallback_reason="model returned empty content")

    lowered = body.lower()
    prescriptions = [w for w in PRESCRIPTION_WORDS if w in lowered]
    if prescriptions:
        # Naming where the numbers are flat is analysis. Telling a coach what to
        # program is not this system's job, and a model that starts doing it has
        # left the boundary the design depends on.
        return BriefingResult(
            text=template_athlete_narrative(s), source="template", guard_passed=False,
            fallback_reason=f"output strayed into training prescription: {prescriptions}",
        )

    ok, offenders = athlete_numeric_guard(body, s)
    if not ok:
        return BriefingResult(text=template_athlete_narrative(s), source="template",
                              guard_passed=False, guard_offenders=offenders,
                              fallback_reason=f"numeric guard rejected: {offenders}")

    return BriefingResult(text=body, source=backend.name, guard_passed=True)
