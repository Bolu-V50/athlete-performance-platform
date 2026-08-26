"""Coach-facing monitoring dashboard.

Design constraint, stated so it can be checked: a coach walking between sessions
must be able to answer "who do I need to look at today, and why" in about five
seconds, without scrolling and without knowing what a z-score is. Everything on
the first screen serves that question; the analysis detail is below it, for the
people who want it.

The dashboard computes nothing. Every number is read from a view in
src/db/views.sql, so the screen and the database cannot disagree.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit Cloud supplies credentials through st.secrets; the rest of the code
# reads os.environ, so bridge them before anything imports the engine.
for key in ("SUPABASE_DB_URL", "GROQ_API_KEY", "GROQ_MODEL"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from src.analytics import queries as q  # noqa: E402
from src.analytics.briefing import collect_snapshot, daily_briefing  # noqa: E402
from src.signal_processing.cmj import analyse_cmj  # noqa: E402
from src.signal_processing.plots import plot_cmj  # noqa: E402

# --- palette (validated; see the dataviz reference) ------------------------
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#dcdad4"
STATUS = {           # status colours are reserved and always ship with a label
    "flag":        ("#d03b3b", "🔴", "Flag"),
    "watch":       ("#fab219", "🟡", "Watch"),
    "normal":      ("#0ca30c", "🟢", "Normal"),
    "no_baseline": ("#52514e", "⚪", "No baseline"),
    "no_data":     ("#52514e", "⚪", "No data"),
}
ZONE = {
    "high_risk":            ("#d03b3b", "🔴", "High risk"),
    "caution":              ("#ec835a", "🟠", "Caution"),
    "sweet_spot":           ("#0ca30c", "🟢", "Sweet spot"),
    "undertrained":         ("#fab219", "🟡", "Undertrained"),
    "insufficient_history": ("#52514e", "⚪", "Not enough history"),
    "no_data":              ("#52514e", "⚪", "No data"),
}

st.set_page_config(
    page_title="Athlete Monitoring",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


def theme_name() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


# ---------------------------------------------------------------------------
# cached data access
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_status() -> pd.DataFrame:
    return q.squad_status()


@st.cache_data(ttl=300, show_spinner=False)
def load_window():
    return q.data_window()


@st.cache_data(ttl=300, show_spinner=False)
def load_cmj(code, lo, hi) -> pd.DataFrame:
    return q.cmj_series(code, lo, hi)


@st.cache_data(ttl=300, show_spinner=False)
def load_acwr(code, lo, hi) -> pd.DataFrame:
    return q.acwr_series(code, lo, hi)


@st.cache_data(ttl=900, show_spinner=False)
def load_briefing():
    s = collect_snapshot()
    r = daily_briefing(s)
    return s, r


@st.cache_data(ttl=300, show_spinner=False)
def load_ops():
    return q.recent_runs(8), q.recent_rejections(30)


@st.cache_data(ttl=3600, show_spinner=False)
def analyse_trace(path_str: str):
    df = pd.read_csv(path_str)
    return analyse_cmj(df, time_col="time_s",
                       force_cols=[c for c in df.columns if c.startswith("Fz")])


def chip(mapping: dict, key: str) -> str:
    colour, icon, label = mapping.get(key, mapping.get("no_data"))
    return f"{icon} {label}"


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
try:
    status = load_status()
    lo_all, hi_all = load_window()
except Exception as exc:
    st.error(
        "Could not reach the database. Set SUPABASE_DB_URL in .env (locally) or "
        f"in the app's secrets (when deployed).\n\n`{type(exc).__name__}: {exc}`"
    )
    st.stop()

st.sidebar.title("Filters")
squad_options = sorted(status["squad"].dropna().unique().tolist())
chosen_squads = st.sidebar.multiselect("Squad", squad_options, default=squad_options)
view = status[status["squad"].isin(chosen_squads)] if chosen_squads else status

athlete_options = view["athlete_code"].tolist()
if not athlete_options:
    st.sidebar.warning("No athletes in the selected squads.")
    st.stop()
chosen_athlete = st.sidebar.selectbox(
    "Athlete detail", athlete_options,
    help="The squad overview above always shows every selected squad; this picks whose history to draw.",
)

default_lo = max(lo_all, hi_all - timedelta(days=56))
date_lo, date_hi = st.sidebar.date_input(
    "Date range", value=(default_lo, hi_all), min_value=lo_all, max_value=hi_all
)
st.sidebar.divider()
st.sidebar.caption(
    f"Data window {lo_all} to {hi_all}. All values are read from database views; "
    "the dashboard does not recompute anything."
)

# ---------------------------------------------------------------------------
# header + briefing
# ---------------------------------------------------------------------------
st.title("Athlete monitoring")
st.caption(f"Neuromuscular readiness and training load · latest testing day {hi_all}")

snapshot, briefing = load_briefing()
badge = (
    f"generated by {briefing.source}, numeric guard passed"
    if briefing.guard_passed
    else f"deterministic template ({briefing.fallback_reason or 'no model configured'})"
)
st.info(f"**Today's briefing** — {briefing.text}", icon="📋")
st.caption(f"Briefing source: {badge}. Every number is computed in SQL; the model only phrases them.")

# --- stat tiles ------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Athletes monitored", snapshot.n_athletes)
c2.metric("Flagged today", snapshot.n_flag, help="CMJ at or below 1.5 SD under the athlete's own 28-day baseline")
c3.metric("On watch", snapshot.n_watch, help="Between 1.0 and 1.5 SD below baseline")
c4.metric("Load concerns", snapshot.n_load_concern, help="ACWR above 1.30")
c5.metric("Rows rejected (24 h)", snapshot.rejected_today, help="Caught by pipeline validation and excluded")

# ---------------------------------------------------------------------------
# attention queue — the five-second answer
# ---------------------------------------------------------------------------
st.subheader("Who needs attention")
attention = view[view["attention_rank"] <= 4].copy()

if attention.empty:
    st.success("No athlete in the selected squads is outside their normal range today.", icon="✅")
else:
    table = pd.DataFrame({
        "Athlete": attention["athlete_code"],
        "Squad": attention["squad"],
        "Neuromuscular": [chip(STATUS, s) for s in attention["baseline_status"]],
        "Jump (cm)": (attention["jump_height_m"].astype(float) * 100).map("{:.1f}".format),
        "Baseline (cm)": (attention["baseline_mean_m"].astype(float) * 100).map("{:.1f}".format),
        "z-score": attention["z_score"].astype(float).map("{:+.2f}".format),
        "Workload": [chip(ZONE, z) for z in attention["acwr_zone"]],
        "ACWR": attention["acwr"].astype(float).map("{:.2f}".format),
        "Last tested": attention["last_cmj_date"],
    })
    st.dataframe(table, hide_index=True, width='stretch')
    st.caption(
        "Ordered by urgency: a neuromuscular flag outranks a workload warning, because the jump "
        "is a measurement of the athlete whereas ACWR is an inference about accumulated exposure."
    )

with st.expander("Full squad"):
    full = pd.DataFrame({
        "Athlete": view["athlete_code"],
        "Squad": view["squad"],
        "Sport": view["sport"],
        "Neuromuscular": [chip(STATUS, s) for s in view["baseline_status"]],
        "Jump (cm)": (view["jump_height_m"].astype(float) * 100).map("{:.1f}".format),
        "z-score": view["z_score"].astype(float).map("{:+.2f}".format),
        "Workload": [chip(ZONE, z) for z in view["acwr_zone"]],
        "ACWR": view["acwr"].astype(float).map("{:.2f}".format),
    })
    st.dataframe(full, hide_index=True, width='stretch')

# ---------------------------------------------------------------------------
# athlete detail
# ---------------------------------------------------------------------------
st.divider()
row = status[status["athlete_code"] == chosen_athlete].iloc[0]
st.subheader(f"{chosen_athlete} · {row.squad}")

cmj = load_cmj(chosen_athlete, date_lo, date_hi)
acwr = load_acwr(chosen_athlete, date_lo, date_hi)

left, right = st.columns([3, 2])

# --- CMJ trend with baseline band -----------------------------------------
with left:
    st.markdown("**Countermovement jump vs individual baseline**")
    if cmj.empty:
        st.info("No CMJ tests in this date range.")
    else:
        d = cmj.copy()
        for c in ("jump_height_m", "baseline_mean", "baseline_sd"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["jump_cm"] = d["jump_height_m"] * 100
        d["base_cm"] = d["baseline_mean"] * 100
        d["sd_cm"] = d["baseline_sd"] * 100
        d["band_lo"] = d["base_cm"] - d["sd_cm"]
        d["band_hi"] = d["base_cm"] + d["sd_cm"]
        d["flag_line"] = d["base_cm"] - 1.5 * d["sd_cm"]
        d["Status"] = d["baseline_status"].map(lambda s: STATUS.get(s, STATUS["no_data"])[2])

        # Weekday names are noise across a multi-week range.
        x = alt.X("session_date:T", title=None,
                  axis=alt.Axis(grid=False, domainColor=GRID, format="%b %d"))

        # The scale must contain the flag threshold and every point. Letting it
        # fit the band alone pushed the one flagged trial -- the whole reason
        # the chart exists -- to the very edge of the plot.
        y_min = float(min(d["jump_cm"].min(), d["flag_line"].min(skipna=True) or d["jump_cm"].min()))
        y_max = float(max(d["jump_cm"].max(), d["band_hi"].max(skipna=True) or d["jump_cm"].max()))
        pad = max((y_max - y_min) * 0.16, 0.8)
        base_chart = alt.Chart(d)

        band = base_chart.mark_area(opacity=0.16, color=BLUE).encode(
            x=x, y=alt.Y("band_lo:Q", title="cm",
                         scale=alt.Scale(domain=[y_min - pad, y_max + pad], nice=False),
                         axis=alt.Axis(gridColor=GRID)),
            y2="band_hi:Q",
        )
        mean_line = base_chart.mark_line(color=MUTED, strokeDash=[5, 4], size=1).encode(x=x, y="base_cm:Q")
        flag_line = base_chart.mark_line(color=STATUS["flag"][0], strokeDash=[2, 3], size=1,
                                         opacity=0.7).encode(x=x, y="flag_line:Q")
        trend = base_chart.mark_line(color=BLUE, size=2).encode(x=x, y="jump_cm:Q")

        # Size alone could not make the flagged trial read as the most important
        # mark on the chart -- in Vega-Lite `size` is area, and a triangle spends
        # far less ink than a circle of equal area. A ring behind the marker
        # settles it, and is redundant with colour, shape and the legend.
        halo = alt.Chart(d[d["baseline_status"] == "flag"]).mark_point(
            shape="circle", filled=False, size=430,
            stroke=STATUS["flag"][0], strokeWidth=2.2, opacity=0.85,
        ).encode(x=x, y="jump_cm:Q")
        pts = base_chart.mark_point(filled=True, stroke="white", strokeWidth=1.2).encode(
            x=x, y="jump_cm:Q",
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Normal", "Watch", "Flag", "No baseline"],
                    range=[STATUS["normal"][0], STATUS["watch"][0],
                           STATUS["flag"][0], STATUS["no_baseline"][0]],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            shape=alt.Shape(
                "Status:N",
                scale=alt.Scale(domain=["Normal", "Watch", "Flag", "No baseline"],
                                range=["circle", "diamond", "triangle-down", "square"]),
                legend=None,
            ),
            # A triangle at the same nominal size draws far less ink than a
            # circle, so the most serious marker was also the faintest. Size
            # now rises with severity instead of fighting it.
            size=alt.Size(
                "Status:N",
                scale=alt.Scale(domain=["Normal", "Watch", "Flag", "No baseline"],
                                range=[85, 160, 300, 85]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("session_date:T", title="Date"),
                alt.Tooltip("jump_cm:Q", title="Jump (cm)", format=".1f"),
                alt.Tooltip("base_cm:Q", title="28-day baseline (cm)", format=".1f"),
                alt.Tooltip("z_score:Q", title="z-score", format=".2f"),
                alt.Tooltip("Status:N", title="Status"),
            ],
        )
        st.altair_chart(
            (band + mean_line + flag_line + trend + halo + pts).properties(height=300)
                .configure_view(strokeWidth=0)
                .configure_axis(titlePadding=8, labelPadding=4),
            width='stretch',
        )
        st.caption(
            "Shaded band is the athlete's own 28-day rolling mean ± 1 SD; the dashed red line is the "
            "−1.5 SD flag threshold. The baseline excludes the current day, so today's value cannot "
            "explain away its own deviation."
        )

# --- ACWR trend ------------------------------------------------------------
with right:
    st.markdown("**Acute:chronic workload ratio (EWMA 7:28)**")
    if acwr.empty or acwr["acwr"].isna().all():
        st.info("Not enough load history in this date range.")
    else:
        a = acwr.copy()
        a["acwr"] = pd.to_numeric(a["acwr"], errors="coerce")
        a = a.dropna(subset=["acwr"])
        a["Zone"] = a["acwr_zone"].map(lambda z: ZONE.get(z, ZONE["no_data"])[2])

        # A scale fitted to the data alone can crop the sweet-spot band, which
        # is the very reference the line is being judged against.
        y_lo = min(0.75, float(a["acwr"].min()) - 0.05)
        y_hi = max(1.40, float(a["acwr"].max()) + 0.05)
        sweet = alt.Chart(pd.DataFrame({"lo": [0.8], "hi": [1.3]})).mark_rect(
            opacity=0.13, color=STATUS["normal"][0]
        ).encode(y="lo:Q", y2="hi:Q")
        xa = alt.X("date:T", title=None,
                   axis=alt.Axis(grid=False, domainColor=GRID, format="%b %d"))
        line = alt.Chart(a).mark_line(color=ORANGE, size=2).encode(
            x=xa,
            y=alt.Y("acwr:Q", title="ratio",
                    scale=alt.Scale(domain=[y_lo, y_hi], nice=False, clamp=True),
                    axis=alt.Axis(gridColor=GRID)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("acwr:Q", title="ACWR", format=".2f"),
                alt.Tooltip("acute_load:Q", title="Acute (7 d EWMA)", format=".0f"),
                alt.Tooltip("chronic_load:Q", title="Chronic (28 d EWMA)", format=".0f"),
                alt.Tooltip("Zone:N", title="Zone"),
            ],
        )
        st.altair_chart((sweet + line).properties(height=300).configure_view(strokeWidth=0)
                        .configure_axis(titlePadding=8, labelPadding=4),
                        width='stretch')
        cur = row.acwr_zone
        colour, icon, label = ZONE.get(cur, ZONE["no_data"])
        acwr_txt = f"{float(row.acwr):.2f}" if row.acwr is not None else "—"
        st.markdown(f"Current zone: {icon} **{label}** · ACWR {acwr_txt}")
        st.caption(
            "Green band is the 0.80–1.30 heuristic from the team-sport literature. It is a "
            "conversation starter, not a rule: the evidence for a hard threshold is contested."
        )

# --- single trial: the raw waveform ---------------------------------------
st.markdown("**Single trial · force-time curve**")
trace_dates = q.available_trace_dates(chosen_athlete)
if not trace_dates:
    st.info(
        "No raw force-plate file is available for this athlete. Only the most recent trial per "
        "athlete ships with the repository; run `python scripts/generate_synthetic_data.py` to "
        "regenerate the full set."
    )
else:
    ingested = q.ingested_session_dates(chosen_athlete)

    def label(d: str) -> str:
        return d if d in ingested else f"{d}  ·  not in database"

    picked = st.selectbox(
        "Trial", trace_dates, key="trial_pick", format_func=label,
        help="Dates with a raw waveform file. A trial can have a file on disk and still have "
             "been excluded by pipeline validation.",
    )
    path = q.find_trace(chosen_athlete, picked)
    if path is None:
        st.info("Raw trace not found on disk.")
    else:
        try:
            result = analyse_trace(str(path))
            rejects = [f for f in result.quality_flags if f.startswith("REJECT")]

            # A trial the pipeline threw away must never be presented as a
            # measurement. The waveform is still drawn -- inspecting a bad trace
            # is exactly how a practitioner works out what went wrong with the
            # plate -- but its numbers are not headline numbers.
            if not result.is_valid:
                st.error(
                    "**This trial was rejected by pipeline validation and is not in the "
                    "database.** " + " ".join(rejects),
                    icon="🚫",
                )
                st.caption(
                    "The waveform is shown so the fault can be diagnosed. The values below are "
                    "what the algorithm computed from a bad trace; they are not a measurement "
                    "of this athlete."
                )
                with st.expander("Values computed from the rejected trace"):
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Jump height", f"{result.jump_height_m * 100:.1f} cm")
                    r2.metric("Peak force", f"{result.peak_force_bw:.2f} ×BW")
                    r3.metric("Body weight", f"{result.body_mass_kg:.1f} kg")
            else:
                if picked not in ingested:
                    st.warning(
                        "This trial passes validation but has not been ingested. Run the pipeline "
                        "to load it.",
                        icon="⚠️",
                    )
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Jump height", f"{result.jump_height_m * 100:.1f} cm")
                m2.metric("RSI-mod", f"{result.rsi_mod:.2f}")
                m3.metric("Peak force", f"{result.peak_force_bw:.2f} ×BW")
                m4.metric("Peak power", f"{result.peak_power_w_kg:.0f} W/kg")
                m5.metric("Contraction", f"{result.contraction_time_s * 1000:.0f} ms")

            fig = plot_cmj(result, title=f"{chosen_athlete} · {picked}", theme=theme_name())
            st.pyplot(fig, width='stretch')

            warns = [f for f in result.quality_flags if not f.startswith("REJECT")]
            if warns:
                st.warning(" · ".join(warns), icon="⚠️")
            st.caption(
                "Computed live from the raw dual-plate waveform, not read back from the database — "
                "this is the same function the ingest pipeline runs, so the dashboard and the "
                "pipeline cannot disagree about whether a trial is valid."
            )
        except Exception as exc:
            st.error(f"Could not analyse this trace: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# pipeline health
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Pipeline health and data quality"):
    runs, rejects = load_ops()
    st.markdown("**Recent ingest runs**")
    st.dataframe(runs, hide_index=True, width='stretch')
    st.markdown("**Rows excluded by validation**")
    if rejects.empty:
        st.caption("Nothing has been rejected.")
    else:
        st.dataframe(rejects, hide_index=True, width='stretch')
    st.caption(
        "Excluded rows are recorded rather than dropped silently, so a practitioner can see which "
        "trials were removed and disagree with the threshold that removed them."
    )
