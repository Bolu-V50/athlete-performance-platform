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
from src.analytics.report import build_report, render_markdown  # noqa: E402
from src.analytics.briefing import (  # noqa: E402
    athlete_narrative,
    collect_snapshot,
    daily_briefing,
)
from src.signal_processing.cmj import analyse_cmj  # noqa: E402
from src.signal_processing.plots import plot_cmj  # noqa: E402

# --- palette (validated; see the dataviz reference) ------------------------
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#dcdad4"
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# Status is carried by a coloured badge with its label inside it, so the colour
# never has to be read on its own. Streamlit's badge colours are used rather
# than emoji: a row of coloured circles reads as a placeholder.
STATUS = {
    "flag":        ("red", "Flag"),
    "watch":       ("orange", "Watch"),
    "normal":      ("green", "Normal"),
    "no_baseline": ("gray", "No baseline"),
    "no_data":     ("gray", "No data"),
}
ZONE = {
    "high_risk":            ("red", "High risk"),
    "caution":              ("orange", "Caution"),
    "sweet_spot":           ("green", "Sweet spot"),
    "undertrained":         ("yellow", "Undertrained"),
    "insufficient_history": ("gray", "Building history"),
    "no_data":              ("gray", "No data"),
}
DIRECTION = {
    "improving":         ("green", "Improving"),
    "stable":            ("gray", "Stable"),
    "declining":         ("red", "Declining"),
    "insufficient_data": ("gray", "Too few tests"),
    "no_data":           ("gray", "No data"),
}
STANDING = {
    "above_reference":  ("green", "Above"),
    "within_reference": ("gray", "Within 1 SD"),
    "below_reference":  ("red", "Below"),
    "no_sd_published":  ("gray", "No SD published"),
    "no_data":          ("gray", "No data"),
}
# Chart marks still need hex, and status hues are reserved for status only.
STATUS_HEX = {"flag": "#d03b3b", "watch": "#fab219", "normal": "#0ca30c",
              "no_baseline": "#52514e", "no_data": "#52514e"}

st.set_page_config(
    page_title="Athlete Monitoring",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit's defaults are built for notebooks: a very large h1, loose vertical
# rhythm, and headings that all carry similar weight. This tightens the type
# scale so the page reads as a document with a hierarchy.
st.html("""
<style>
  .block-container { padding-top: 3.0rem; max-width: 1520px; }

  /* Streamlit's defaults are sized for notebooks: a very large h1 and headings
     that all carry similar weight. */
  h1 { font-size: 1.7rem !important; font-weight: 650 !important;
       letter-spacing: -0.015em; margin-bottom: 0 !important; }
  h2 { font-size: 1.12rem !important; font-weight: 620 !important;
       margin-top: 1.7rem !important; margin-bottom: 0.55rem !important;
       letter-spacing: -0.005em; }
  /* st.subheader renders an h3, so the h2 rule never reached the section
     headings on this page. */
  h3 { font-size: 1.12rem !important; font-weight: 620 !important;
       margin-top: 1.7rem !important; margin-bottom: 0.55rem !important;
       letter-spacing: -0.005em; }
  h6 { font-weight: 620 !important; }
  h6 { font-size: 0.76rem !important; text-transform: uppercase;
       letter-spacing: 0.075em; color: #8b8a83 !important;
       margin-bottom: 0.35rem !important; }

  /* Cards lift to white against the page. */
  [data-testid="stVerticalBlockBorderWrapper"] {
      background: #ffffff; border-radius: 10px; border-color: #e5e3dd;
  }

  [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 620; }
  [data-testid="stMetricLabel"] p { font-size: 0.76rem; color: #6b6a65; }
  [data-testid="stCaptionContainer"] p { font-size: 0.79rem; line-height: 1.5; }

  [data-testid="stDataFrame"] { border-radius: 8px; }
  .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
  hr { margin: 1.5rem 0 !important; border-color: #e5e3dd !important; }
  section[data-testid="stSidebar"] h1 { font-size: 1.0rem !important; }

  /* The status stripe at the top of an attention card. */
  .status-rule { height: 3px; border-radius: 3px; margin: -2px 0 10px 0; }
</style>
""")


def status_rule(colour_key: str) -> None:
    """A coloured stripe across the top of a card, so a flagged athlete is
    identifiable before any text is read."""
    st.html(f'<div class="status-rule" style="background:{STATUS_HEX.get(colour_key, "#dcdad4")}"></div>')


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
def load_profile(code) -> pd.DataFrame:
    return q.quality_profile(code)


@st.cache_data(ttl=300, show_spinner=False)
def load_normative(code) -> pd.DataFrame:
    return q.normative_comparison(code)


@st.cache_data(ttl=3600, show_spinner=False)
def load_studies() -> pd.DataFrame:
    return q.reference_studies()


@st.cache_data(ttl=300, show_spinner=False)
def load_missing_reference(code) -> pd.DataFrame:
    return q.metrics_without_reference(code)


@st.cache_data(ttl=300, show_spinner=False)
def load_headline_history(code) -> pd.DataFrame:
    return q.headline_history(code)


@st.cache_data(ttl=300, show_spinner=False)
def load_test_days(code) -> pd.DataFrame:
    return q.test_days(code)


@st.cache_data(ttl=300, show_spinner=False)
def load_day_detail(code, day) -> pd.DataFrame:
    return q.test_day_detail(code, day)


@st.cache_data(ttl=900, show_spinner=False)
def load_narrative(code):
    return athlete_narrative(code)


@st.cache_data(ttl=900, show_spinner="Assembling the report…")
def load_report(code) -> tuple[str, dict[str, tuple[str, object, str | None]]]:
    """Returns the rendered markdown plus per-slot provenance.

    The AthleteReport object holds DataFrames, which cache awkwardly and are not
    needed once rendered; the markdown and the provenance are.
    """
    rep = build_report(code)
    prov = {k: (v.source, v.guard_passed, v.fallback_reason) for k, v in rep.slots.items()}
    return render_markdown(rep), prov


@st.cache_data(ttl=300, show_spinner=False)
def load_ops():
    return q.recent_runs(8), q.recent_rejections(30)


@st.cache_data(ttl=3600, show_spinner=False)
def analyse_trace(path_str: str):
    df = pd.read_csv(path_str)
    return analyse_cmj(df, time_col="time_s",
                       force_cols=[c for c in df.columns if c.startswith("Fz")])


def chip(mapping: dict, key: str) -> str:
    """A badge, rendered by a MarkdownColumn or st.markdown."""
    colour, label = mapping.get(key, mapping["no_data"])
    return f":{colour}-badge[{label}]"


# Streamlit's `:color-badge[...]` directive is its own markdown flavour and the
# dataframe renderer does not read it, so status inside a table is coloured text
# through a Styler instead. Badges are used where markdown is rendered directly.
STATUS_COLOURS = {
    "red": "#c0392b", "orange": "#b8651b", "yellow": "#8a6d0b",
    "green": "#0a7d0a", "gray": "#6b6a65",
}


def label(mapping: dict, key: str) -> str:
    return mapping.get(key, mapping["no_data"])[1]


def num(v) -> str:
    """Readable across a battery that spans 0.213 m and 1398 m.

    A single %g switches to scientific notation above four digits, which turned
    a Yo-Yo distance into 1.4e+03 in the middle of a coach-facing table.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "\u2014"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def fmt(series, pattern: str = "{:.2f}", dash: str = "\u2014"):
    """Pre-format a column to strings.

    Streamlit's table frontend renders a missing cell as the literal "None",
    ignoring the Styler's display value for it. Formatting in pandas and handing
    over strings keeps empty cells looking empty. The cost is that the column
    sorts as text, which is acceptable on tables this size.
    """
    out = pd.to_numeric(series, errors="coerce")
    return out.map(lambda v: dash if pd.isna(v) else pattern.format(v))


def colour_status(frame, columns: list[str], mapping: dict):
    """Colour the status words in a table. The label always carries the meaning;
    the colour is a second channel on top of it."""
    lookup = {v[1]: STATUS_COLOURS[v[0]] for v in mapping.values()}

    def paint(v):
        c = lookup.get(v)
        return f"color: {c}; font-weight: 600" if c else ""

    return frame.style.map(paint, subset=columns)


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
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
with head_l:
    st.title("Athlete monitoring")
    st.caption("Neuromuscular readiness and training load")
with head_r:
    st.markdown(
        f"<div style='text-align:right;color:#6b6a65;font-size:0.82rem;line-height:1.7'>"
        f"<span style='color:#0b0b0b;font-weight:600'>{len(status)} athletes</span> · "
        f"{status['squad'].nunique()} squads · {status['sport'].nunique()} sports<br>"
        f"Latest testing day <b>{hi_all:%-d %b %Y}</b></div>",
        unsafe_allow_html=True,
    )
st.html('<hr style="margin:0.9rem 0 1.3rem 0;border:0;border-top:1px solid #e5e3dd">')

snapshot, briefing = load_briefing()

with st.container(border=True):
    st.markdown("###### Today")
    st.markdown(briefing.text)
    badge = (
        f"Written by {briefing.source}, numeric guard passed"
        if briefing.guard_passed
        else f"Deterministic template ({briefing.fallback_reason or 'no model configured'})"
    )
    st.caption(f"{badge}. Every number is computed in SQL; the model only phrases them.")

    st.markdown("")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Athletes monitored", snapshot.n_athletes)
    c2.metric("Flagged", snapshot.n_flag,
              help="CMJ at or below 1.5 SD under the athlete's own 28-day baseline")
    c3.metric("On watch", snapshot.n_watch, help="Between 1.0 and 1.5 SD below baseline")
    c4.metric("Load concerns", snapshot.n_load_concern, help="ACWR above 1.30")
    c5.metric("Rejected, latest ingest", snapshot.rejected_today,
              help="Rows the pipeline validation excluded on the most recent run of each source. "
                   "Not a running total: re-running over the same files is idempotent, so summing "
                   "across runs would count the same bad row repeatedly.")

# ---------------------------------------------------------------------------
# attention queue — the five-second answer
# ---------------------------------------------------------------------------
st.subheader("Who needs attention")
attention = view[view["attention_rank"] <= 4].copy()

if attention.empty:
    st.success("No athlete in the selected squads is outside their normal range today.", icon=":material/check_circle:")
else:
    for r in attention.itertuples(index=False):
        with st.container(border=True):
            status_rule(r.baseline_status if r.baseline_status in STATUS_HEX else "no_data")
            left, right = st.columns([3, 5], vertical_alignment="center")
            with left:
                st.markdown(
                    f"**{r.athlete_code}** &nbsp; {chip(STATUS, r.baseline_status)} "
                    f"{chip(ZONE, r.acwr_zone) if r.acwr_zone in ('caution', 'high_risk') else ''}"
                )
                st.caption(f"{r.squad} · last tested {r.last_cmj_date:%-d %b}")
            with right:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Jump", f"{float(r.jump_height_m) * 100:.1f} cm",
                          delta=f"{(float(r.jump_height_m) - float(r.baseline_mean_m)) * 100:+.1f} cm",
                          delta_color="normal", border=False)
                m2.metric("28-day baseline", f"{float(r.baseline_mean_m) * 100:.1f} cm")
                m3.metric("z-score", f"{float(r.z_score):+.2f}")
                m4.metric("ACWR", f"{float(r.acwr):.2f}" if r.acwr is not None else "—")

    st.caption(
        "Ordered by urgency: a neuromuscular flag outranks a workload warning, because the jump "
        "is a measurement of the athlete whereas ACWR is an inference about accumulated exposure."
    )

with st.expander("Full squad"):
    full = pd.DataFrame({
        "Athlete": view["athlete_code"],
        "Squad": view["squad"],
        "Sport": view["sport"],
        "Neuromuscular": [label(STATUS, x) for x in view["baseline_status"]],
        "Jump (cm)": fmt(view["jump_height_m"].astype(float) * 100, "{:.1f}"),
        "z": fmt(view["z_score"], "{:+.2f}"),
        "Workload": [label(ZONE, x) for x in view["acwr_zone"]],
        "ACWR": fmt(view["acwr"], "{:.2f}"),
    })
    st.dataframe(
        colour_status(full, ["Neuromuscular"], STATUS),
        hide_index=True, width="stretch",
    )

# ---------------------------------------------------------------------------
# athlete detail
# ---------------------------------------------------------------------------
st.divider()
row = status[status["athlete_code"] == chosen_athlete].iloc[0]
# Squad names in this programme already carry the sport ("Football - Women"),
# so repeating it reads as a mistake rather than as extra information.
_sport_suffix = "" if str(row.sport).lower() in str(row.squad).lower() else f" · {row.sport}"
st.subheader(f"{chosen_athlete} · {row.squad}{_sport_suffix}")

narrative = load_narrative(chosen_athlete)
with st.container(border=True):
    st.markdown("###### Development summary")
    st.markdown(narrative.text)
    if narrative.guard_passed:
        st.caption(
            f"Written by {narrative.source}, numeric guard passed. Trends are fitted in SQL "
            "across every test; the model selects and phrases them and is blocked from "
            "prescribing training."
        )
    else:
        st.caption(
            f"Deterministic summary ({narrative.fallback_reason or 'no model configured'}). "
            "The system reports the same trends with or without a model."
        )

tab_profile, tab_day, tab_nm, tab_report = st.tabs(
    ["Physical qualities", "Test day", "Neuromuscular monitoring", "Development report"]
)

# --- physical qualities ----------------------------------------------------
with tab_profile:
    prof = load_profile(chosen_athlete)
    if prof.empty:
        st.info("No physical testing on record for this athlete.")
    else:
        table = pd.DataFrame({
            "Quality": prof["quality_name"],
            "Measure": prof["display_name"],
            "Tests": prof["n_tests"].astype(int),
            "First": prof["first_value"].astype(float),
            "Latest": prof["latest_value"].astype(float),
            "Unit": prof["unit"],
            "Trend": fmt(prof["pct_improvement_fitted"], "{:+.1f}%"),
            "Direction": [label(DIRECTION, d) for d in prof["direction"]],
            "Fit r\u00b2": fmt(prof["trend_r2"], "{:.2f}"),
        })
        st.dataframe(
            colour_status(table, ["Direction"], DIRECTION).format({"First": num, "Latest": num}),
            hide_index=True, width="stretch",
        )
        st.caption(
            "**Trend is positive when the athlete got better, whichever way the metric runs** — "
            "a 505 time falling and a Yo-Yo distance rising are both progress, and the sign is "
            "taken from the metric catalogue rather than from the arithmetic. The percentage is "
            "fitted across every test by least squares, not first-versus-latest: two endpoints "
            "carry the full test-retest error of both days and can invert the direction of a "
            "real trend. r² says how much of the scatter the line actually explains — a low r² "
            "with a large percentage is a trend you should not lean on."
        )

        hist = load_headline_history(chosen_athlete)
        if not hist.empty:
            h = hist.copy()
            h["metric_value"] = pd.to_numeric(h["metric_value"], errors="coerce")
            # A panel whose metric runs downwards must say so. The 10 m sprint
            # line rising looks like progress and is the opposite; silently
            # inverting the axis instead would be worse, because the reader
            # would have no way to know it had happened.
            h["panel"] = (
                h["quality_name"] + " · " + h["display_name"] + " (" + h["unit"] + ")"
                + h["higher_is_better"].map({True: "", False: "  ↓ lower is better"})
            )
            order = (
                h.drop_duplicates("panel").sort_values(["quality_order", "display_name"])["panel"].tolist()
            )
            xq = alt.X("session_date:T", title=None,
                       axis=alt.Axis(grid=False, format="%b", domainColor=GRID, tickCount=4))
            yq = alt.Y("metric_value:Q", title=None, scale=alt.Scale(zero=False),
                       axis=alt.Axis(gridColor=GRID, tickCount=4))
            # The table quotes a fitted percentage; without the line it is
            # asserted rather than shown.
            # The fitted line carries the direction the table reports. Eight
            # panels in one colour read as wallpaper; colouring the slope makes
            # a declining quality findable without reading a single label.
            direction_by_metric = dict(zip(prof["display_name"], prof["direction"]))
            h["direction"] = h["display_name"].map(direction_by_metric).fillna("no_data")
            base_q = alt.Chart(h)
            series = base_q.mark_line(
                color=BLUE, size=1.6,
                point=alt.OverlayMarkDef(size=26, filled=True, color=BLUE),
            ).encode(
                x=xq, y=yq,
                tooltip=[
                    alt.Tooltip("session_date:T", title="Date"),
                    alt.Tooltip("display_name:N", title="Metric"),
                    alt.Tooltip("metric_value:Q", title="Value", format=".3f"),
                    alt.Tooltip("unit:N", title="Unit"),
                ],
            )
            # `direction` has to be in the groupby: transform_regression drops
            # every column that is not, so the colour encoding had nothing to
            # read and the fitted lines silently vanished. It is constant within
            # a panel, so grouping on it changes no fit.
            fitted = base_q.transform_regression(
                "session_date", "metric_value", groupby=["panel", "direction"]
            ).mark_line(size=2.0, strokeDash=[6, 4], opacity=0.95).encode(
                x=xq, y=yq,
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["improving", "stable", "declining", "insufficient_data"],
                        range=["#0ca30c", "#8b8a83", "#d03b3b", "#c9c7c0"],
                    ),
                    legend=alt.Legend(orient="top", title="Fitted trend", direction="horizontal",
                                      labelExpr="datum.label == 'insufficient_data' "
                                                "? 'Too few tests' : datum.label"),
                ),
            )

            small = (
                alt.layer(series, fitted)
                .properties(width=250, height=130)
                .facet(facet=alt.Facet("panel:N", title=None, sort=order,
                                       header=alt.Header(labelFontSize=11, labelAnchor="start")),
                       columns=3)
                # Each quality is on its own scale: metres, seconds and metres of
                # shuttle running share no axis, and forcing them onto one would
                # flatten every panel but the largest.
                .resolve_scale(y="independent", x="shared")
            )
            st.altair_chart(small.configure_view(strokeWidth=0), width='stretch')
            st.caption(
                "Blue is the measured series. The dashed line is the least-squares fit whose "
                "slope produces the Trend column above, coloured by the direction it reports. "
                "Panels marked ↓ are metrics where a falling line is an improvement."
            )

        # ---- against published norms -------------------------------------
        st.markdown("---")
        st.markdown("**Against published normative data**")
        norm = load_normative(chosen_athlete)
        missing = load_missing_reference(chosen_athlete)

        if norm.empty:
            st.info(
                f"No published reference in the library matches {row.sport} athletes of this sex "
                "for the tests this athlete performs. The comparison is left empty rather than "
                "substituted with a population that does not apply.",
                icon=":material/menu_book:",
            )
        else:
            n = norm.copy()
            for c in ("athlete_value", "reference_mean", "reference_sd",
                      "reference_low", "reference_high", "pct_vs_reference", "z_vs_reference"):
                n[c] = pd.to_numeric(n[c], errors="coerce")

            def spread(r):
                if r.spread_type == "sd":
                    return f"± {r.reference_sd:g} SD"
                if r.spread_type == "ci95":
                    return f"95% CI {r.reference_low:g}–{r.reference_high:g}"
                if r.spread_type == "range":
                    return f"range {r.reference_low:g}–{r.reference_high:g}"
                return "—"

            comparison = pd.DataFrame({
                "Measure": n["display_name"],
                "This athlete": n["athlete_value"].astype(float),
                "Unit": n["unit"],
                "Reference": n["reference_mean"].astype(float),
                "Spread": [spread(r) for r in n.itertuples(index=False)],
                "n": n["reference_n"].astype(int),
                "vs ref": fmt(n["pct_vs_reference"], "{:+.1f}%"),
                "z": fmt(n["z_vs_reference"], "{:+.2f}"),
                "Standing": [label(STANDING, x) for x in n["standing"]],
                "Reference population": n["population"],
                "Study": n["study_key"] + " " + n["reference_year"].astype(str),
            })
            st.dataframe(
                colour_status(comparison, ["Standing"], STANDING).format(
                    {"This athlete": num, "Reference": num},
                ),
                hide_index=True, width="stretch",
            )

            plot = n.dropna(subset=["z_vs_reference"]).copy()
            if not plot.empty:
                plot["label"] = plot["display_name"] + " — " + plot["population"].str.slice(0, 34)
                band = alt.Chart(pd.DataFrame({"lo": [-1.0], "hi": [1.0]})).mark_rect(
                    opacity=0.13, color=STATUS_HEX["normal"]
                ).encode(x="lo:Q", x2="hi:Q")
                zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(
                    color=MUTED, strokeDash=[4, 4], size=1
                ).encode(x="z:Q")
                pts = alt.Chart(plot).mark_point(
                    filled=True, size=150, color=BLUE, stroke="white", strokeWidth=1.2
                ).encode(
                    x=alt.X("z_vs_reference:Q",
                            title="Standard deviations from the published mean  (right = better)",
                            scale=alt.Scale(domain=[-4, 4], clamp=True),
                            axis=alt.Axis(gridColor=GRID)),
                    y=alt.Y("label:N", title=None, sort=None,
                            axis=alt.Axis(labelLimit=340, domainColor=GRID)),
                    tooltip=[
                        alt.Tooltip("display_name:N", title="Measure"),
                        alt.Tooltip("athlete_value:Q", title="This athlete"),
                        alt.Tooltip("reference_mean:Q", title="Reference mean"),
                        alt.Tooltip("reference_sd:Q", title="Reference SD"),
                        alt.Tooltip("z_vs_reference:Q", title="z", format="+.2f"),
                        alt.Tooltip("population:N", title="Reference population"),
                        alt.Tooltip("citation:N", title="Source"),
                    ],
                )
                st.altair_chart(
                    (band + zero + pts).properties(height=28 * len(plot) + 90)
                    .configure_view(strokeWidth=0),
                    width="stretch",
                )
                st.caption(
                    "Plotted in standard deviations because the measures are in different units; "
                    "the sign is corrected so right is always better. The green band is one SD "
                    "either side of the published mean. **Only studies that published an SD "
                    "appear here** — a 95% confidence interval describes uncertainty about the "
                    "mean, not the spread of athletes, and dividing by it would make an athlete "
                    "look several standard deviations from normal when they are a fraction of one."
                )

            notes = [x for x in n["protocol_note"].dropna().unique() if x]
            if notes:
                with st.expander("Protocol differences that affect these comparisons"):
                    for x in notes:
                        st.markdown(f"- {x}")

        if not missing.empty:
            st.caption(
                "**No published reference for:** "
                + ", ".join(missing["display_name"])
                + ". The library covers football and basketball; swimming and sprint athletics "
                "are not yet represented."
            )

        with st.expander("Reference library — every source, with its DOI"):
            studies = load_studies()
            for r in studies.itertuples(index=False):
                ident = f"DOI [{r.doi}](https://doi.org/{r.doi})" if r.doi else (
                    f"PMID [{r.pmid}](https://pubmed.ncbi.nlm.nih.gov/{r.pmid}/)" if r.pmid else "—")
                st.markdown(f"**{r.citation}**")
                st.markdown(f"{ident} · {r.n_values} value(s) in use · verified {r.verified_on}")
                if r.note:
                    st.caption(f"Caveat: {r.note}")
                st.markdown("")
            st.caption(
                "Each entry was retrieved and read before being loaded. A value that could not "
                "be verified against the source was left out rather than approximated."
            )

# --- one test day ----------------------------------------------------------
with tab_day:
    days_df = load_test_days(chosen_athlete)
    if days_df.empty:
        st.info("No test days on record.")
    else:
        opts = days_df["session_date"].tolist()

        def day_label(d):
            r = days_df[days_df["session_date"] == d].iloc[0]
            kinds = str(r["tests"]).replace("_test", "").replace("_", " ")
            return f"{d}  ·  {r['n_metrics']} measurements  ·  {kinds}"

        picked_day = st.selectbox("Test day", opts, format_func=day_label, key="test_day")
        detail = load_day_detail(chosen_athlete, picked_day)
        if detail.empty:
            st.info("Nothing recorded on this date.")
        else:
            st.caption(
                f"{len(detail)} measurements across "
                f"{detail['quality_name'].nunique()} physical qualities. "
                "**z is against this athlete's own history for that metric**, sign-corrected so "
                "positive is always a better-than-usual result — not a comparison with the squad, "
                "who may be a different sport and sex entirely."
            )
            for quality in detail["quality_name"].unique():
                block = detail[detail["quality_name"] == quality]
                st.markdown(f"**{quality}**")
                st.dataframe(
                    pd.DataFrame({
                        "Measurement": block["display_name"],
                        "Value": block["value"].astype(float).map("{:g}".format),
                        "Unit": block["unit"],
                        "Athlete's mean": block["athlete_mean"].astype(float).map("{:g}".format),
                        "z vs own mean": [
                            "—" if pd.isna(v) else f"{float(v):+.2f}" for v in block["z_vs_own_mean"]
                        ],
                        "Test": block["session_type"].str.replace("_test", "", regex=False),
                        "Source": block["source"],
                    }),
                    hide_index=True, width='stretch',
                )

# --- neuromuscular monitoring ---------------------------------------------
with tab_nm:
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
            d["Status"] = d["baseline_status"].map(lambda s: STATUS.get(s, STATUS["no_data"])[1])

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
            flag_line = base_chart.mark_line(color=STATUS_HEX["flag"], strokeDash=[2, 3], size=1,
                                             opacity=0.7).encode(x=x, y="flag_line:Q")
            trend = base_chart.mark_line(color=BLUE, size=2).encode(x=x, y="jump_cm:Q")

            # Size alone could not make the flagged trial read as the most important
            # mark on the chart -- in Vega-Lite `size` is area, and a triangle spends
            # far less ink than a circle of equal area. A ring behind the marker
            # settles it, and is redundant with colour, shape and the legend.
            halo = alt.Chart(d[d["baseline_status"] == "flag"]).mark_point(
                shape="circle", filled=False, size=430,
                stroke=STATUS_HEX["flag"], strokeWidth=2.2, opacity=0.85,
            ).encode(x=x, y="jump_cm:Q")
            pts = base_chart.mark_point(filled=True, stroke="white", strokeWidth=1.2).encode(
                x=x, y="jump_cm:Q",
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Normal", "Watch", "Flag", "No baseline"],
                        range=[STATUS_HEX["normal"], STATUS_HEX["watch"],
                               STATUS_HEX["flag"], STATUS_HEX["no_baseline"]],
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
            a["Zone"] = a["acwr_zone"].map(lambda z: ZONE.get(z, ZONE["no_data"])[1])

            # A scale fitted to the data alone can crop the sweet-spot band, which
            # is the very reference the line is being judged against.
            y_lo = min(0.75, float(a["acwr"].min()) - 0.05)
            y_hi = max(1.40, float(a["acwr"].max()) + 0.05)
            sweet = alt.Chart(pd.DataFrame({"lo": [0.8], "hi": [1.3]})).mark_rect(
                opacity=0.13, color=STATUS_HEX["normal"]
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
            acwr_txt = f"{float(row.acwr):.2f}" if row.acwr is not None else "—"
            st.markdown(
                f"Current ratio **{acwr_txt}** &nbsp; {chip(ZONE, row.acwr_zone)}"
            )
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
                        icon=":material/block:",
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
                            icon=":material/warning:",
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
                    st.warning(" · ".join(warns), icon=":material/warning:")
                st.caption(
                    "Computed live from the raw dual-plate waveform, not read back from the database — "
                    "this is the same function the ingest pipeline runs, so the dashboard and the "
                    "pipeline cannot disagree about whether a trial is valid."
                )
            except Exception as exc:
                st.error(f"Could not analyse this trace: {type(exc).__name__}: {exc}")


# --- full development report ----------------------------------------------
with tab_report:
    st.caption(
        "A review document rather than a dashboard panel. Every table and figure below is "
        "assembled from SQL; three sections are written by the model, each against its own "
        "facts and each checked separately — so one rejected paragraph costs that paragraph "
        "and not the report."
    )
    if st.button("Generate report", key="gen_report", type="primary"):
        st.session_state["report_for"] = chosen_athlete

    if st.session_state.get("report_for") == chosen_athlete:
        markdown, provenance = load_report(chosen_athlete)

        cols = st.columns(len(provenance))
        for col, (slot, (source, guard, reason)) in zip(cols, provenance.items()):
            if guard:
                col.success(f"**{slot}** — {source}, guard passed", icon=":material/check_circle:")
            elif guard is False:
                col.warning(f"**{slot}** — fell back to template. {reason}", icon=":material/warning:")
            else:
                col.info(f"**{slot}** — {source}. {reason or ''}", icon=":material/info:")

        st.download_button(
            "Download as Markdown",
            data=markdown,
            file_name=f"{chosen_athlete}_development_report.md",
            mime="text/markdown",
        )
        st.divider()
        st.markdown(markdown)
    else:
        st.info(
            "Press **Generate report** to build the full review for this athlete. "
            "It makes three model calls and takes a couple of seconds.",
            icon=":material/article:",
        )

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
