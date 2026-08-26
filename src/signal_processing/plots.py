"""Diagnostic figure for a single CMJ trial.

The point of this figure is not decoration: a practitioner has to be able to see
*where the analyser thinks each phase begins and ends*, because almost every bad
CMJ number traces back to a mis-detected onset or take-off. Phase boundaries are
therefore drawn on the trace, not just reported as durations.

Force and velocity are plotted on two stacked panels sharing a time axis rather
than on twin y-axes -- two different units on one axis invites the reader to
compare heights that are not comparable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cmj import CMJResult

__all__ = ["plot_cmj"]

# Validated categorical palette (see dataviz reference palette).
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text="#0b0b0b",
        text_secondary="#52514e",
        grid="#dcdad4",
        force="#2a78d6",
        velocity="#eb6834",
        rule="#8a8880",
    ),
    "dark": dict(
        surface="#1a1a19",
        text="#ffffff",
        text_secondary="#c3c2b7",
        grid="#3a3a37",
        force="#3987e5",
        velocity="#d95926",
        rule="#8a8880",
    ),
}

# Phase bands: tinted background regions, each carrying a direct text label, so
# identity never rests on colour alone.
PHASE_TINTS = {
    "Weighing": "#8a8880",
    "Unweighting": "#2a78d6",
    "Braking": "#eb6834",
    "Propulsion": "#1baf7a",
    "Flight": "#eda100",
    "Landing": "#e87ba4",
}


def plot_cmj(
    result: CMJResult,
    title: str | None = None,
    theme: str = "light",
    save_to: str | Path | None = None,
    dpi: int = 150,
):
    """Render the annotated force-time diagnostic. Returns the matplotlib Figure."""
    if result._time is None or result._force is None:
        raise ValueError("result carries no trace; analyse_cmj must be given raw data")

    c = THEMES[theme]
    t, f = result._time, result._force
    idx = result.indices
    bw = result.body_weight_n

    fig, (ax_f, ax_v) = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True, height_ratios=[2.3, 1.0], dpi=dpi
    )
    fig.patch.set_facecolor(c["surface"])

    # ---- phase bands ---------------------------------------------------
    bands: list[tuple[str, int, int]] = []
    if idx:
        bands = [
            ("Weighing", idx["quiet_start"], idx["onset"]),
            ("Unweighting", idx["onset"], idx["min_velocity"]),
            ("Braking", idx["min_velocity"], idx["zero_velocity"]),
            ("Propulsion", idx["zero_velocity"], idx["takeoff"]),
            ("Flight", idx["takeoff"], idx["landing"]),
            ("Landing", idx["landing"], len(t) - 1),
        ]
    for ax in (ax_f, ax_v):
        ax.set_facecolor(c["surface"])
        for name, a, b in bands:
            if b > a:
                ax.axvspan(t[a], t[b], color=PHASE_TINTS[name], alpha=0.10, lw=0, zorder=0)

    # ---- force panel ---------------------------------------------------
    ax_f.plot(t, f, color=c["force"], lw=1.6, zorder=3, solid_joinstyle="round")
    ax_f.axhline(bw, color=c["rule"], lw=1.0, ls=(0, (5, 4)), zorder=2)
    # Parked in the quiet-standing region: the only reliably empty part of the
    # panel. The right-hand side carries the landing ring-down.
    ax_f.annotate(
        f"body weight  {bw:.0f} N  ({result.body_mass_kg:.1f} kg)",
        xy=(t[0], bw),
        xytext=(8, 7),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=c["text_secondary"],
        fontsize=8.5,
    )

    ymax = float(np.max(f)) * 1.16
    ax_f.set_ylim(0, ymax)
    # The braking and propulsion phases are inherently short. Labelling every
    # band at one height makes their text overlap, so narrow bands alternate
    # between two rows with a leader tick back to their own band.
    span = t[-1] - t[0]
    narrow_n = 0
    for name, a, b in bands:
        if b <= a:
            continue
        frac = (t[b] - t[a]) / span
        if frac < 0.015:
            continue
        if frac >= 0.09:
            y = ymax * 0.955
        else:
            y = ymax * (0.955 if narrow_n % 2 == 0 else 0.870)
            narrow_n += 1
        xm = (t[a] + t[b]) / 2
        ax_f.text(xm, y, name, ha="center", va="top", fontsize=8.5,
                  color=c["text_secondary"], zorder=4)
        if frac < 0.09:
            ax_f.plot([xm, xm], [y - ymax * 0.030, ymax * 0.955],
                      color=c["grid"], lw=0.7, zorder=3)
    ax_f.set_ylabel("Vertical GRF (N)", color=c["text"], fontsize=10)

    # ---- event markers -------------------------------------------------
    for key, label in (("onset", "onset"), ("takeoff", "take-off"), ("landing", "landing")):
        if key in idx:
            for ax in (ax_f, ax_v):
                ax.axvline(t[idx[key]], color=c["rule"], lw=0.9, ls=":", zorder=2)
            ax_f.annotate(
                label,
                xy=(t[idx[key]], 0),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=8,
                color=c["text_secondary"],
                rotation=90,
                va="bottom",
            )

    # ---- velocity panel ------------------------------------------------
    if result._velocity is not None and idx:
        seg = slice(idx["onset"], idx["takeoff"] + 1)
        ax_v.plot(t[seg], result._velocity, color=c["velocity"], lw=1.8, zorder=3)
        ax_v.axhline(0, color=c["rule"], lw=1.0, ls=(0, (5, 4)), zorder=2)
        ax_v.annotate(
            f"take-off  {result.takeoff_velocity_ms:.2f} m/s",
            xy=(t[idx["takeoff"]], result.takeoff_velocity_ms),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=c["text_secondary"],
        )
        ax_v.plot([t[idx["takeoff"]]], [result.takeoff_velocity_ms], "o",
                  ms=5, color=c["velocity"], zorder=4)
    ax_v.set_ylabel("COM velocity (m/s)", color=c["text"], fontsize=10)
    ax_v.set_xlabel("Time (s)", color=c["text"], fontsize=10)

    # ---- chrome --------------------------------------------------------
    for ax in (ax_f, ax_v):
        ax.grid(True, color=c["grid"], lw=0.6, alpha=0.7, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
        ax.tick_params(colors=c["text_secondary"], labelsize=9)
    ax_f.set_xlim(t[0], t[-1])

    # ---- headline metrics ----------------------------------------------
    head = (
        f"jump height {result.jump_height_m * 100:.1f} cm"
        f"     RSI-mod {result.rsi_mod:.2f}"
        f"     peak force {result.peak_force_bw:.2f}×BW"
        f"     peak power {result.peak_power_w_kg:.1f} W/kg"
        f"     contraction {result.contraction_time_s * 1000:.0f} ms"
    )
    fig.suptitle(title or "Countermovement jump — force-time diagnostic",
                 x=0.008, ha="left", fontsize=12.5, color=c["text"], y=0.985)
    fig.text(0.008, 0.928, head, ha="left", fontsize=9.5, color=c["text_secondary"])

    if result.quality_flags:
        fig.text(
            0.008,
            0.012,
            "  |  ".join(result.quality_flags),
            ha="left",
            fontsize=8,
            color="#e34948" if not result.is_valid else c["text_secondary"],
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.915))

    if save_to:
        p = Path(save_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor=c["surface"], dpi=dpi, bbox_inches="tight")
    return fig
