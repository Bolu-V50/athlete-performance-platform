"""Synthetic laboratory and field test batteries.

Force-plate jump testing is only one of the physical qualities a high-performance
service monitors. This module generates the rest of a realistic battery --
maximal strength, speed, change of direction, anaerobic capacity, aerobic
endurance and body composition -- at the frequencies each is actually tested at.
A Wingate is not run twice a week; a CMJ is.

Values are anchored on published ranges for the relevant population and are
differentiated by sport and sex, because a netball athlete and a male sprinter
do not share a normative band for anything.

Two export shapes are produced on purpose: laboratory tests and field tests
arrive as separate wide files with sparse columns, which is what an athlete
management system actually exports. Reshaping those into the long metrics table
is the pipeline's job.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

# quality -> per-athlete seasonal drift, as a fraction of the starting value.
# Positive always means "better", regardless of whether the underlying metric
# is higher- or lower-is-better; the sign is applied per metric.
TRENDS: dict[str, dict[str, float]] = {
    "ATH-001": {"max_strength": 0.06, "speed": 0.02, "aerobic": -0.01},
    "ATH-002": {"max_strength": 0.09, "speed": 0.03, "anaerobic": 0.05},
    "ATH-003": {"aerobic": 0.11, "cod": 0.04, "max_strength": 0.01},
    "ATH-004": {"power": 0.05, "cod": 0.06},
    "ATH-005": {"max_strength": 0.12, "power": 0.06, "aerobic": -0.03},
    "ATH-006": {"power": 0.03, "speed": 0.01},
    "ATH-007": {"aerobic": 0.08, "anaerobic": 0.04},
    "ATH-008": {"cod": 0.07, "aerobic": 0.03},
    # A genuine decline to find: strength and power falling away while aerobic
    # work continues. This is the pattern a coach should be told about.
    "ATH-009": {"max_strength": -0.09, "power": -0.06, "aerobic": 0.05},
    "ATH-010": {"speed": 0.04, "anaerobic": 0.06},
    "ATH-011": {"max_strength": 0.05, "aerobic": 0.02},
    "ATH-012": {"power": 0.04, "cod": 0.03, "aerobic": 0.06},
}

# test type -> interval in days
SCHEDULE = {
    "IMTP_test": 28,
    "wingate_test": 42,
    "sprint_test": 21,
    "agility_test": 28,
    "aerobic_test": 42,
    "anthropometry": 28,
}


def _profile(sport: str, sex: str, mass: float, rng) -> dict[str, float]:
    """Starting values for one athlete, drawn from population-appropriate bands."""
    male = sex == "M"
    sprint_pop = sport == "Athletics"

    if sprint_pop and male:
        s10 = rng.uniform(1.63, 1.76); s30 = s10 * 2.38 + rng.uniform(-0.05, 0.05)
        vmax = rng.uniform(9.4, 10.6); a505 = rng.uniform(2.32, 2.52)
        imtp_rel = rng.uniform(34, 46); wg_pk = rng.uniform(13.0, 16.5)
        yoyo = rng.uniform(720, 1440); skin = rng.uniform(38, 62)
    elif sprint_pop:
        s10 = rng.uniform(1.79, 1.93); s30 = s10 * 2.42 + rng.uniform(-0.05, 0.05)
        vmax = rng.uniform(8.4, 9.3); a505 = rng.uniform(2.44, 2.66)
        imtp_rel = rng.uniform(28, 38); wg_pk = rng.uniform(10.0, 13.0)
        yoyo = rng.uniform(680, 1320); skin = rng.uniform(52, 88)
    else:  # netball
        s10 = rng.uniform(1.84, 1.99); s30 = s10 * 2.46 + rng.uniform(-0.06, 0.06)
        vmax = rng.uniform(7.8, 8.8); a505 = rng.uniform(2.46, 2.70)
        imtp_rel = rng.uniform(26, 36); wg_pk = rng.uniform(9.0, 12.2)
        yoyo = rng.uniform(880, 1720); skin = rng.uniform(58, 96)

    return {
        "sprint_10m_s": s10,
        "sprint_30m_s": s30,
        "max_velocity_ms": vmax,
        "agility_505_s": a505,
        "imtp_relative_force_nkg": imtp_rel,
        "imtp_peak_force_n": imtp_rel * mass,
        "imtp_rfd_0_250ms_ns": imtp_rel * mass * rng.uniform(2.4, 3.6),
        "wingate_peak_power_w_kg": wg_pk,
        "wingate_mean_power_w_kg": wg_pk * rng.uniform(0.66, 0.75),
        "wingate_fatigue_index_pct": rng.uniform(34, 58),
        "yoyo_ir1_distance_m": yoyo,
        "body_mass_kg": mass,
        "sum7_skinfolds_mm": skin,
    }


# metric -> (quality, higher_is_better). Kept in step with metric_catalog.
POLARITY = {
    "sprint_10m_s": ("speed", False),
    "sprint_30m_s": ("speed", False),
    "max_velocity_ms": ("speed", True),
    "agility_505_s": ("cod", False),
    "imtp_relative_force_nkg": ("max_strength", True),
    "imtp_peak_force_n": ("max_strength", True),
    "imtp_rfd_0_250ms_ns": ("max_strength", True),
    "wingate_peak_power_w_kg": ("anaerobic", True),
    "wingate_mean_power_w_kg": ("anaerobic", True),
    "wingate_fatigue_index_pct": ("anaerobic", False),
    "yoyo_ir1_distance_m": ("aerobic", True),
    "body_mass_kg": ("body_comp", True),
    "sum7_skinfolds_mm": ("body_comp", False),
}

NOISE = {  # within-athlete typical error, as a fraction
    "sprint_10m_s": 0.012, "sprint_30m_s": 0.010, "max_velocity_ms": 0.015,
    "agility_505_s": 0.016, "imtp_relative_force_nkg": 0.045,
    "imtp_peak_force_n": 0.045, "imtp_rfd_0_250ms_ns": 0.10,
    "wingate_peak_power_w_kg": 0.035, "wingate_mean_power_w_kg": 0.030,
    "wingate_fatigue_index_pct": 0.08, "yoyo_ir1_distance_m": 0.06,
    "body_mass_kg": 0.010, "sum7_skinfolds_mm": 0.045,
}


def _value(metric: str, start: float, frac_through: float, code: str, rng) -> float:
    quality, higher_better = POLARITY[metric]
    drift = TRENDS.get(code, {}).get(quality, 0.0) * frac_through
    # A trend expressed as "better" must move the metric in its own direction.
    signed = drift if higher_better else -drift
    v = start * (1 + signed) * (1 + rng.normal(0, NOISE[metric]))
    return float(v)


def _due(day_index: int, interval: int, offset: int) -> bool:
    return (day_index - offset) % interval == 0 and day_index >= offset


def _weekday_safe_offsets(days: list[date], rng) -> dict[str, int]:
    """Pick per-test offsets that land on a weekday.

    Every testing interval here is a multiple of seven, so an athlete's test day
    falls on the same weekday all season. A naive random offset meant whoever
    drew a Saturday was never tested at all -- entire batteries missing for
    individual athletes.

    Choosing an offset in 0..4 does not fix it either: which weekday index 0
    lands on depends on the season start. So pick the target weekday first
    (Monday to Friday) and derive the offset from the calendar.
    """
    out: dict[str, int] = {}
    for t in SCHEDULE:
        target = int(rng.integers(0, 5))                     # Monday..Friday
        out[t] = next(i for i, d in enumerate(days) if d.weekday() == target)
    return out


def write_batteries(out_dir: Path, athletes: list[tuple], days: list[date], rng) -> tuple[int, int]:
    """Write lab_tests.csv and field_tests.csv. Returns (lab_rows, field_rows)."""
    profiles = {a[0]: _profile(a[1], a[2], a[4], rng) for a in athletes}

    lab_cols = ["imtp_peak_force_n", "imtp_relative_force_nkg", "imtp_rfd_0_250ms_ns",
                "wingate_peak_power_w_kg", "wingate_mean_power_w_kg", "wingate_fatigue_index_pct"]
    field_cols = ["sprint_10m_s", "sprint_30m_s", "max_velocity_ms", "agility_505_s",
                  "yoyo_ir1_distance_m", "body_mass_kg", "sum7_skinfolds_mm"]

    lab_rows: list[dict] = []
    field_rows: list[dict] = []
    n = len(days)

    for code, _sport, _sex, _squad, _mass, *_ in athletes:
        prof = profiles[code]
        offs = _weekday_safe_offsets(days, rng)
        for i, d in enumerate(days):
            frac = i / max(n - 1, 1)

            if d.weekday() > 4:
                continue  # tests are shifted onto weekdays by the offset above
            imtp = _due(i, SCHEDULE["IMTP_test"], offs["IMTP_test"])
            wing = _due(i, SCHEDULE["wingate_test"], offs["wingate_test"])
            if imtp or wing:
                row = {"athlete_code": code, "date": d.isoformat()}
                for c in lab_cols:
                    is_imtp = c.startswith("imtp")
                    row[c] = (round(_value(c, prof[c], frac, code, rng), 1)
                              if (imtp if is_imtp else wing) else "")
                lab_rows.append(row)

            spr = _due(i, SCHEDULE["sprint_test"], offs["sprint_test"])
            agi = _due(i, SCHEDULE["agility_test"], offs["agility_test"])
            aer = _due(i, SCHEDULE["aerobic_test"], offs["aerobic_test"])
            ant = _due(i, SCHEDULE["anthropometry"], offs["anthropometry"])
            if spr or agi or aer or ant:
                row = {"athlete_code": code, "date": d.isoformat()}
                group = {"sprint_10m_s": spr, "sprint_30m_s": spr, "max_velocity_ms": spr,
                         "agility_505_s": agi, "yoyo_ir1_distance_m": aer,
                         "body_mass_kg": ant, "sum7_skinfolds_mm": ant}
                for c in field_cols:
                    dp = 3 if c.endswith("_s") else (1 if c != "yoyo_ir1_distance_m" else 0)
                    row[c] = round(_value(c, prof[c], frac, code, rng), dp) if group[c] else ""
                field_rows.append(row)

    # ---- injected faults, mirroring the force-plate ones -------------------
    # A 10 m sprint of 0.94 s would be a world record by a distance: a gate was
    # triggered early. A Yo-Yo of 9999 m is a data-entry sentinel value.
    field_rows.append({"athlete_code": "ATH-004", "date": days[150].isoformat(),
                       **{c: "" for c in field_cols}, "sprint_10m_s": 0.94, "sprint_30m_s": 4.31,
                       "max_velocity_ms": 8.9})
    field_rows.append({"athlete_code": "ATH-008", "date": days[152].isoformat(),
                       **{c: "" for c in field_cols}, "yoyo_ir1_distance_m": 9999})
    lab_rows.append({"athlete_code": "ATH-777", "date": days[154].isoformat(),
                     **{c: "" for c in lab_cols}, "imtp_relative_force_nkg": 38.0,
                     "imtp_peak_force_n": 3000.0})

    with (out_dir / "lab_tests.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["athlete_code", "date"] + lab_cols)
        w.writeheader(); w.writerows(lab_rows)
    with (out_dir / "field_tests.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["athlete_code", "date"] + field_cols)
        w.writeheader(); w.writerows(field_rows)

    return len(lab_rows), len(field_rows)
