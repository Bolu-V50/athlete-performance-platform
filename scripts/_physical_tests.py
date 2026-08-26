"""Synthetic laboratory and field test batteries, differentiated by sport.

Two things this module exists to get right.

**Different sports run different tests.** A land-based 10 m sprint and a Yo-Yo
IR1 say very little about a swimmer, and a service that ran them anyway would be
filling a database with numbers no coach would act on. Swimming therefore
carries pool-based speed and endurance measures instead, which is also why the
metric catalogue attaches `quality` to the metric rather than to the test: two
sports can fill the same physical quality with entirely different measurements
and the capability profile still lines up.

**Different sports have different normative bands.** A basketball centre and a
distance swimmer do not share a range for anything, so nothing here is drawn
from one global distribution. Values are anchored on published ranges for the
relevant population and sex.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np

# quality -> per-athlete seasonal drift, as a fraction of the starting value.
# Positive always means "better"; the sign is applied per metric according to
# whether that metric improves by rising or by falling.
TRENDS: dict[str, dict[str, float]] = {
    "ATH-001": {"max_strength": 0.07, "speed": 0.03, "aerobic": 0.02},
    "ATH-002": {"max_strength": 0.09, "anaerobic": 0.05, "speed": 0.02},
    "ATH-003": {"speed": 0.05, "power": 0.04},
    "ATH-004": {"aerobic": 0.10, "max_strength": 0.02},
    "ATH-005": {"max_strength": 0.12, "power": 0.06, "aerobic": -0.03},
    "ATH-006": {"aerobic": 0.08, "speed": 0.01},
    "ATH-007": {"aerobic": 0.09, "cod": 0.05},
    "ATH-008": {"cod": 0.07, "speed": 0.03, "aerobic": 0.02},
    # The athlete a coach should be told about: strength and power falling away
    # while aerobic work carries on.
    "ATH-009": {"max_strength": -0.09, "power": -0.06, "cod": -0.05, "aerobic": 0.04},
    "ATH-010": {"speed": 0.04, "anaerobic": 0.06},
    "ATH-011": {"power": 0.05, "max_strength": 0.04},
    "ATH-012": {"max_strength": 0.08, "anaerobic": 0.03, "aerobic": -0.04},
    "ATH-013": {"power": 0.06, "cod": 0.04},
    "ATH-014": {"speed": 0.04, "max_strength": 0.05},
    "ATH-015": {"max_strength": 0.03, "power": 0.02},
    "ATH-016": {"speed": 0.06, "power": 0.05, "cod": 0.03},
}

SCHEDULE = {           # test type -> interval in days
    "IMTP_test": 28,
    "wingate_test": 42,
    "sprint_test": 21,
    "agility_test": 28,
    "aerobic_test": 42,
    "anthropometry": 28,
    "swim_test": 28,
}

LAB_COLS = ["imtp_peak_force_n", "imtp_relative_force_nkg", "imtp_rfd_0_250ms_ns",
            "wingate_peak_power_w_kg", "wingate_mean_power_w_kg", "wingate_fatigue_index_pct"]
FIELD_COLS = ["sprint_10m_s", "sprint_30m_s", "max_velocity_ms", "agility_505_s",
              "yoyo_ir1_distance_m", "swim_100m_free_s", "css_ms",
              "body_mass_kg", "sum7_skinfolds_mm"]

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
    "swim_100m_free_s": ("speed", False),
    "css_ms": ("aerobic", True),
    "body_mass_kg": ("body_comp", True),
    "sum7_skinfolds_mm": ("body_comp", False),
}

NOISE = {  # within-athlete typical error, as a fraction
    "sprint_10m_s": 0.012, "sprint_30m_s": 0.010, "max_velocity_ms": 0.015,
    "agility_505_s": 0.016, "imtp_relative_force_nkg": 0.045,
    "imtp_peak_force_n": 0.045, "imtp_rfd_0_250ms_ns": 0.10,
    "wingate_peak_power_w_kg": 0.035, "wingate_mean_power_w_kg": 0.030,
    "wingate_fatigue_index_pct": 0.08, "yoyo_ir1_distance_m": 0.06,
    "swim_100m_free_s": 0.008, "css_ms": 0.020,
    "body_mass_kg": 0.010, "sum7_skinfolds_mm": 0.045,
}

# which columns belong to which test type
TEST_OF = {
    "imtp_peak_force_n": "IMTP_test", "imtp_relative_force_nkg": "IMTP_test",
    "imtp_rfd_0_250ms_ns": "IMTP_test",
    "wingate_peak_power_w_kg": "wingate_test", "wingate_mean_power_w_kg": "wingate_test",
    "wingate_fatigue_index_pct": "wingate_test",
    "sprint_10m_s": "sprint_test", "sprint_30m_s": "sprint_test",
    "max_velocity_ms": "sprint_test",
    "agility_505_s": "agility_test",
    "yoyo_ir1_distance_m": "aerobic_test",
    "swim_100m_free_s": "swim_test", "css_ms": "swim_test",
    "body_mass_kg": "anthropometry", "sum7_skinfolds_mm": "anthropometry",
}


def _profile(sport: str, sex: str, squad: str, mass: float, rng) -> dict[str, float]:
    """Starting values for one athlete, from population-appropriate bands.

    Where a published reference exists in src/db/normative.sql, the band is
    centred on it. The comparison panel is worthless if the synthetic squad sits
    a standard deviation above every literature mean.
    """
    male = sex == "M"
    p: dict[str, float] = {"body_mass_kg": mass}

    if sport == "Swimming":
        distance = "Distance" in squad
        p["imtp_relative_force_nkg"] = rng.uniform(29, 38) if male else rng.uniform(24, 32)
        p["wingate_peak_power_w_kg"] = (
            (rng.uniform(10.0, 12.5) if distance else rng.uniform(11.5, 14.5)) if male
            else (rng.uniform(7.8, 9.8) if distance else rng.uniform(9.0, 11.5))
        )
        p["wingate_fatigue_index_pct"] = rng.uniform(28, 46) if distance else rng.uniform(38, 58)
        base100 = (52.5 if distance else 49.8) if male else (60.0 if distance else 56.5)
        p["swim_100m_free_s"] = base100 + rng.uniform(-1.4, 1.8)
        p["css_ms"] = (
            (rng.uniform(1.46, 1.60) if distance else rng.uniform(1.38, 1.50)) if male
            else (rng.uniform(1.30, 1.44) if distance else rng.uniform(1.24, 1.36))
        )
        p["sum7_skinfolds_mm"] = rng.uniform(40, 62) if male else rng.uniform(66, 92)

    elif sport == "Football":
        p["imtp_relative_force_nkg"] = rng.uniform(26, 34) if male else rng.uniform(18.5, 25.0)
        p["wingate_peak_power_w_kg"] = rng.uniform(11.5, 14.5) if male else rng.uniform(9.0, 12.0)
        p["wingate_fatigue_index_pct"] = rng.uniform(34, 54)
        p["sprint_10m_s"] = rng.uniform(1.86, 2.00) if male else rng.uniform(2.02, 2.28)
        p["max_velocity_ms"] = rng.uniform(8.4, 9.3) if male else rng.uniform(7.5, 8.4)
        p["agility_505_s"] = rng.uniform(2.32, 2.50) if male else rng.uniform(2.50, 2.88)
        # Yo-Yo IR1 was developed and validated on footballers; they score well.
        p["yoyo_ir1_distance_m"] = rng.uniform(1400, 2200) if male else rng.uniform(1000, 1900)
        p["sum7_skinfolds_mm"] = rng.uniform(42, 66) if male else rng.uniform(55, 88)

    elif sport == "Basketball":
        p["imtp_relative_force_nkg"] = rng.uniform(30, 40) if male else rng.uniform(26, 35)
        p["wingate_peak_power_w_kg"] = rng.uniform(11.0, 14.5) if male else rng.uniform(9.0, 12.0)
        p["wingate_fatigue_index_pct"] = rng.uniform(36, 56)
        p["sprint_10m_s"] = rng.uniform(1.78, 1.93) if male else rng.uniform(1.92, 2.08)
        p["max_velocity_ms"] = rng.uniform(8.6, 9.5) if male else rng.uniform(7.8, 8.7)
        p["agility_505_s"] = rng.uniform(2.36, 2.56) if male else rng.uniform(2.46, 2.66)
        p["yoyo_ir1_distance_m"] = rng.uniform(900, 1550) if male else rng.uniform(760, 1320)
        p["sum7_skinfolds_mm"] = rng.uniform(42, 70) if male else rng.uniform(58, 92)

    else:  # Athletics sprints
        p["imtp_relative_force_nkg"] = rng.uniform(36, 46) if male else rng.uniform(29, 38)
        p["wingate_peak_power_w_kg"] = rng.uniform(14.0, 17.0) if male else rng.uniform(10.5, 13.5)
        p["wingate_fatigue_index_pct"] = rng.uniform(44, 62)
        p["sprint_10m_s"] = rng.uniform(1.68, 1.80) if male else rng.uniform(1.86, 1.98)
        p["max_velocity_ms"] = rng.uniform(9.6, 10.7) if male else rng.uniform(8.5, 9.3)
        p["agility_505_s"] = rng.uniform(2.30, 2.48) if male else rng.uniform(2.42, 2.60)
        p["sum7_skinfolds_mm"] = rng.uniform(34, 56) if male else rng.uniform(48, 76)

    if "sprint_10m_s" in p:
        p["sprint_30m_s"] = p["sprint_10m_s"] * rng.uniform(2.38, 2.48)
    p["imtp_peak_force_n"] = p["imtp_relative_force_nkg"] * mass
    p["imtp_rfd_0_250ms_ns"] = p["imtp_peak_force_n"] * rng.uniform(2.4, 3.6)
    p["wingate_mean_power_w_kg"] = p["wingate_peak_power_w_kg"] * rng.uniform(0.66, 0.75)
    return p


def _value(metric: str, start: float, frac_through: float, code: str, rng) -> float:
    quality, higher_better = POLARITY[metric]
    drift = TRENDS.get(code, {}).get(quality, 0.0) * frac_through
    signed = drift if higher_better else -drift          # move the metric its own way
    return float(start * (1 + signed) * (1 + rng.normal(0, NOISE[metric])))


def _due(day_index: int, interval: int, offset: int) -> bool:
    return (day_index - offset) % interval == 0 and day_index >= offset


def _weekday_safe_offsets(days: list[date], rng) -> dict[str, int]:
    """Pick per-test offsets that land on a weekday.

    Every interval here is a multiple of seven, so an athlete's test day falls on
    the same weekday all season. A naive random offset meant whoever drew a
    Saturday was never tested at all. Choosing an offset in 0..4 does not fix it
    either, because which weekday index 0 lands on depends on the season start:
    pick the target weekday first, then derive the offset from the calendar.
    """
    out: dict[str, int] = {}
    for t in SCHEDULE:
        target = int(rng.integers(0, 5))                  # Monday..Friday
        out[t] = next(i for i, d in enumerate(days) if d.weekday() == target)
    return out


def write_batteries(
    out_dir: Path, athletes: list[tuple], days: list[date],
    sport_battery: dict[str, set[str]], rng,
) -> tuple[int, int]:
    """Write lab_tests.csv and field_tests.csv. Returns (lab_rows, field_rows)."""
    lab_rows: list[dict] = []
    field_rows: list[dict] = []
    n = len(days)

    for code, sport, sex, squad, mass, *_ in athletes:
        prof = _profile(sport, sex, squad, mass, rng)
        runs = sport_battery.get(sport, set())
        offs = _weekday_safe_offsets(days, rng)

        for i, d in enumerate(days):
            if d.weekday() > 4:
                continue
            frac = i / max(n - 1, 1)
            due = {t: (t in runs and _due(i, SCHEDULE[t], offs[t])) for t in SCHEDULE}

            def cell(col: str) -> object:
                if not due.get(TEST_OF[col]) or col not in prof:
                    return ""
                dp = 3 if col.endswith("_s") or col == "css_ms" else (
                    0 if col == "yoyo_ir1_distance_m" else 1)
                return round(_value(col, prof[col], frac, code, rng), dp)

            if any(due[t] for t in ("IMTP_test", "wingate_test")):
                lab_rows.append({"athlete_code": code, "date": d.isoformat(),
                                 **{c: cell(c) for c in LAB_COLS}})
            if any(due[t] for t in ("sprint_test", "agility_test", "aerobic_test",
                                    "anthropometry", "swim_test")):
                field_rows.append({"athlete_code": code, "date": d.isoformat(),
                                   **{c: cell(c) for c in FIELD_COLS}})

    # ---- injected faults, mirroring the force-plate ones -------------------
    blank_f = {c: "" for c in FIELD_COLS}
    blank_l = {c: "" for c in LAB_COLS}
    # A 0.94 s 10 m would be a world record by a distance: a gate fired early.
    field_rows.append({"athlete_code": "ATH-007", "date": days[150].isoformat(),
                       **blank_f, "sprint_10m_s": 0.94, "sprint_30m_s": 4.31,
                       "max_velocity_ms": 8.9})
    # 9999 is a data-entry sentinel, not a distance.
    field_rows.append({"athlete_code": "ATH-008", "date": days[152].isoformat(),
                       **blank_f, "yoyo_ir1_distance_m": 9999})
    # An athlete code that is not on the roster.
    lab_rows.append({"athlete_code": "ATH-777", "date": days[154].isoformat(),
                     **blank_l, "imtp_relative_force_nkg": 38.0, "imtp_peak_force_n": 3000.0})

    with (out_dir / "lab_tests.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["athlete_code", "date"] + LAB_COLS)
        w.writeheader(); w.writerows(lab_rows)
    with (out_dir / "field_tests.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["athlete_code", "date"] + FIELD_COLS)
        w.writeheader(); w.writerows(field_rows)
    return len(lab_rows), len(field_rows)
