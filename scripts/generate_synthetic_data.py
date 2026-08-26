"""Generate a synthetic squad-season dataset.

Produces the three things a real service would receive from three different
places: an athlete roster, one raw force-plate export per CMJ trial, and an
sRPE training diary. The force files are genuine waveforms, not summary rows --
the pipeline has to run the signal-processing module to get a jump height out
of them, which is the point.

Deterministic: a fixed seed means anyone can regenerate byte-identical data, so
the bulk force files do not need to live in version control.

**Dirty rows are injected on purpose.** A validation layer that has never
rejected anything is not evidence of anything, so the generator plants an
unknown athlete code, out-of-range sRPE values, a duplicated diary row, a flat
trace with no jump in it, and one physiologically impossible trial.
"""

from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._physical_tests import TRENDS, write_batteries  # noqa: E402
from src.signal_processing.synthetic import synth_cmj_trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
TRACES = OUT / "force_plate"
SAMPLES = OUT / "sample_traces"

SEASON_END = date(2026, 8, 25)
SEASON_DAYS = 180
SAMPLE_RATE = 1000

ATHLETES = [
    # Baseline values are anchored on the published means loaded in
    # src/db/normative.sql, so the squad sits inside the reference bands rather
    # than floating above them. Before this, the demo had professional male
    # basketballers jumping 50 cm against a literature mean of 39.2 cm and
    # female footballers running 10 m in 1.88 s against a reported 2.14 s --
    # every athlete would have read as world class, which tells a coach nothing.
    #
    # code,      sport,        sex, squad,                mass, baseline CMJ h, fatigue block
    # --- Swimming: strong vertically, unremarkable on land-running tests -----
    ("ATH-001", "Swimming",   "M", "Swimming - Sprint",   79.0, 0.41, None),
    ("ATH-002", "Swimming",   "M", "Swimming - Sprint",   82.5, 0.39, (120, 134)),
    ("ATH-003", "Swimming",   "F", "Swimming - Sprint",   66.0, 0.31, None),
    ("ATH-004", "Swimming",   "F", "Swimming - Distance", 62.5, 0.28, None),
    ("ATH-005", "Swimming",   "M", "Swimming - Distance", 74.0, 0.35, None),
    ("ATH-006", "Swimming",   "F", "Swimming - Distance", 64.0, 0.29, None),
    # --- Football: the population Yo-Yo IR1 was actually validated on --------
    ("ATH-007", "Football",   "F", "Football - Women",    63.0, 0.28, None),
    ("ATH-008", "Football",   "F", "Football - Women",    58.5, 0.32, None),
    ("ATH-009", "Football",   "F", "Football - Women",    67.5, 0.26, (174, 179)),
    ("ATH-010", "Football",   "F", "Football - Women",    61.0, 0.30, (96, 110)),
    # --- Basketball: tall, heavy, highest jumps in the programme -------------
    ("ATH-011", "Basketball", "M", "Basketball - Men",    96.0, 0.40, None),
    ("ATH-012", "Basketball", "M", "Basketball - Men",   102.0, 0.37, None),
    ("ATH-013", "Basketball", "M", "Basketball - Men",    89.5, 0.43, None),
    # --- Athletics sprints: fastest on the track, worst on the Yo-Yo ---------
    ("ATH-014", "Athletics",  "M", "Athletics - Sprints", 80.0, 0.46, (176, 179)),
    ("ATH-015", "Athletics",  "M", "Athletics - Sprints", 77.5, 0.44, None),
    ("ATH-016", "Athletics",  "F", "Athletics - Sprints", 59.0, 0.36, None),
]

# Athletes whose training load spikes late in the block, so ACWR has something
# real to flag rather than a number that never leaves the sweet spot.
LOAD_SPIKE = {"ATH-002": (118, 136), "ATH-010": (94, 112), "ATH-009": (172, 179)}

# Which batteries each sport actually runs. A land-based Yo-Yo IR1 and a 10 m
# sprint say very little about a swimmer; running them anyway would fill the
# database with numbers no coach would act on.
SPORT_BATTERY = {
    "Swimming":   {"IMTP_test", "wingate_test", "anthropometry", "swim_test"},
    "Football":   {"IMTP_test", "wingate_test", "sprint_test", "agility_test",
                   "aerobic_test", "anthropometry"},
    "Basketball": {"IMTP_test", "wingate_test", "sprint_test", "agility_test",
                   "aerobic_test", "anthropometry"},
    "Athletics":  {"IMTP_test", "wingate_test", "sprint_test", "agility_test",
                   "anthropometry"},
}


def season_dates() -> list[date]:
    return [SEASON_END - timedelta(days=SEASON_DAYS - 1 - i) for i in range(SEASON_DAYS)]


def write_roster() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "athletes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["athlete_code", "sport", "sex", "squad"])
        for code, sport, sex, squad, *_ in ATHLETES:
            w.writerow([code, sport, sex, squad])


def write_force_traces(rng: np.random.Generator) -> int:
    TRACES.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for old in list(TRACES.glob("*.csv")):
        old.unlink()

    days = season_dates()
    written = 0
    last_for_athlete: dict[str, tuple] = {}

    for code, _sport, _sex, _squad, mass, base_h, fatigue in ATHLETES:
        for i, d in enumerate(days):
            if d.weekday() not in (0, 3):          # test Mondays and Thursdays
                continue
            if rng.random() < 0.12:                 # athletes miss sessions
                continue

            # Seasonal drift comes from the same per-athlete trend table the
            # laboratory batteries use. A blanket adaptation term applied to
            # everyone made a "power declining" athlete post a 22% jump-height
            # gain -- the two generators were telling different stories about
            # the same person.
            drift = TRENDS.get(code, {}).get("power", 0.0) * (i / max(len(days) - 1, 1))
            h = base_h * (1 + drift) + rng.normal(0, 0.012)
            if fatigue and fatigue[0] <= i <= fatigue[1]:
                # Neuromuscular fatigue: a real, sustained depression, not a blip
                h -= 0.030 + rng.normal(0, 0.006)

            tr = synth_cmj_trace(
                mass_kg=mass + rng.normal(0, 0.4),
                jump_height_m=max(h, 0.10),
                sample_rate=SAMPLE_RATE,
                quiet_s=1.0,
                tail_s=0.4,
                unweight_s=0.40 + rng.normal(0, 0.03),
                rise_s=0.22 + rng.normal(0, 0.02),
                noise_n=rng.uniform(1.5, 3.0),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            _write_trace(TRACES / f"{code}_{d.isoformat()}.csv", tr, rng)
            written += 1
            # The most recent trial per athlete is also committed, so the
            # deployed dashboard can draw a real force-time curve for anyone
            # without shipping all 16 MB of raw traces.
            last_for_athlete[code] = (d, tr)

    for old in list(SAMPLES.glob("*.csv")):
        old.unlink()
    for code, (d, tr) in last_for_athlete.items():
        _write_trace(SAMPLES / f"{code}_{d.isoformat()}.csv", tr, rng)

    written += _inject_bad_traces(rng, days)
    return written


def _write_trace(path: Path, tr, rng: np.random.Generator) -> None:
    """Split system force across two plates, as a dual-plate system exports it."""
    share = float(np.clip(rng.normal(0.50, 0.03), 0.40, 0.60))
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "Fz_left_N", "Fz_right_N"])
        for t, f in zip(tr.time, tr.force):
            w.writerow([f"{t:.4f}", f"{f * share:.2f}", f"{f * (1 - share):.2f}"])


def _inject_bad_traces(rng: np.random.Generator, days: list[date]) -> int:
    """Plant the failures the validation layer is supposed to catch."""
    n = 0

    # 1. An athlete code that is not on the roster -- a typo at collection time.
    tr = synth_cmj_trace(mass_kg=75, jump_height_m=0.35, quiet_s=1.0, tail_s=0.4, seed=901)
    _write_trace(TRACES / f"ATH-999_{days[160].isoformat()}.csv", tr, rng)
    n += 1

    # 2. A flat trace: the athlete stepped on the plate and the trial was aborted.
    flat = np.full(2400, 735.0) + rng.normal(0, 2.0, 2400)
    path = TRACES / f"ATH-003_{days[164].isoformat()}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "Fz_left_N", "Fz_right_N"])
        for i, f in enumerate(flat):
            w.writerow([f"{i / SAMPLE_RATE:.4f}", f"{f * 0.5:.2f}", f"{f * 0.5:.2f}"])
    n += 1

    # 3. A physiologically impossible jump -- a mis-set plate amplifier gain.
    tr = synth_cmj_trace(mass_kg=70, jump_height_m=1.55, quiet_s=1.0, tail_s=0.4, seed=903)
    _write_trace(TRACES / f"ATH-013_{days[168].isoformat()}.csv", tr, rng)
    n += 1

    return n


def write_srpe_diary(rng: np.random.Generator) -> int:
    days = season_dates()
    rows: list[tuple] = []

    for code, *_rest in [(a[0],) for a in ATHLETES]:
        spike = LOAD_SPIKE.get(code)
        for i, d in enumerate(days):
            wd = d.weekday()
            if wd == 6 or (wd == 5 and rng.random() < 0.6):
                continue                                    # rest days
            duration = float(np.clip(rng.normal(80, 18), 30, 150))
            srpe = float(np.clip(rng.normal(6.0, 1.3), 1, 10))
            if wd in (1, 4):
                srpe = float(np.clip(srpe + 1.4, 1, 10))     # hard days
            if spike and spike[0] <= i <= spike[1]:
                duration *= 1.85
                srpe = float(np.clip(srpe + 2.6, 1, 10))
            rows.append((code, d.isoformat(), round(duration, 1), round(srpe, 1)))

    # 4. sRPE outside the Borg CR-10 scale -- a data-entry slip.
    rows.append(("ATH-005", days[150].isoformat(), 95.0, 14.0))
    rows.append(("ATH-008", days[151].isoformat(), 88.0, -2.0))
    # 5. The same athlete-day submitted twice with different numbers. Anchored
    #    on a day that is genuinely already in the diary -- a duplicate planted
    #    on a rest day is not a duplicate, and the test would silently pass.
    first_ath1 = next(r for r in rows if r[0] == "ATH-001")
    rows.append(("ATH-001", first_ath1[1], 110.0, 8.5))
    # 6. A negative duration.
    rows.append(("ATH-011", days[153].isoformat(), -45.0, 6.0))

    with (OUT / "srpe_diary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["athlete_code", "date", "duration_min", "srpe"])
        w.writerows(rows)
    return len(rows)


def main() -> None:
    rng = np.random.default_rng(20260826)
    write_roster()
    n_traces = write_force_traces(rng)
    n_srpe = write_srpe_diary(rng)
    n_lab, n_field = write_batteries(OUT, ATHLETES, season_dates(), SPORT_BATTERY, rng)
    size_mb = sum(p.stat().st_size for p in TRACES.glob("*.csv")) / 1e6
    print(f"roster        : {len(ATHLETES)} athletes -> data/synthetic/athletes.csv")
    print(f"force traces  : {n_traces} files ({size_mb:.1f} MB) -> data/synthetic/force_plate/")
    print(f"sample traces : {len(list(SAMPLES.glob('*.csv')))} files (committed) -> data/synthetic/sample_traces/")
    print(f"sRPE diary    : {n_srpe} rows -> data/synthetic/srpe_diary.csv")
    print(f"lab tests     : {n_lab} rows -> data/synthetic/lab_tests.csv   (IMTP, Wingate)")
    print(f"field tests   : {n_field} rows -> data/synthetic/field_tests.csv (sprint, 505, Yo-Yo, anthropometry)")
    print("\ninjected faults: unknown athlete code, flat trace, impossible jump,")
    print("                sRPE out of 0-10, negative duration, duplicated athlete-day")


if __name__ == "__main__":
    main()
