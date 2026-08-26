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

from src.signal_processing.synthetic import synth_cmj_trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
TRACES = OUT / "force_plate"
SAMPLES = OUT / "sample_traces"

SEASON_END = date(2026, 8, 25)
SEASON_DAYS = 90
SAMPLE_RATE = 1000

ATHLETES = [
    # code,      sport,        sex, squad,            mass, baseline h, fatigue block
    ("ATH-001", "Athletics", "M", "Sprints",        78.0, 0.42, None),
    ("ATH-002", "Athletics", "M", "Sprints",        84.5, 0.38, (52, 64)),
    ("ATH-003", "Athletics", "F", "Sprints",        61.0, 0.31, None),
    ("ATH-004", "Athletics", "F", "Jumps",          58.5, 0.36, None),
    ("ATH-005", "Athletics", "M", "Jumps",          75.0, 0.47, None),
    ("ATH-006", "Athletics", "M", "Jumps",          80.0, 0.44, (70, 84)),
    ("ATH-007", "Netball",   "F", "Netball-Senior", 72.0, 0.28, None),
    ("ATH-008", "Netball",   "F", "Netball-Senior", 68.5, 0.30, None),
    ("ATH-009", "Netball",   "F", "Netball-Senior", 75.5, 0.26, None),
    ("ATH-010", "Netball",   "F", "Netball-Senior", 70.0, 0.29, (30, 40)),
    ("ATH-011", "Athletics", "M", "Sprints",        82.0, 0.40, None),
    ("ATH-012", "Athletics", "F", "Jumps",          60.0, 0.34, None),
]

# Athletes whose training load spikes late in the block, so ACWR has something
# real to flag rather than a number that never leaves the sweet spot.
LOAD_SPIKE = {"ATH-002": (50, 66), "ATH-010": (28, 42)}


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
    sample_taken = 0

    for code, _sport, _sex, _squad, mass, base_h, fatigue in ATHLETES:
        for i, d in enumerate(days):
            if d.weekday() not in (0, 3):          # test Mondays and Thursdays
                continue
            if rng.random() < 0.12:                 # athletes miss sessions
                continue

            h = base_h + rng.normal(0, 0.012)       # day-to-day biological noise
            h += 0.00035 * i                        # slow training adaptation
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
            if sample_taken < 3 and i > 40:
                _write_trace(SAMPLES / f"{code}_{d.isoformat()}.csv", tr, rng)
                sample_taken += 1

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
    _write_trace(TRACES / f"ATH-999_{days[70].isoformat()}.csv", tr, rng)
    n += 1

    # 2. A flat trace: the athlete stepped on the plate and the trial was aborted.
    flat = np.full(2400, 735.0) + rng.normal(0, 2.0, 2400)
    path = TRACES / f"ATH-003_{days[74].isoformat()}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "Fz_left_N", "Fz_right_N"])
        for i, f in enumerate(flat):
            w.writerow([f"{i / SAMPLE_RATE:.4f}", f"{f * 0.5:.2f}", f"{f * 0.5:.2f}"])
    n += 1

    # 3. A physiologically impossible jump -- a mis-set plate amplifier gain.
    tr = synth_cmj_trace(mass_kg=70, jump_height_m=1.55, quiet_s=1.0, tail_s=0.4, seed=903)
    _write_trace(TRACES / f"ATH-007_{days[78].isoformat()}.csv", tr, rng)
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
                duration *= 1.45
                srpe = float(np.clip(srpe + 1.6, 1, 10))
            rows.append((code, d.isoformat(), round(duration, 1), round(srpe, 1)))

    # 4. sRPE outside the Borg CR-10 scale -- a data-entry slip.
    rows.append(("ATH-005", days[60].isoformat(), 95.0, 14.0))
    rows.append(("ATH-008", days[61].isoformat(), 88.0, -2.0))
    # 5. The same athlete-day submitted twice with different numbers. Anchored
    #    on a day that is genuinely already in the diary -- a duplicate planted
    #    on a rest day is not a duplicate, and the test would silently pass.
    first_ath1 = next(r for r in rows if r[0] == "ATH-001")
    rows.append(("ATH-001", first_ath1[1], 110.0, 8.5))
    # 6. A negative duration.
    rows.append(("ATH-011", days[63].isoformat(), -45.0, 6.0))

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
    size_mb = sum(p.stat().st_size for p in TRACES.glob("*.csv")) / 1e6
    print(f"roster        : {len(ATHLETES)} athletes -> data/synthetic/athletes.csv")
    print(f"force traces  : {n_traces} files ({size_mb:.1f} MB) -> data/synthetic/force_plate/")
    print(f"sample traces : 3 files (committed) -> data/synthetic/sample_traces/")
    print(f"sRPE diary    : {n_srpe} rows -> data/synthetic/srpe_diary.csv")
    print("\ninjected faults: unknown athlete code, flat trace, impossible jump,")
    print("                sRPE out of 0-10, negative duration, duplicated athlete-day")


if __name__ == "__main__":
    main()
