"""Tests for the CMJ analyser.

The synthetic generator solves peak force analytically so that net impulse
equals m*sqrt(2gh); the requested jump height is therefore genuine ground truth
and these are accuracy tests, not just smoke tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal_processing.cmj import analyse_cmj
from src.signal_processing.synthetic import synth_cmj_trace


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mass,height",
    [(55, 0.22), (70, 0.30), (78, 0.36), (95, 0.48), (60, 0.08), (90, 0.65)],
)
def test_recovers_known_jump_height(mass, height):
    tr = synth_cmj_trace(mass_kg=mass, jump_height_m=height, seed=int(mass))
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    assert r.is_valid, r.quality_flags
    assert r.jump_height_m == pytest.approx(height, abs=0.005)  # within 5 mm
    assert r.body_weight_n == pytest.approx(tr.true_body_weight_n, rel=0.005)
    assert r.takeoff_velocity_ms == pytest.approx(tr.true_takeoff_velocity_ms, rel=0.02)


@pytest.mark.parametrize("fs", [500, 1000, 2000])
def test_sample_rate_invariance(fs):
    tr = synth_cmj_trace(mass_kg=72, jump_height_m=0.34, sample_rate=fs, seed=3)
    r = analyse_cmj(tr.force, sample_rate=fs)
    assert r.jump_height_m == pytest.approx(0.34, abs=0.006)


# ---------------------------------------------------------------------------
# regression: the onset bug
# ---------------------------------------------------------------------------
def test_onset_is_not_fooled_by_bodyweight_recrossing():
    """Vertical force necessarily crosses back through body weight between the
    unweighting and braking phases. A backward search from take-off stops at
    that crossing, truncates the negative impulse and inflates jump height by
    tens of centimetres. Onset must land at the true start of movement.
    """
    tr = synth_cmj_trace(mass_kg=78, jump_height_m=0.36, seed=11)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    # 30 ms lookback is applied deliberately, so onset should sit slightly early
    lookback = int(0.030 * tr.sample_rate)
    assert -lookback - 15 <= r.indices["onset"] - tr.true_onset_idx <= 15
    # and the metric it protects must be right
    assert r.jump_height_m == pytest.approx(0.36, abs=0.005)


def test_takeoff_and_landing_indices_are_exact():
    """Threshold crossings are detected on the raw trace. Detecting them on the
    filtered trace pulls both inward and shortens measured flight time."""
    tr = synth_cmj_trace(mass_kg=78, jump_height_m=0.36, seed=12)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    assert abs(r.indices["takeoff"] - tr.true_takeoff_idx) <= 3
    assert abs(r.indices["landing"] - tr.true_landing_idx) <= 3


def test_two_height_methods_agree_on_clean_data():
    """Impulse-momentum and flight-time heights should agree on a synthetic
    trace with a symmetric landing. Divergence in real data means the athlete
    changed posture in the air, which is why both are reported."""
    tr = synth_cmj_trace(mass_kg=80, jump_height_m=0.40, seed=13)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    assert abs(r.jump_height_m - r.jump_height_flight_time_m) < 0.01


# ---------------------------------------------------------------------------
# phase structure
# ---------------------------------------------------------------------------
def test_phase_durations_are_ordered_and_sum_to_contraction_time():
    tr = synth_cmj_trace(mass_kg=75, jump_height_m=0.33, seed=14)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    total = r.unweighting_duration_s + r.ecc_duration_s + r.con_duration_s
    assert total == pytest.approx(r.contraction_time_s, abs=2.0 / tr.sample_rate)
    i = r.indices
    assert i["onset"] < i["min_velocity"] < i["zero_velocity"] < i["takeoff"] < i["landing"]
    assert r.countermovement_depth_m < 0  # the athlete must go down first


# ---------------------------------------------------------------------------
# rejection / quality control
# ---------------------------------------------------------------------------
def test_rejects_trace_with_no_flight_phase():
    quiet = np.full(3000, 700.0) + np.random.default_rng(0).normal(0, 2, 3000)
    r = analyse_cmj(quiet, 1000)
    assert not r.is_valid
    assert any("no flight phase" in f for f in r.quality_flags)


def test_rejects_physiologically_impossible_height():
    tr = synth_cmj_trace(mass_kg=70, jump_height_m=1.60, seed=15)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    assert not r.is_valid
    assert any("outside 0.05-1.20" in f for f in r.quality_flags)


def test_warns_on_unstable_weighing_period():
    tr = synth_cmj_trace(mass_kg=70, jump_height_m=0.30, noise_n=40, seed=16)
    r = analyse_cmj(tr.force, sample_rate=tr.sample_rate)
    assert any("unstable quiet standing" in f for f in r.quality_flags)


def test_rejects_nan_input():
    tr = synth_cmj_trace(seed=17)
    bad = tr.force.copy()
    bad[500] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        analyse_cmj(bad, 1000)


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def test_dual_plate_dataframe_with_time_column():
    """A dual force-plate export: one time column, one Fz channel per plate.
    The two plates measure one athlete, so system force is their sum."""
    tr = synth_cmj_trace(mass_kg=78, jump_height_m=0.36, seed=18)
    df = pd.DataFrame(
        {
            "time_s": tr.time,
            "Fz_left": tr.force * 0.52,
            "Fz_right": tr.force * 0.48,
        }
    )
    r = analyse_cmj(df)  # sample rate inferred from the time column
    assert r.jump_height_m == pytest.approx(0.36, abs=0.006)
    assert r.sample_rate_hz == pytest.approx(1000.0, rel=1e-6)


def test_to_dict_is_flat_and_scalar():
    """The ingest pipeline writes this dict into the long-format metrics table,
    so every value must be a scalar the database can store."""
    tr = synth_cmj_trace(seed=19)
    d = analyse_cmj(tr.force, 1000).to_dict()
    assert "jump_height_m" in d and "rsi_mod" in d
    assert not any(k.startswith("_") for k in d)
    assert all(isinstance(v, (int, float)) for v in d.values())
