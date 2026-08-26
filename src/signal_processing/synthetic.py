"""Synthetic force-plate traces.

Real athlete force-plate data cannot be committed to a public repository, so the
demo runs on physiologically-shaped synthetic traces. The trace is built from a
prescribed force waveform whose net impulse is solved analytically to hit a
target jump height -- which means the generator carries ground truth, and the
CMJ analyser can be tested against a known answer rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665

__all__ = ["SynthTrace", "synth_cmj_trace"]


@dataclass
class SynthTrace:
    time: np.ndarray
    force: np.ndarray
    sample_rate: float
    # ground truth
    true_jump_height_m: float
    true_takeoff_velocity_ms: float
    true_body_weight_n: float
    true_mass_kg: float
    true_flight_time_s: float
    true_onset_idx: int
    true_takeoff_idx: int
    true_landing_idx: int


def synth_cmj_trace(
    mass_kg: float = 78.0,
    jump_height_m: float = 0.36,
    sample_rate: float = 1000.0,
    quiet_s: float = 1.5,
    unweight_s: float = 0.40,
    rise_s: float = 0.22,
    fall_s: float = 0.12,
    unweight_depth_frac: float = 0.35,
    landing_peak_bw: float = 4.0,
    noise_n: float = 2.0,
    drift_n: float = 0.0,
    tail_s: float = 0.6,
    seed: int | None = 7,
) -> SynthTrace:
    """Generate one CMJ vertical-GRF trace with known ground truth.

    The ground phase is three shaped segments -- unweighting dip, propulsive
    rise, fall to take-off. Peak force is solved so that the net impulse equals
    ``m * sqrt(2 g h)``, so the trace genuinely encodes the requested jump
    height rather than approximating it.
    """
    rng = np.random.default_rng(seed)
    fs = float(sample_rate)
    dt = 1.0 / fs
    bw = mass_kg * G
    v_to = float(np.sqrt(2 * G * jump_height_m))
    required_impulse = mass_kg * v_to

    d1, d2, d3 = unweight_s, rise_s, fall_s
    dip = unweight_depth_frac * bw
    k = 2.0 / np.pi  # mean value of sin(pi u) and of sin/cos(pi u /2) over [0,1]

    # J = Fpk*k*(d2+d3) - k*dip*d1 - bw*(k*d2 + d3)   ->  solve for Fpk
    f_peak = (required_impulse + k * dip * d1 + bw * (k * d2 + d3)) / (k * (d2 + d3))

    n_quiet = int(round(quiet_s * fs))
    n1, n2, n3 = int(round(d1 * fs)), int(round(d2 * fs)), int(round(d3 * fs))

    quiet = np.full(n_quiet, bw)
    u1 = np.arange(n1) / n1
    seg1 = bw - dip * np.sin(np.pi * u1)                       # dip below BW and back
    u2 = np.arange(n2) / n2
    seg2 = bw + (f_peak - bw) * np.sin(np.pi / 2 * u2)         # BW -> peak
    u3 = np.arange(n3) / n3
    seg3 = f_peak * np.cos(np.pi / 2 * u3)                     # peak -> 0 at take-off

    flight_time = 2 * v_to / G
    n_flight = int(round(flight_time * fs))
    flight = np.zeros(n_flight)

    n_tail = int(round(tail_s * fs))
    tl = np.arange(n_tail) * dt
    landing = bw + (landing_peak_bw * bw - bw) * np.exp(-tl / 0.055) * np.cos(2 * np.pi * 7.0 * tl)

    force = np.concatenate([quiet, seg1, seg2, seg3, flight, landing])
    force = np.maximum(force, 0.0)  # a plate cannot read negative vertical force

    if drift_n:
        force = force + np.linspace(0.0, drift_n, force.size)
    if noise_n:
        force = force + rng.normal(0.0, noise_n, force.size)
        force = np.maximum(force, 0.0)

    onset = n_quiet
    takeoff = n_quiet + n1 + n2 + n3
    landing_idx = takeoff + n_flight

    return SynthTrace(
        time=np.arange(force.size) * dt,
        force=force,
        sample_rate=fs,
        true_jump_height_m=jump_height_m,
        true_takeoff_velocity_ms=v_to,
        true_body_weight_n=bw,
        true_mass_kg=mass_kg,
        true_flight_time_s=flight_time,
        true_onset_idx=onset,
        true_takeoff_idx=takeoff,
        true_landing_idx=landing_idx,
    )
