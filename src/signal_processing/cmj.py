"""Countermovement jump (CMJ) analysis from raw vertical ground reaction force.

The input is a raw force-plate trace: a time column and one or more vertical
force (Fz) channels. Everything downstream -- jump height, RSI-mod, phase
durations -- is derived here, so this module is the place where sports-science
judgement is encoded rather than assumed.

Method notes
------------
Jump height uses the **impulse-momentum** method, not flight time. Net vertical
impulse from movement onset to take-off gives take-off velocity directly:

    J = integral (F(t) - BW) dt          v_to = J / m          h = v_to^2 / 2g

Flight time is also reported as a cross-check. The two disagree when the athlete
lands in a different posture than they took off in (tucking the legs inflates
flight time), so a large discrepancy is a data-quality signal, not noise.

Movement onset uses the 5-SD threshold of the quiet-standing period with a 30 ms
lookback (Owen et al. 2014); taking onset too late truncates the unweighting
impulse and systematically underestimates jump height.

References
----------
Owen et al. (2014) J Strength Cond Res -- onset detection and BW quantification.
McMahon et al. (2018) Strength Cond J -- CMJ phase definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

G = 9.80665  # m/s^2

__all__ = ["CMJResult", "analyse_cmj", "prepare_force"]


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def prepare_force(
    data: Any,
    sample_rate: float | None = None,
    time_col: str | None = None,
    force_cols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Coerce assorted force-plate exports into (time, summed_Fz, sample_rate).

    Accepts a 1-D array, a 2-D array (samples x plates), or a DataFrame. Multiple
    Fz channels are summed: a dual-plate setup measures one athlete, and the
    system-level force is what the impulse calculation needs.
    """
    t = None

    if hasattr(data, "columns"):  # pandas DataFrame
        cols = list(data.columns)
        if time_col is None:
            time_col = next(
                (c for c in cols if str(c).lower() in {"time", "t", "time_s", "timestamp"}),
                None,
            )
        if force_cols is None:
            candidates = [c for c in cols if c != time_col]
            fz = [c for c in candidates if "fz" in str(c).lower() or "force" in str(c).lower()]
            force_cols = fz or candidates
        fz_arr = np.asarray(data[list(force_cols)], dtype=float)
        if time_col is not None:
            t = np.asarray(data[time_col], dtype=float)
    else:
        fz_arr = np.asarray(data, dtype=float)

    if fz_arr.ndim == 2:
        fz_arr = fz_arr.sum(axis=1)
    fz_arr = np.asarray(fz_arr, dtype=float).ravel()

    if t is not None and t.size == fz_arr.size and t.size > 1:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            raise ValueError("time column is not monotonically increasing")
        fs = 1.0 / dt
    elif sample_rate is not None:
        fs = float(sample_rate)
        t = np.arange(fz_arr.size, dtype=float) / fs
    else:
        raise ValueError("provide either a time column or sample_rate")

    if not np.all(np.isfinite(fz_arr)):
        raise ValueError("force trace contains NaN/Inf")

    return t, fz_arr, fs


def _lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass. filtfilt so phase is not shifted -- a phase
    shift would move the detected onset and take-off instants."""
    nyq = fs / 2.0
    if cutoff >= nyq:
        return x.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.size <= padlen:
        return x.copy()
    return filtfilt(b, a, x)


def _quiet_standing(
    f: np.ndarray, fs: float, window_s: float = 1.0, search_s: float = 2.0
) -> tuple[float, float, int, int]:
    """Body weight from the most stable `window_s` window inside the first
    `search_s` seconds -- more robust than blindly taking the first second, which
    often contains the athlete still settling onto the plate."""
    w = max(int(round(window_s * fs)), 10)
    limit = min(int(round(search_s * fs)), f.size)
    if limit < w:
        w, limit = max(f.size // 4, 10), f.size
    best_sd, best_i = np.inf, 0
    for i in range(0, limit - w + 1, max(int(fs // 100), 1)):
        sd = float(np.std(f[i : i + w]))
        if sd < best_sd:
            best_sd, best_i = sd, i
    return float(np.mean(f[best_i : best_i + w])), best_sd, best_i, best_i + w


def _find_flight(f: np.ndarray, fs: float, thr: float, min_flight_s: float = 0.10):
    """Longest run of near-zero force lasting at least `min_flight_s`."""
    below = f < thr
    if not below.any():
        return None
    edges = np.diff(below.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if below[0]:
        starts.insert(0, 0)
    if below[-1]:
        ends.append(f.size)
    min_len = int(round(min_flight_s * fs))
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def _find_onset(
    f: np.ndarray,
    bw: float,
    thr: float,
    lo: int,
    takeoff_idx: int,
    fs: float,
    lookback_s: float,
    sustain_s: float = 0.030,
) -> tuple[int, str]:
    """Locate movement onset.

    Searching backwards from take-off is the obvious approach and it is wrong:
    vertical force necessarily crosses back through body weight between the
    unweighting and braking phases, so a backward search stops at that crossing,
    truncates the negative impulse, and inflates jump height by tens of cm.

    Anchor instead on the deepest unweighting point -- which must precede the
    propulsive force peak -- because between true onset and that minimum the
    force never returns to body weight. Walk backwards from there.

    A trace with no unweighting at all (a squat jump fed to a CMJ analyser)
    has no such anchor, so fall back to a forward scan for the first
    *sustained* departure from body weight; a single noisy sample must not
    define onset.
    """
    lo = max(min(lo, takeoff_idx - 2), 0)
    n_look = int(round(lookback_s * fs))

    peak_idx = int(np.argmax(f[lo:takeoff_idx])) + lo
    if peak_idx > lo:
        min_idx = int(np.argmin(f[lo:peak_idx])) + lo
        if f[min_idx] < bw - thr:
            onset = lo
            for i in range(min_idx, lo, -1):
                if abs(f[i] - bw) <= thr:
                    onset = i
                    break
            return max(onset - n_look, 0), "countermovement"

    n_sus = max(int(round(sustain_s * fs)), 1)
    dev = np.abs(f - bw) > thr
    for i in range(lo, max(takeoff_idx - n_sus, lo + 1)):
        if dev[i : i + n_sus].all():
            return max(i - n_look, 0), "no_countermovement"
    return max(lo - n_look, 0), "fallback"


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class CMJResult:
    # anthropometrics / calibration
    body_weight_n: float
    body_mass_kg: float
    bw_sd_n: float
    # primary outcomes
    jump_height_m: float
    jump_height_flight_time_m: float
    takeoff_velocity_ms: float
    rsi_mod: float
    # kinetics
    peak_force_n: float
    peak_force_bw: float
    peak_power_w: float
    peak_power_w_kg: float
    net_impulse_ns: float
    # phase structure
    unweighting_duration_s: float
    ecc_duration_s: float
    con_duration_s: float
    contraction_time_s: float
    flight_time_s: float
    countermovement_depth_m: float
    # provenance
    sample_rate_hz: float
    filter_cutoff_hz: float
    indices: dict[str, int] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    # diagnostic series (excluded from to_dict)
    _time: np.ndarray | None = field(default=None, repr=False)
    _force: np.ndarray | None = field(default=None, repr=False)
    _velocity: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Flat scalar dict -- the shape the ingest pipeline writes to the
        long-format performance_metrics table."""
        d = asdict(self)
        for k in ("_time", "_force", "_velocity", "indices", "quality_flags"):
            d.pop(k, None)
        return d

    @property
    def is_valid(self) -> bool:
        return not any(fl.startswith("REJECT") for fl in self.quality_flags)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def analyse_cmj(
    force_data: Any,
    sample_rate: float = 1000.0,
    *,
    time_col: str | None = None,
    force_cols: Sequence[str] | None = None,
    filter_cutoff_hz: float = 50.0,
    onset_sd_multiple: float = 5.0,
    onset_lookback_s: float = 0.030,
) -> CMJResult:
    """Compute CMJ metrics from a raw vertical ground reaction force trace.

    Parameters
    ----------
    force_data : array or DataFrame
        Raw trace. 1-D Fz, 2-D (samples x plates), or a DataFrame with an
        optional time column and one or more Fz columns.
    sample_rate : float
        Hz. Ignored if a usable time column is present.
    filter_cutoff_hz : float
        4th-order zero-lag Butterworth low-pass. 50 Hz retains the propulsive
        waveform while removing plate resonance.

    Returns
    -------
    CMJResult
        Check ``.is_valid`` before trusting the numbers; ``.quality_flags``
        explains any rejection.
    """
    t, raw, fs = prepare_force(force_data, sample_rate, time_col, force_cols)
    f = _lowpass(raw, fs, filter_cutoff_hz)
    flags: list[str] = []

    # --- body weight from quiet standing -------------------------------
    bw, bw_sd, qs0, qs1 = _quiet_standing(f, fs)
    if bw <= 0:
        raise ValueError("non-positive body weight from quiet standing; check units/sign")
    mass = bw / G
    # 1% of BW is the usual acceptance limit for a stable weighing period
    if bw_sd > 0.01 * bw:
        flags.append("WARN: unstable quiet standing (SD > 1% BW); BW may be biased")

    # --- flight phase --------------------------------------------------
    # Threshold-crossing events are detected on the RAW trace, not the filtered
    # one. A zero-lag filter smears sharp edges symmetrically in time, so the
    # landing impact bleeds backwards into the flight phase and take-off/landing
    # are both pulled inward -- measured flight time comes out systematically
    # short. Integration still uses the filtered trace, where smoothing helps.
    raw_sd = float(np.std(raw[qs0:qs1]))
    flight_thr = max(10.0, onset_sd_multiple * raw_sd)
    flight = _find_flight(raw, fs, flight_thr)
    if flight is None:
        # Report the numbers a practitioner needs to tell the two causes apart:
        # a genuine non-jump reads near BW throughout, whereas baseline drift or
        # a bad zero leaves the plate reading tens of newtons while airborne.
        fmin = float(np.min(raw[qs1:]))
        flags.append(
            "REJECT: no flight phase detected -- minimum force after the weighing "
            f"period was {fmin:.1f} N against a {flight_thr:.1f} N threshold "
            f"({fmin / bw * 100:.1f}% BW). Either this is not a jump, or the plate "
            "has baseline drift / a bad zero."
        )
        return _degenerate(bw, mass, bw_sd, fs, filter_cutoff_hz, flags, t, f)
    takeoff_idx, landing_idx = flight
    flight_time = (landing_idx - takeoff_idx) / fs

    # --- movement onset ------------------------------------------------
    thr = onset_sd_multiple * bw_sd
    onset_idx, onset_mode = _find_onset(
        f, bw, thr, qs0, takeoff_idx, fs, onset_lookback_s
    )
    if onset_mode == "no_countermovement":
        flags.append("WARN: no unweighting phase detected; trace may be a squat jump")
    elif onset_mode == "fallback":
        flags.append("REJECT: movement onset could not be located")

    if takeoff_idx - onset_idx < int(0.10 * fs):
        flags.append("WARN: contraction shorter than 100 ms; onset detection may have failed")

    # --- kinematics by numerical integration ---------------------------
    seg = slice(onset_idx, takeoff_idx + 1)
    accel = (f[seg] - bw) / mass
    vel = np.concatenate([[0.0], cumulative_trapezoid(accel, dx=1.0 / fs)])
    disp = np.concatenate([[0.0], cumulative_trapezoid(vel, dx=1.0 / fs)])

    v_takeoff = float(vel[-1])
    if v_takeoff <= 0:
        flags.append("REJECT: non-positive take-off velocity; onset or BW is wrong")
        return _degenerate(bw, mass, bw_sd, fs, filter_cutoff_hz, flags, t, f)

    jump_height = v_takeoff**2 / (2 * G)
    jump_height_ft = G * flight_time**2 / 8.0

    # --- phase boundaries from the velocity trace ----------------------
    v_min_rel = int(np.argmin(vel))
    after = np.flatnonzero(vel[v_min_rel:] >= 0.0)
    zero_rel = v_min_rel + (int(after[0]) if after.size else len(vel) - 1)

    unweighting_s = v_min_rel / fs
    ecc_s = (zero_rel - v_min_rel) / fs
    con_s = (len(vel) - 1 - zero_rel) / fs
    contraction_s = (takeoff_idx - onset_idx) / fs

    # --- kinetics ------------------------------------------------------
    peak_force = float(np.max(f[seg]))
    power = f[seg] * vel
    peak_power = float(np.max(power[zero_rel:])) if zero_rel < len(power) else float(np.max(power))
    net_impulse = float(np.trapezoid(f[seg] - bw, dx=1.0 / fs))
    cm_depth = float(np.min(disp))

    # --- physiological plausibility ------------------------------------
    # These bounds are sports-science judgement, not arbitrary numbers: an adult
    # athlete CMJ below 5 cm or above 120 cm is a measurement fault, not a jump.
    if not (0.05 <= jump_height <= 1.20):
        flags.append(f"REJECT: jump_height {jump_height:.3f} m outside 0.05-1.20 m")
    if abs(jump_height - jump_height_ft) > 0.05:
        flags.append(
            "WARN: impulse-momentum and flight-time heights differ by "
            f"{abs(jump_height - jump_height_ft) * 100:.1f} cm; check landing posture"
        )
    if peak_force / bw > 5.0:
        flags.append(f"WARN: peak force {peak_force / bw:.1f} x BW is implausibly high")
    if cm_depth < -0.60:
        flags.append(f"WARN: countermovement depth {cm_depth:.2f} m is unusually deep")

    return CMJResult(
        body_weight_n=float(bw),
        body_mass_kg=float(mass),
        bw_sd_n=float(bw_sd),
        jump_height_m=float(jump_height),
        jump_height_flight_time_m=float(jump_height_ft),
        takeoff_velocity_ms=v_takeoff,
        rsi_mod=float(jump_height / contraction_s) if contraction_s > 0 else float("nan"),
        peak_force_n=peak_force,
        peak_force_bw=float(peak_force / bw),
        peak_power_w=peak_power,
        peak_power_w_kg=float(peak_power / mass),
        net_impulse_ns=net_impulse,
        unweighting_duration_s=float(unweighting_s),
        ecc_duration_s=float(ecc_s),
        con_duration_s=float(con_s),
        contraction_time_s=float(contraction_s),
        flight_time_s=float(flight_time),
        countermovement_depth_m=cm_depth,
        sample_rate_hz=float(fs),
        filter_cutoff_hz=float(filter_cutoff_hz),
        indices={
            "quiet_start": int(qs0),
            "quiet_end": int(qs1),
            "onset": int(onset_idx),
            "min_velocity": int(onset_idx + v_min_rel),
            "zero_velocity": int(onset_idx + zero_rel),
            "takeoff": int(takeoff_idx),
            "landing": int(landing_idx),
        },
        quality_flags=flags,
        _time=t,
        _force=f,
        _velocity=vel,
    )


def _degenerate(bw, mass, bw_sd, fs, cutoff, flags, t, f) -> CMJResult:
    """Result object for a trace that failed detection -- carries the flags and
    the trace so the failure can still be plotted and inspected."""
    nan = float("nan")
    return CMJResult(
        body_weight_n=float(bw),
        body_mass_kg=float(mass),
        bw_sd_n=float(bw_sd),
        jump_height_m=nan,
        jump_height_flight_time_m=nan,
        takeoff_velocity_ms=nan,
        rsi_mod=nan,
        peak_force_n=nan,
        peak_force_bw=nan,
        peak_power_w=nan,
        peak_power_w_kg=nan,
        net_impulse_ns=nan,
        unweighting_duration_s=nan,
        ecc_duration_s=nan,
        con_duration_s=nan,
        contraction_time_s=nan,
        flight_time_s=nan,
        countermovement_depth_m=nan,
        sample_rate_hz=float(fs),
        filter_cutoff_hz=float(cutoff),
        quality_flags=flags,
        _time=t,
        _force=f,
    )
