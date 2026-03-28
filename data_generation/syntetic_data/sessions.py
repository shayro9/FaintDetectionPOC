import warnings

import numpy as np
import pandas as pd

from config import ACTIVITY_PROFILES, FS
from signals import (
    sim_accel, sim_gyro, sim_hr, sim_spo2, sim_temp,
    add_artifacts, add_motion_artifacts,
)
from user_profile import get_user_profile


def _interp_1hz(sig, t, fs):
    """
    Average into 1 Hz windows then linearly interpolate back to the full grid.

    - Uses window-averaged anchors centred in each second to suppress noise.
    - Forward-fills any NaN anchors (caused by dropout artifacts) before
      interpolating so they don't create flat artifact segments.
    - Appends the last sample as an extra anchor so interpolation reaches
      the very end of the session without extrapolation.
    """
    n         = len(t)
    n_seconds = n // fs

    # Window-average: shape (n_seconds, fs) -> (n_seconds,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        anchors = np.nanmean(sig[:n_seconds * fs].reshape(n_seconds, fs), axis=1)

    # Forward-fill NaN anchors (e.g. entire dropout windows)
    nans = np.isnan(anchors)
    if nans.any():
        idx = np.where(~nans, np.arange(n_seconds), 0)
        np.maximum.accumulate(idx, out=idx)
        anchors = anchors[idx]

    # If the session starts with NaN anchors, back-fill from first valid anchor.
    if np.isnan(anchors[0]):
        valid = np.where(~np.isnan(anchors))[0]
        if len(valid) == 0:
            return np.full_like(t, np.nan, dtype=float)
        anchors[:valid[0]] = anchors[valid[0]]

    # Anchor timestamps at the centre of each 1-second block
    t_anchors = t[fs // 2 :: fs][:n_seconds]

    # Append the final sample so we don't extrapolate at the tail
    t_anchors = np.append(t_anchors, t[-1])
    anchors   = np.append(anchors, anchors[-1])

    return np.interp(t, t_anchors, anchors)


def _intensity_envelope(t, fs, rng):
    """
    Build a smooth exercise-intensity envelope with occasional short pauses.

    This imitates real free-living sessions where pace naturally drifts and
    people briefly stop at crossings/turns instead of moving at constant speed.
    """
    n = len(t)
    base = 1.0 + 0.12 * np.sin(2 * np.pi * 0.004 * t + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, 0.35, n)

    # Strong low-pass to create minute-scale effort drifts
    from scipy.signal import butter, filtfilt
    b, a = butter(3, 0.03 / (fs / 2.0), btype="low")
    drift = filtfilt(b, a, noise)

    env = np.clip(base + 0.10 * drift, 0.70, 1.30)

    # 0–2 short pauses where movement drops but doesn't become perfectly still
    n_pauses = int(rng.integers(0, 3))
    for _ in range(n_pauses):
        dur_s = int(rng.integers(3, 12))
        start = int(rng.integers(0, max(1, len(t) - dur_s * fs)))
        end = min(len(t), start + dur_s * fs)
        env[start:end] *= rng.uniform(0.35, 0.60)

    return env


def _add_orientation_shift(ax, ay, az, rng, fs):
    """Add piecewise-constant orientation offsets (wrist rotation / strap shift)."""
    n = len(ax)
    ox = np.zeros(n)
    oy = np.zeros(n)
    oz = np.zeros(n)

    idx = 0
    while idx < n:
        seg_len = int(rng.integers(35 * fs, 120 * fs))
        end = min(n, idx + seg_len)
        ox[idx:end] = rng.normal(0, 0.15)
        oy[idx:end] = rng.normal(0, 0.15)
        oz[idx:end] = rng.normal(0, 0.08)
        idx = end

    return ax + ox, ay + oy, az + oz




def _activity_context_blocks(t, fs, rng):
    """Generate random activity context blocks (sitting/walking/running)."""
    n = len(t)
    hr_offset = np.zeros(n)
    accel_scale = np.ones(n)

    phase_cfg = {
        "sitting": {"hr": 60.0, "accel_std": 0.05},
        "walking": {"hr": 90.0, "accel_std": 0.4},
        "running": {"hr": 140.0, "accel_std": 1.2},
    }

    idx = 0
    while idx < n:
        seg_len = int(rng.integers(20 * fs, 90 * fs))
        end = min(n, idx + seg_len)
        phase = rng.choice(["sitting", "walking", "running"], p=[0.3, 0.45, 0.25])
        cfg = phase_cfg[phase]
        hr_offset[idx:end] = cfg["hr"]
        accel_scale[idx:end] = np.interp(cfg["accel_std"], [0.05, 1.2], [0.35, 1.35])
        idx = end

    return hr_offset, accel_scale

def simulate_session(activity, duration_s=300, fs=FS, user_id=1,
                     seed=None, artifacts=True, logger=None):
    rng     = np.random.default_rng(seed)
    profile = ACTIVITY_PROFILES[activity]
    t       = np.linspace(0, duration_s, duration_s * fs)

    # ── Per-user stable biases ────────────────────────────────────────────────
    user_rng = np.random.default_rng(user_id * 9973)
    up       = get_user_profile(user_id, user_rng)

    # ── Generate raw signals ──────────────────────────────────────────────────
    ax, ay, az = sim_accel(t, activity, profile, rng, cadence_bias=up.cadence_bias)
    gx, gy, gz = sim_gyro(t, activity, profile, rng, cadence_bias=up.cadence_bias)

    hr   = sim_hr(  t, profile, rng, fitness=up.fitness) + up.hr_bias
    spo2 = sim_spo2(t, profile, rng)                     + up.spo2_bias
    temp = sim_temp(t, profile, rng, activity=activity)  + up.temp_bias

    if activity != "sleeping":
        ctx_hr_target, ctx_accel_scale = _activity_context_blocks(t, fs, rng)
        baseline_hr = float(np.nanmedian(hr))
        hr += 0.12 * (ctx_hr_target - baseline_hr)
        ax *= ctx_accel_scale
        ay *= ctx_accel_scale
        az = 9.81 + (az - 9.81) * ctx_accel_scale

    # ── In-session intensity drift (free-living realism) ─────────────────────
    if activity != "sleeping":
        env = _intensity_envelope(t, fs, rng)

        ax *= env
        ay *= env
        az = 9.81 + (az - 9.81) * env

        gx *= env
        gy *= env
        gz *= env

        hr += (env - 1.0) * 18.0
        spo2 -= np.maximum(0.0, env - 1.0) * 0.5

    # ── Cardiac warm-up ramp ──────────────────────────────────────────────────
    if activity != "sleeping":
        warmup_s = min(60, duration_s // 4)
        warmup_n = warmup_s * fs
        ramp = 1.0 - np.exp(-np.arange(warmup_n) / (15.0 * fs))
        hr[:warmup_n] -= 10.0 * (1.0 - ramp)

    # ── Artifact injection ────────────────────────────────────────────────────
    if artifacts:
        ax, ay, az = (add_artifacts(s, rng=rng, fs=fs) for s in (ax, ay, az))
        hr         = add_artifacts(hr, dropout=0.002, rng=rng, fs=fs)
        hr, spo2   = add_motion_artifacts(hr, spo2, ax, ay, az, rng, fs=fs)

    # Wrist orientation can shift over time regardless of artifact mode
    ax, ay, az = _add_orientation_shift(ax, ay, az, rng, fs)

    spo2 = np.clip(spo2, 90.0, 100.0)

    # ── Resample to 1 Hz then interpolate back to full grid ───────────────────
    # Real smartwatches report HR/SpO₂ once per second.  We average each 1-second
    # window (1 Hz), then linearly interpolate back to the full sample grid.
    # This eliminates the staircase from np.repeat while keeping smooth
    # continuity at event-phase boundaries.
    hr_full   = _interp_1hz(hr,   t, fs)
    spo2_full = np.clip(_interp_1hz(spo2, t, fs), 90.0, 100.0)
    temp_full = temp   # already slow-moving via thermal-lag IIR

    df = pd.DataFrame({
        "timestamp":   t,
        "user_id":     user_id,
        "activity":    activity,
        "label":       "normal",
        "event_phase": "none",
        "accel_x": ax,  "accel_y": ay,  "accel_z": az,
        "gyro_x":  gx,  "gyro_y":  gy,  "gyro_z":  gz,
        "heart_rate": hr_full,
        "spo2":       spo2_full,
        "skin_temp":  temp_full,
    })

    if logger:
        logger.debug(
            f"  Session | user={user_id} activity={activity} duration={duration_s}s "
            f"seed={seed} HR_base={profile['hr']['mean'] + up.hr_bias:.1f} "
            f"fitness={up.fitness:.2f} cadence_bias={up.cadence_bias:+.2f} "
            f"spo2_bias={up.spo2_bias:+.2f}"
        )
    return df
