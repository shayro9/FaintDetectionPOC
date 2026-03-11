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
    anchors = np.nanmean(sig[:n_seconds * fs].reshape(n_seconds, fs), axis=1)

    # Forward-fill NaN anchors (e.g. entire dropout windows)
    nans = np.isnan(anchors)
    if nans.any():
        idx = np.where(~nans, np.arange(n_seconds), 0)
        np.maximum.accumulate(idx, out=idx)
        anchors = anchors[idx]

    # Anchor timestamps at the centre of each 1-second block
    t_anchors = t[fs // 2 :: fs][:n_seconds]

    # Append the final sample so we don't extrapolate at the tail
    t_anchors = np.append(t_anchors, t[-1])
    anchors   = np.append(anchors, anchors[-1])

    return np.interp(t, t_anchors, anchors)


def simulate_session(activity, duration_s=300, fs=FS, user_id=1,
                     seed=None, artifacts=True, logger=None):
    rng     = np.random.default_rng(seed)
    profile = ACTIVITY_PROFILES[activity]
    t       = np.linspace(0, duration_s, duration_s * fs)
    n       = len(t)

    # ── Per-user stable biases ────────────────────────────────────────────────
    user_rng = np.random.default_rng(user_id * 9973)
    up       = get_user_profile(user_id, user_rng)

    # ── Generate raw signals ──────────────────────────────────────────────────
    ax, ay, az = sim_accel(t, activity, profile, rng, cadence_bias=up.cadence_bias)
    gx, gy, gz = sim_gyro( t, activity, profile, rng, cadence_bias=up.cadence_bias)

    hr   = sim_hr(  t, profile, rng, fitness=up.fitness) + up.hr_bias
    spo2 = sim_spo2(t, profile, rng)                     + up.spo2_bias
    temp = sim_temp(t, profile, rng, activity=activity)  + up.temp_bias

    # ── Cardiac warm-up ramp ──────────────────────────────────────────────────
    if activity != "sleeping":
        warmup_s = min(60, duration_s // 4)
        warmup_n = warmup_s * fs
        ramp = 1.0 - np.exp(-np.arange(warmup_n) / (15.0 * fs))
        hr[:warmup_n] -= 10.0 * (1.0 - ramp)

    # ── Artifact injection ────────────────────────────────────────────────────
    if artifacts:
        ax, ay, az = (add_artifacts(s, rng=rng) for s in (ax, ay, az))
        hr         = add_artifacts(hr, dropout=0.002, rng=rng)
        hr, spo2   = add_motion_artifacts(hr, spo2, ax, ay, az, rng, fs=fs)

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