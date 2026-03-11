import numpy as np
import pandas as pd

from config import ACTIVITY_PROFILES, FS
from signals import (
    sim_accel, sim_gyro, sim_hr, sim_spo2, sim_temp,
    add_artifacts, add_motion_artifacts,
)
from user_profile import get_user_profile


def simulate_session(activity, duration_s=300, fs=FS, user_id=1,
                     seed=None, artifacts=True, logger=None):
    rng     = np.random.default_rng(seed)
    profile = ACTIVITY_PROFILES[activity]
    t       = np.linspace(0, duration_s, duration_s * fs)
    n       = len(t)

    # ── Per-user stable biases ────────────────────────────────────────────────
    # Use a deterministic seed derived from user_id so each user always gets the
    # same profile regardless of generation order.
    user_rng = np.random.default_rng(user_id * 9973)
    up       = get_user_profile(user_id, user_rng)

    # ── Generate raw signals ──────────────────────────────────────────────────
    ax, ay, az = sim_accel(t, activity, profile, rng, cadence_bias=up.cadence_bias)
    gx, gy, gz = sim_gyro( t, activity, profile, rng, cadence_bias=up.cadence_bias)

    hr   = sim_hr(  t, profile, rng, fitness=up.fitness) + up.hr_bias
    spo2 = sim_spo2(t, profile, rng)                     + up.spo2_bias
    temp = sim_temp(t, profile, rng, activity=activity)  + up.temp_bias

    # ── Cardiac warm-up ramp ──────────────────────────────────────────────────
    # At the start of a walking/running session the heart hasn't reached its
    # steady-state rate yet; it ramps up over ~60 s.
    if activity != "sleeping":
        warmup_s = min(60, duration_s // 4)
        warmup_n = warmup_s * fs
        # Exponential ramp: starts ~10 bpm below target, converges to 0 offset
        ramp = 1.0 - np.exp(-np.arange(warmup_n) / (15.0 * fs))
        hr[:warmup_n] -= 10.0 * (1.0 - ramp)

    # ── Artifact injection ────────────────────────────────────────────────────
    if artifacts:
        ax, ay, az = (add_artifacts(s, rng=rng) for s in (ax, ay, az))
        hr         = add_artifacts(hr, dropout=0.002, rng=rng)
        # Motion-correlated PPG artifacts — must run AFTER accel artifacts so
        # the envelope reflects realistic sensor values
        hr, spo2   = add_motion_artifacts(hr, spo2, ax, ay, az, rng, fs=fs)

    spo2 = np.clip(spo2, 90.0, 100.0)

    # ── Downsample physiological signals to 1 Hz ─────────────────────────────
    # Real smartwatches report HR/SpO2/temp at ~1 Hz; the finer structure is
    # already captured in the artifact and HRV variation above.
    hr_full   = np.repeat(hr[::fs],   fs)[:n]
    spo2_full = np.repeat(spo2[::fs], fs)[:n]
    temp_full = np.repeat(temp[::fs], fs)[:n]

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