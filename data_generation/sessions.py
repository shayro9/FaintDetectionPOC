import pandas as pd

from config import *
from signals import *

def simulate_session(activity, duration_s=300, fs=FS, user_id=1,
                     seed=None, artifacts=True, logger=None):
    rng     = np.random.default_rng(seed)
    profile = ACTIVITY_PROFILES[activity]
    t       = np.linspace(0, duration_s, duration_s * fs)
    n       = len(t)

    hr_off   = rng.normal(0, 8.0)
    temp_off = rng.normal(0, 0.3)

    ax, ay, az = sim_accel(t, activity, profile, rng)
    gx, gy, gz = sim_gyro(t, activity, profile, rng)
    hr   = sim_hr(t, profile, rng) + hr_off
    spo2 = sim_spo2(t, profile, rng)
    temp = sim_temp(t, profile, rng) + temp_off

    if artifacts:
        ax, ay, az = (add_artifacts(s, rng=rng) for s in (ax, ay, az))
        hr = add_artifacts(hr, dropout=0.002, rng=rng)

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
            f"seed={seed} HR_base={profile['hr']['mean']+hr_off:.1f}"
        )
    return df