import numpy as np

from config import FS


def plant_faint_event(df, event_start_s, fs=FS, rng=None, logger=None):
    """
    Injects a multiphase fainting event into a session DataFrame.

    Phases
    ------
    pre_syncope : 20–50 s  — HR compensatory tachycardia then sudden vagal crash,
                             SpO₂ drops, tremor increases
    syncope     : 5–20 s   — fall-impact spike, near-zero movement, very low HR/SpO₂
    recovery    : 40–90 s  — sigmoid HR/SpO₂ restoration, minimal movement
    """
    if rng is None:
        rng = np.random.default_rng()

    df = df.copy()
    total_samples = len(df)

    # Randomise phase durations
    pre_dur  = int(rng.integers(20, 51))
    sync_dur = int(rng.integers(5,  21))
    rec_dur  = int(rng.integers(40, 91))

    i0 = int(event_start_s * fs)
    i1 = i0  + pre_dur  * fs
    i2 = i1  + sync_dur * fs
    i3 = min(i2 + rec_dur * fs, total_samples)

    # i3 is already clipped by min(), but i1/i2 can still overrun if the session
    # is short relative to pre_dur + sync_dur — skip in that case.
    if i2 > total_samples or i0 < 0:
        if logger:
            logger.warning(f"  Event at {event_start_s}s overruns session — skipped")
        return df, None

    event_meta = {
        "event_start_s":    event_start_s,
        "pre_start_s":      i0 / fs,
        "syncope_start_s":  i1 / fs,
        "recovery_start_s": i2 / fs,
        "recovery_end_s":   i3 / fs,
        "pre_dur_s":        pre_dur,
        "sync_dur_s":       sync_dur,
        "rec_dur_s":        rec_dur,
    }

    # ── PRE-SYNCOPE ────────────────────────────────────────────────────────────
    n_pre = i1 - i0
    t_pre = np.linspace(0, 1, n_pre)

    # HR: realistic vasovagal shape
    #   Phase 1 (first 60%): gentle compensatory tachycardia (+15 bpm)
    #   Phase 2 (last 40%):  sudden vagal withdrawal crash (−50 bpm from peak)
    # The original quadratic was too symmetric and too steep at onset.
    split       = int(0.60 * n_pre)
    hr_mod      = np.empty(n_pre)
    hr_mod[:split]  = 15.0 * np.linspace(0.0, 1.0, split)
    hr_mod[split:]  = 15.0 - 65.0 * np.linspace(0.0, 1.0, n_pre - split)

    df.iloc[i0:i1, df.columns.get_loc("heart_rate")] += hr_mod

    # SpO₂: gradual drop (−4%)
    df.iloc[i0:i1, df.columns.get_loc("spo2")] -= 4.0 * t_pre

    # Temp: peripheral vasoconstriction / cold sweat
    df.iloc[i0:i1, df.columns.get_loc("skin_temp")] -= 0.8 * t_pre

    # Accel: increasing tremor / gait instability
    tremor_scale = np.linspace(0, 0.6, n_pre)
    df.iloc[i0:i1, df.columns.get_loc("accel_x")] += rng.normal(0, 1, n_pre) * tremor_scale
    df.iloc[i0:i1, df.columns.get_loc("accel_y")] += rng.normal(0, 1, n_pre) * tremor_scale * 0.7

    # Gyro: erratic rotations
    df.iloc[i0:i1, df.columns.get_loc("gyro_x")] += rng.normal(0, 0.5, n_pre) * tremor_scale
    df.iloc[i0:i1, df.columns.get_loc("gyro_y")] += rng.normal(0, 0.5, n_pre) * tremor_scale

    df.iloc[i0:i1, df.columns.get_loc("label")]       = "pre_syncope"
    df.iloc[i0:i1, df.columns.get_loc("event_phase")] = "pre_syncope"

    # ── SYNCOPE (FALL + UNCONSCIOUS) ───────────────────────────────────────────
    n_sync = i2 - i1

    fall_len     = min(20, n_sync)
    fall_spike   = np.zeros(n_sync)
    fall_profile = np.array([0,1,4,10,18,12,6,3,1.5,0.8,0.4,0.2,0.1,0,0,0,0,0,0,0], dtype=float)
    fall_spike[:fall_len] = fall_profile[:fall_len] * rng.uniform(0.8, 1.4)

    df.iloc[i1:i2, df.columns.get_loc("accel_x")] = fall_spike + rng.normal(0, 0.04, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("accel_y")] = rng.normal(0, 0.04, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("accel_z")] = 9.81 + rng.normal(0, 0.03, n_sync)

    gyro_spike = np.zeros(n_sync)
    gyro_spike[:fall_len] = fall_profile[:fall_len] * 0.4
    df.iloc[i1:i2, df.columns.get_loc("gyro_x")] = gyro_spike + rng.normal(0, 0.01, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("gyro_y")] = rng.normal(0, 0.01, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("gyro_z")] = rng.normal(0, 0.01, n_sync)

    df.iloc[i1:i2, df.columns.get_loc("heart_rate")] = 40 + rng.normal(0, 2, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("spo2")]        = 89 + rng.normal(0, 0.6, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("skin_temp")]  -= 1.2

    df.iloc[i1:i2, df.columns.get_loc("label")]       = "syncope"
    df.iloc[i1:i2, df.columns.get_loc("event_phase")] = "syncope"

    # ── RECOVERY ───────────────────────────────────────────────────────────────
    n_rec = i3 - i2
    t_rec = np.linspace(0, 1, n_rec)

    hr_recovery = 40 + 55 * (1 / (1 + np.exp(-8 * (t_rec - 0.4))))
    df.iloc[i2:i3, df.columns.get_loc("heart_rate")] = hr_recovery + rng.normal(0, 3, n_rec)

    df.iloc[i2:i3, df.columns.get_loc("spo2")] = (
        np.clip(89 + 9*t_rec + rng.normal(0, 0.4, n_rec), None, 99.0)
    )
    df.iloc[i2:i3, df.columns.get_loc("skin_temp")] += 1.5 * t_rec

    micro = np.zeros(n_rec)
    micro[n_rec//2:] = rng.normal(0, 0.05, n_rec - n_rec//2) * t_rec[n_rec//2:]
    df.iloc[i2:i3, df.columns.get_loc("accel_x")] = micro
    df.iloc[i2:i3, df.columns.get_loc("accel_y")] = rng.normal(0, 0.02, n_rec)
    df.iloc[i2:i3, df.columns.get_loc("accel_z")] = 9.81 + rng.normal(0, 0.02, n_rec)

    df.iloc[i2:i3, df.columns.get_loc("label")]       = "recovery"
    df.iloc[i2:i3, df.columns.get_loc("event_phase")] = "recovery"

    if logger:
        logger.debug(
            f"  Event planted | start={event_start_s}s "
            f"pre={pre_dur}s sync={sync_dur}s rec={rec_dur}s "
            f"total_event={pre_dur + sync_dur + rec_dur}s"
        )

    return df, event_meta