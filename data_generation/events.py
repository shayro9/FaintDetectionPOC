import numpy as np

from config import FS


def plant_faint_event(df, event_start_s, fs=FS, rng=None, logger=None):
    """
    Injects a multiphase fainting event into a session DataFrame.

    Phases
    ------
    pre_syncope : 20–50 s  — compensatory tachycardia (HR capped at 185 bpm)
                             then sudden vagal crash, SpO2 drops, tremor builds
    syncope     : 5–20 s   — fall-impact spike on ALL axes, then near-zero
                             movement; HR/SpO2 at floor values; temp ramps down
    recovery    : 40–90 s  — sigmoid HR/SpO2 targeting session baseline;
                             gyro noise scaled to lying-still level (visible
                             but clearly lower than active motion)
    """
    if rng is None:
        rng = np.random.default_rng()

    df = df.copy()
    total_samples = len(df)

    pre_dur  = int(rng.integers(20, 51))
    sync_dur = int(rng.integers(5,  21))
    rec_dur  = int(rng.integers(40, 91))

    i0 = int(event_start_s * fs)
    i1 = i0 + pre_dur  * fs
    i2 = i1 + sync_dur * fs
    i3 = min(i2 + rec_dur * fs, total_samples)

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

    # ── Read session baseline (read BEFORE any modifications) ────────────────
    baseline_hr   = float(df.iloc[min(i3, total_samples - 1)]["heart_rate"])
    baseline_spo2 = float(df.iloc[min(i3, total_samples - 1)]["spo2"])
    baseline_temp = float(df.iloc[min(i3, total_samples - 1)]["skin_temp"])

    # ── PRE-SYNCOPE ───────────────────────────────────────────────────────────
    n_pre = i1 - i0
    t_pre = np.linspace(0, 1, n_pre)

    # HR: compensatory tachycardia (+15 bpm, hard-capped at 185 bpm)
    #     then sudden vagal withdrawal crash.
    # The cap prevents the running+high-fitness combination from producing
    # impossible values (>190 bpm) before the person even collapses.
    current_hr  = df.iloc[i0:i1, df.columns.get_loc("heart_rate")].values.copy()
    split       = int(0.60 * n_pre)
    hr_mod      = np.empty(n_pre)
    hr_mod[:split]  = 15.0 * np.linspace(0.0, 1.0, split)
    hr_mod[split:]  = 15.0 - 65.0 * np.linspace(0.0, 1.0, n_pre - split)
    new_hr      = np.clip(current_hr + hr_mod, None, 185.0)
    df.iloc[i0:i1, df.columns.get_loc("heart_rate")] = new_hr

    # SpO2: gradual drop (−4%)
    df.iloc[i0:i1, df.columns.get_loc("spo2")] -= 4.0 * t_pre

    # Temp: gradual peripheral vasoconstriction ramp (not a step)
    df.iloc[i0:i1, df.columns.get_loc("skin_temp")] -= 0.8 * t_pre

    # Accel: increasing tremor
    tremor = np.linspace(0, 0.6, n_pre)
    df.iloc[i0:i1, df.columns.get_loc("accel_x")] += rng.normal(0, 1.0, n_pre) * tremor
    df.iloc[i0:i1, df.columns.get_loc("accel_y")] += rng.normal(0, 1.0, n_pre) * tremor * 0.7

    # Gyro: erratic rotations building up — additive on existing signal
    df.iloc[i0:i1, df.columns.get_loc("gyro_x")] += rng.normal(0, 0.5, n_pre) * tremor
    df.iloc[i0:i1, df.columns.get_loc("gyro_y")] += rng.normal(0, 0.5, n_pre) * tremor

    df.iloc[i0:i1, df.columns.get_loc("label")]       = "pre_syncope"
    df.iloc[i0:i1, df.columns.get_loc("event_phase")] = "pre_syncope"

    # ── SYNCOPE (FALL + UNCONSCIOUS) ──────────────────────────────────────────
    n_sync = i2 - i1

    fall_len     = min(20, n_sync)
    fall_profile = np.array(
        [0,1,4,10,18,12,6,3,1.5,0.8,0.4,0.2,0.1,0,0,0,0,0,0,0], dtype=float
    )
    scale = rng.uniform(0.8, 1.4)

    # Accel fall spike then dead-still
    fall_x = np.zeros(n_sync)
    fall_x[:fall_len] = fall_profile[:fall_len] * scale
    df.iloc[i1:i2, df.columns.get_loc("accel_x")] = fall_x + rng.normal(0, 0.04, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("accel_y")] = rng.normal(0, 0.04, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("accel_z")] = 9.81 + rng.normal(0, 0.03, n_sync)

    # Gyro: fall spike on ALL three axes then near-zero lying-still noise.
    # Previously only gyro_x got the spike; gyro_y/z went from running amplitude
    # (~1 rad/s) to std=0.01 instantaneously — looked like a hard cut.
    # Now all axes get a proportional spike then settle to a visible but clearly
    # reduced noise floor (~0.05 rad/s) consistent with a body lying still.
    lying_std = 0.05   # visible on plot but far below active motion amplitude
    for col, spike_scale in [("gyro_x", 0.40), ("gyro_y", 0.25), ("gyro_z", 0.15)]:
        spike = np.zeros(n_sync)
        spike[:fall_len] = fall_profile[:fall_len] * spike_scale * scale
        df.iloc[i1:i2, df.columns.get_loc(col)] = spike + rng.normal(0, lying_std, n_sync)

    df.iloc[i1:i2, df.columns.get_loc("heart_rate")] = 40 + rng.normal(0, 2, n_sync)
    df.iloc[i1:i2, df.columns.get_loc("spo2")]        = 89 + rng.normal(0, 0.6, n_sync)

    # Temp: ramp down across the syncope window instead of a sudden step.
    # The pre_syncope ended at baseline_temp - 0.8; continue ramping to -1.2
    # total so the decline is continuous.
    temp_at_syncope_start = baseline_temp - 0.8
    temp_at_syncope_end   = baseline_temp - 1.2
    df.iloc[i1:i2, df.columns.get_loc("skin_temp")] = np.linspace(
        temp_at_syncope_start, temp_at_syncope_end, n_sync
    )

    df.iloc[i1:i2, df.columns.get_loc("label")]       = "syncope"
    df.iloc[i1:i2, df.columns.get_loc("event_phase")] = "syncope"

    # ── RECOVERY ──────────────────────────────────────────────────────────────
    n_rec = i3 - i2
    t_rec = np.linspace(0, 1, n_rec)

    # HR: sigmoid from 40 bpm floor to session baseline
    hr_recovery = 40.0 + (baseline_hr - 40.0) * (
        1 / (1 + np.exp(-8 * (t_rec - 0.4)))
    )
    df.iloc[i2:i3, df.columns.get_loc("heart_rate")] = (
        hr_recovery + rng.normal(0, 3, n_rec)
    )

    # SpO2: linear recovery to baseline
    df.iloc[i2:i3, df.columns.get_loc("spo2")] = np.clip(
        89.0 + (baseline_spo2 - 89.0) * t_rec + rng.normal(0, 0.4, n_rec),
        None, 99.0
    )

    # Temp: linear recovery from syncope floor to baseline
    df.iloc[i2:i3, df.columns.get_loc("skin_temp")] = (
        temp_at_syncope_end + (baseline_temp - temp_at_syncope_end) * t_rec
    )

    # Accel: lying still with micro-movements returning in second half
    micro = np.zeros(n_rec)
    micro[n_rec//2:] = rng.normal(0, 0.05, n_rec - n_rec//2) * t_rec[n_rec//2:]
    df.iloc[i2:i3, df.columns.get_loc("accel_x")] = micro
    df.iloc[i2:i3, df.columns.get_loc("accel_y")] = rng.normal(0, 0.02, n_rec)
    df.iloc[i2:i3, df.columns.get_loc("accel_z")] = 9.81 + rng.normal(0, 0.02, n_rec)

    # Gyro: lying-still noise throughout, micro-movements building in second half.
    # Use the same lying_std as syncope so there's no amplitude jump at the
    # syncope→recovery boundary.  Movements gradually increase toward recovery end.
    gyro_scale = np.where(
        np.arange(n_rec) >= n_rec // 2,
        lying_std + 0.05 * t_rec,   # slowly growing micro-movements
        lying_std                    # flat lying-still floor
    )
    for col in ("gyro_x", "gyro_y", "gyro_z"):
        df.iloc[i2:i3, df.columns.get_loc(col)] = rng.normal(0, 1, n_rec) * gyro_scale

    df.iloc[i2:i3, df.columns.get_loc("label")]       = "recovery"
    df.iloc[i2:i3, df.columns.get_loc("event_phase")] = "recovery"

    if logger:
        logger.debug(
            f"  Event planted | start={event_start_s}s "
            f"pre={pre_dur}s sync={sync_dur}s rec={rec_dur}s "
            f"total={pre_dur + sync_dur + rec_dur}s "
            f"HR_target={baseline_hr:.1f} SpO2_target={baseline_spo2:.1f} "
            f"temp_target={baseline_temp:.2f}"
        )

    return df, event_meta