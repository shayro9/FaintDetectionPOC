import numpy as np

from config import FS
from signals import smooth_noise


def approach(target, start, t, tau=0.4):
    return target + (start - target) * np.exp(-t / tau)


def blend_transition(arr_a, arr_b, blend_len):
    """Softly blend boundary between two contiguous signal segments."""
    if blend_len <= 0:
        return arr_a, arr_b

    blend_len = min(blend_len, len(arr_a), len(arr_b))
    alpha = np.linspace(0.0, 1.0, blend_len)
    blended = arr_a[-blend_len:] * (1.0 - alpha) + arr_b[:blend_len] * alpha
    arr_a[-blend_len:] = blended
    arr_b[:blend_len] = blended
    return arr_a, arr_b


def plant_faint_event(df, event_start_s, fs=FS, rng=None, logger=None):
    """Inject a multiphase fainting event into a session DataFrame."""
    if rng is None:
        rng = np.random.default_rng()

    df = df.copy()
    total_samples = len(df)
    cols = {c: df.columns.get_loc(c) for c in df.columns}

    pre_dur = int(rng.integers(20, 51))
    sync_dur = int(rng.integers(5, 21))
    rec_dur = int(rng.integers(40, 91))

    i0 = int(event_start_s * fs)
    i1 = i0 + pre_dur * fs
    i2 = i1 + sync_dur * fs
    i3 = min(i2 + rec_dur * fs, total_samples)

    if i2 > total_samples or i0 < 0:
        if logger:
            logger.warning(f"  Event at {event_start_s}s overruns session — skipped")
        return df, None

    variant = rng.choice(["vasovagal_syncope", "orthostatic_faint", "exercise_collapse"])
    variant_cfg = {
        "vasovagal_syncope": {"hr_drop": 45.0, "movement": 0.8, "spo2_drop": 3.5},
        "orthostatic_faint": {"hr_drop": 35.0, "movement": 0.6, "spo2_drop": 2.5},
        "exercise_collapse": {"hr_drop": 55.0, "movement": 1.1, "spo2_drop": 4.5},
    }[variant]

    event_meta = {
        "event_start_s": event_start_s,
        "pre_start_s": i0 / fs,
        "syncope_start_s": i1 / fs,
        "recovery_start_s": i2 / fs,
        "recovery_end_s": i3 / fs,
        "pre_dur_s": pre_dur,
        "sync_dur_s": sync_dur,
        "rec_dur_s": rec_dur,
        "variant": variant,
    }

    baseline_hr = float(df.iloc[min(i3, total_samples - 1)]["heart_rate"])
    baseline_spo2 = float(df.iloc[min(i3, total_samples - 1)]["spo2"])
    baseline_temp = float(df.iloc[min(i3, total_samples - 1)]["skin_temp"])

    # PRE-SYNCOPE
    n_pre = i1 - i0
    t_pre = np.linspace(0, 1, n_pre)
    current_hr = df.iloc[i0:i1, cols["heart_rate"]].values.copy()

    split = int(0.60 * n_pre)
    hr_mod = np.empty(n_pre)
    hr_mod[:split] = 15.0 * np.linspace(0.0, 1.0, split)
    hr_mod[split:] = 15.0 - variant_cfg["hr_drop"] * np.linspace(0.0, 1.0, n_pre - split)
    new_hr = np.clip(current_hr + hr_mod, None, 185.0)

    pre_ax = df.iloc[i0:i1, cols["accel_x"]].values
    pre_ay = df.iloc[i0:i1, cols["accel_y"]].values
    pre_az = df.iloc[i0:i1, cols["accel_z"]].values
    movement_intensity = np.clip(np.sqrt(pre_ax**2 + pre_ay**2 + (pre_az - 9.81) ** 2), 0.0, 2.0)

    new_hr = new_hr + movement_intensity * 5.0
    pre_spo2 = df.iloc[i0:i1, cols["spo2"]].values - variant_cfg["spo2_drop"] * t_pre
    pre_spo2 = pre_spo2 - 0.03 * (baseline_hr - new_hr)

    df.iloc[i0:i1, cols["heart_rate"]] = new_hr
    df.iloc[i0:i1, cols["spo2"]] = pre_spo2
    df.iloc[i0:i1, cols["skin_temp"]] -= 0.8 * t_pre

    tremor = np.linspace(0, 0.6 * variant_cfg["movement"], n_pre)
    df.iloc[i0:i1, cols["accel_x"]] += smooth_noise(rng, n_pre, 1.0, kernel=8) * tremor
    df.iloc[i0:i1, cols["accel_y"]] += smooth_noise(rng, n_pre, 0.8, kernel=8) * tremor
    df.iloc[i0:i1, cols["gyro_x"]] += smooth_noise(rng, n_pre, 0.5, kernel=8) * tremor
    df.iloc[i0:i1, cols["gyro_y"]] += smooth_noise(rng, n_pre, 0.5, kernel=8) * tremor

    df.iloc[i0:i1, cols["label"]] = "pre_syncope"
    df.iloc[i0:i1, cols["event_phase"]] = "pre_syncope"

    # SYNCOPE + FALL
    n_sync = i2 - i1
    fall_len = min(20, n_sync)
    fall_profile = np.array([0, 1, 4, 10, 18, 12, 6, 3, 1.5, 0.8, 0.4, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    scale = rng.uniform(0.8, 1.4) * variant_cfg["movement"]

    post_fall_jerks = np.zeros(n_sync)
    n_jerks = int(rng.integers(1, 4))
    if n_sync - fall_len > 5:
        jerk_idx = rng.choice(np.arange(fall_len + 2, n_sync), size=n_jerks, replace=False)
        post_fall_jerks[jerk_idx] = rng.uniform(0.3, 0.9, size=n_jerks)

    for col, spike_scale in (("accel_x", 1.0), ("accel_y", 0.7), ("accel_z", 0.6)):
        spike = np.zeros(n_sync)
        spike[:fall_len] = fall_profile[:fall_len] * spike_scale * scale
        df.iloc[i1:i2, cols[col]] = spike + 0.5 * post_fall_jerks + smooth_noise(rng, n_sync, 0.05, kernel=6)

    orientation = rng.choice(["x", "y", "z"])
    gravity = {"x": np.array([9.81, 0.0, 0.0]), "y": np.array([0.0, 9.81, 0.0]), "z": np.array([0.0, 0.0, 9.81])}[orientation]
    settle = np.linspace(0.0, 1.0, n_sync)
    for idx, col in enumerate(("accel_x", "accel_y", "accel_z")):
        current = df.iloc[i1:i2, cols[col]].values
        df.iloc[i1:i2, cols[col]] = current * (1.0 - settle) + gravity[idx] * settle

    lying_std = 0.05
    for col, spike_scale in (("gyro_x", 0.40), ("gyro_y", 0.25), ("gyro_z", 0.15)):
        spike = np.zeros(n_sync)
        spike[:fall_len] = fall_profile[:fall_len] * spike_scale * scale
        df.iloc[i1:i2, cols[col]] = spike + 0.5 * post_fall_jerks + smooth_noise(rng, n_sync, lying_std, kernel=6)

    t_sync = np.arange(n_sync) / fs
    hr_start = max(45.0, float(new_hr[-1]))
    hr_floor = max(38.0, hr_start - variant_cfg["hr_drop"])
    hr_sync = approach(hr_floor, hr_start, t_sync, tau=0.4)
    sync_move = np.sqrt(df.iloc[i1:i2, cols["accel_x"]].values ** 2 + df.iloc[i1:i2, cols["accel_y"]].values ** 2)
    hr_sync = hr_sync + np.clip(sync_move, 0.0, 1.5) * 5.0
    df.iloc[i1:i2, cols["heart_rate"]] = hr_sync

    spo2_nadir = rng.uniform(87.5, 89.5) - 0.4 * variant_cfg["spo2_drop"]
    spo2_start = float(pre_spo2[-1])
    spo2_sync = approach(spo2_nadir, spo2_start, t_sync, tau=0.7)
    spo2_sync = spo2_sync - 0.03 * (baseline_hr - hr_sync)
    df.iloc[i1:i2, cols["spo2"]] = spo2_sync

    temp_at_syncope_start = baseline_temp - 0.8
    temp_at_syncope_end = baseline_temp - 1.2
    df.iloc[i1:i2, cols["skin_temp"]] = approach(
        temp_at_syncope_end, temp_at_syncope_start, t_sync, tau=0.8
    )

    df.iloc[i1:i2, cols["label"]] = "syncope"
    df.iloc[i1:i2, cols["event_phase"]] = "syncope"

    # RECOVERY
    n_rec = i3 - i2
    t_rec = np.arange(n_rec) / fs

    rec_hr_start = float(df.iloc[i2 - 1, cols["heart_rate"]])
    hr_recovery = approach(baseline_hr, rec_hr_start, t_rec, tau=0.9)
    mayer_wave = 1.8 * np.sin(2.0 * np.pi * 0.1 * t_rec + rng.uniform(0.0, 2.0 * np.pi))
    rsa_wave = 1.2 * np.sin(2.0 * np.pi * 0.24 * t_rec + rng.uniform(0.0, 2.0 * np.pi))
    rec_move = np.clip(np.linspace(0.02, 0.20, n_rec) * variant_cfg["movement"], 0.0, 0.5)
    hr_recovery = hr_recovery + rec_move * 5.0 + mayer_wave + rsa_wave
    df.iloc[i2:i3, cols["heart_rate"]] = hr_recovery + smooth_noise(rng, n_rec, 0.8, kernel=10)

    rec_spo2_start = float(df.iloc[i2 - 1, cols["spo2"]])
    spo2_recovery = approach(baseline_spo2, rec_spo2_start, t_rec, tau=1.2)
    spo2_recovery = spo2_recovery - 0.03 * (baseline_hr - hr_recovery)
    df.iloc[i2:i3, cols["spo2"]] = np.clip(spo2_recovery + smooth_noise(rng, n_rec, 0.2, kernel=12), None, 99.0)

    temp_recovery = approach(baseline_temp, temp_at_syncope_end, t_rec, tau=2.4)
    df.iloc[i2:i3, cols["skin_temp"]] = temp_recovery + smooth_noise(rng, n_rec, 0.03, kernel=14)

    micro = smooth_noise(rng, n_rec, 0.02, kernel=20)
    df.iloc[i2:i3, cols["accel_x"]] = gravity[0] + micro
    df.iloc[i2:i3, cols["accel_y"]] = gravity[1] + micro
    df.iloc[i2:i3, cols["accel_z"]] = gravity[2] + smooth_noise(rng, n_rec, 0.015, kernel=20)

    gyro_scale = np.linspace(lying_std, lying_std * 1.8, n_rec)
    for col in ("gyro_x", "gyro_y", "gyro_z"):
        df.iloc[i2:i3, cols[col]] = smooth_noise(rng, n_rec, 1.0, kernel=8) * gyro_scale

    # Soften phase boundaries to avoid hard corners between event segments.
    blend_len = max(3, int(fs * 1.5))
    for signal in ("heart_rate", "spo2", "skin_temp"):
        pre = df.iloc[i0:i1, cols[signal]].values
        sync = df.iloc[i1:i2, cols[signal]].values
        rec = df.iloc[i2:i3, cols[signal]].values

        pre, sync = blend_transition(pre, sync, blend_len)
        sync, rec = blend_transition(sync, rec, blend_len)

        df.iloc[i0:i1, cols[signal]] = pre
        df.iloc[i1:i2, cols[signal]] = sync
        df.iloc[i2:i3, cols[signal]] = rec

    df.iloc[i2:i3, cols["label"]] = "recovery"
    df.iloc[i2:i3, cols["event_phase"]] = "recovery"

    if logger:
        logger.debug(
            f"  Event planted | variant={variant} start={event_start_s}s "
            f"pre={pre_dur}s sync={sync_dur}s rec={rec_dur}s "
            f"HR_target={baseline_hr:.1f} SpO2_target={baseline_spo2:.1f} "
            f"temp_target={baseline_temp:.2f}"
        )

    return df, event_meta
