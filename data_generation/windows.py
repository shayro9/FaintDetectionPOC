import numpy as np

from config import LABEL_MAP

def create_windows(df, window_size=128, stride=32, logger=None):
    """
    Sliding window segmentation grouped by session to avoid cross-session leakage.

    Window label = the most common label in that window
    (so a window is 'pre_syncope' only if most samples are pre_syncope)

    Returns
    -------
    X        (N, window_size, 9)  float32
    y        (N,)                 int   0=normal 1=pre_syncope 2=syncope 3=recovery
    sessions (N,)                 str   session_id for grouping
    users    (N,)                 int   user_id for LOSO split
    """
    sensor_cols = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x",  "gyro_y",  "gyro_z",
        "heart_rate", "spo2", "skin_temp",
    ]

    X, y, sessions, users = [], [], [], []
    label_enc = {v: k for k, v in LABEL_MAP.items()}   # int → str (for logging)

    for (sid, uid), grp in df.groupby(["session_id", "user_id"]):
        data   = grp[sensor_cols].ffill().bfill().values.astype(np.float32)
        labels = grp["label"].map(LABEL_MAP).fillna(0).values.astype(int)

        for i in range(0, len(data) - window_size, stride):
            window_labels = labels[i : i + window_size]
            # Majority vote for window label
            counts    = np.bincount(window_labels, minlength=len(LABEL_MAP))
            win_label = int(np.argmax(counts))
            X.append(data[i : i + window_size])
            y.append(win_label)
            sessions.append(sid)
            users.append(uid)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if logger:
        logger.info(f"Windows created: X={X.shape}  y={y.shape}")
        win_counts = np.bincount(y, minlength=len(LABEL_MAP))
        for idx, cnt in enumerate(win_counts):
            name = {v: k for k, v in LABEL_MAP.items()}.get(idx, str(idx))
            logger.info(f"  Window label [{idx}] {name:15s}  {cnt:>6,} windows")

    return X, y, np.array(sessions), np.array(users)