import pandas as pd
import numpy as np

from config import FS
from events import plant_faint_event
from sessions import simulate_session
from user_profile import clear_profile_cache


def build_dataset(
    n_normal_sessions: int   = 30,
    n_event_sessions:  int   = 20,
    duration_s:        int   = 300,
    fs:                int   = FS,
    event_activities:  list  = None,
    normal_activities: list  = None,
    logger=None,
) -> tuple[pd.DataFrame, list]:
    """
    Build a full labeled dataset.

    Returns
    -------
    df         : complete DataFrame (all sessions concatenated)
    event_log  : list of dicts describing each planted event
    """
    if event_activities  is None: event_activities  = ["walking", "running"]
    if normal_activities is None: normal_activities = ["walking", "running", "sleeping"]

    # Reset the per-user profile cache so every build starts from a clean state.
    # This ensures reproducibility when build_dataset() is called more than once.
    clear_profile_cache()

    all_dfs   = []
    event_log = []
    rng       = np.random.default_rng(0)

    if logger:
        logger.info(f"Building dataset | normal={n_normal_sessions} "
                    f"event={n_event_sessions} duration={duration_s}s")

    # ── Normal sessions ──────────────────────────────────
    if logger:
        logger.info(f"Generating {n_normal_sessions} normal sessions …")

    for i in range(n_normal_sessions):
        activity = str(rng.choice(normal_activities))
        uid      = int(rng.integers(1, 21))  # 20 different users
        seed     = int(rng.integers(0, 100000))
        df       = simulate_session(activity, duration_s, fs, uid, seed,
                                    artifacts=True, logger=logger)
        df["session_id"]    = f"normal_{i:04d}"
        df["has_event"]     = False
        all_dfs.append(df)

        if (i+1) % 10 == 0 and logger:
            logger.info(f"  Normal sessions: {i+1}/{n_normal_sessions}")

    # ── Sessions with faint events ────────────────────────
    if logger:
        logger.info(f"Generating {n_event_sessions} sessions with faint events …")

    for i in range(n_event_sessions):
        activity = str(rng.choice(event_activities))
        uid      = int(rng.integers(1, 21))
        seed     = int(rng.integers(0, 100000))

        df = simulate_session(activity, duration_s, fs, uid, seed,
                              artifacts=True, logger=logger)

        # Plant event in the middle third of the session
        margin     = duration_s // 5
        pre_dur_max = 50
        sync_dur_max = 20
        rec_dur_max  = 90
        event_budget = pre_dur_max + sync_dur_max + rec_dur_max
        latest_start = duration_s - event_budget - margin
        event_start  = int(rng.integers(margin, max(margin+1, latest_start)))

        df, meta = plant_faint_event(df, event_start, fs=fs, rng=rng, logger=logger)

        if meta is not None:
            meta["session_id"] = f"event_{i:04d}"
            meta["user_id"]    = uid
            meta["activity"]   = activity
            event_log.append(meta)

        df["session_id"] = f"event_{i:04d}"
        df["has_event"]  = True
        all_dfs.append(df)

        if (i+1) % 10 == 0 and logger:
            logger.info(f"  Event sessions: {i+1}/{n_event_sessions}")

    full_df = pd.concat(all_dfs, ignore_index=True)

    if logger:
        counts = full_df["label"].value_counts()
        logger.info("Label distribution:")
        for lbl, cnt in counts.items():
            logger.info(f"  {lbl:15s}  {cnt:>8,} samples  ({cnt/len(full_df)*100:.1f}%)")

    return full_df, event_log