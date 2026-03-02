"""
=============================================================
  Realistic Smartwatch Sensor Data Simulator
  with Planted Fainting Events
=============================================================
  Outputs:
    smartwatch_dataset.csv   — full raw sensor data + labels
    smartwatch_windows.npz   — windowed arrays ready for DL
    simulation.log           — detailed run log

  Requirements:
    pip install numpy pandas scipy

  Usage:
    python smartwatch_simulator.py
=============================================================
"""
import os
import time
from datetime import datetime

from config import FS, OUTPUT_PATH
from dataset import build_dataset
from file_io import setup_logger, save_outputs
from windows import create_windows


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    logger = setup_logger("simulation.log")

    logger.info("=" * 60)
    logger.info("  Smartwatch Sensor Data Simulator")
    logger.info(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    t_start = time.time()

    # ── Generate dataset ──────────────────────────────────
    df, event_log = build_dataset(
        n_normal_sessions = 40,
        n_event_sessions  = 30,
        duration_s        = 300,   # 5 minutes per session
        fs                = FS,
        logger            = logger,
    )

    # ── Summary stats ─────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Dataset summary:")
    logger.info(f"  Total samples   : {len(df):,}")
    logger.info(f"  Total sessions  : {df['session_id'].nunique()}")
    logger.info(f"  Unique users    : {df['user_id'].nunique()}")
    logger.info(f"  Activities      : {df['activity'].unique().tolist()}")
    logger.info(f"  Faint events    : {len(event_log)}")

    for act in df["activity"].unique():
        n = (df["activity"] == act).sum()
        logger.info(f"  {act:10s} samples: {n:>9,}")

    # ── Create windows ────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Creating sliding windows (size=128, stride=32) …")
    X, y, sessions, users = create_windows(df, window_size=128, stride=32, logger=logger)

    # ── Save ─────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Saving outputs …")
    save_outputs(
        df, X, y, sessions, users, event_log,
        csv_path=os.path.join(OUTPUT_PATH, "smartwatch_dataset.csv"),
        npz_path=os.path.join(OUTPUT_PATH, "smartwatch_windows.npz"),
        eventlog_path=os.path.join(OUTPUT_PATH, "event_log.csv"),
        logger=logger,
    )

    elapsed = time.time() - t_start
    logger.info("-" * 60)
    logger.info(f"Done (elapsed: {elapsed:.1f}s)")
    logger.info("=" * 60)

    # ── Quick console summary ─────────────────────────────
    print("\n" + "="*50)
    print("  Files generated:")
    print("  * smartwatch_dataset.csv  - raw sensor data")
    print("  * smartwatch_windows.npz  - X, y arrays for DL")
    print("  * event_log.csv           - planted event metadata")
    print("  * simulation.log          - full run log")
    print(f"\n  X shape : {X.shape}   (samples, timesteps, channels)")
    print(f"  y shape : {y.shape}")
    print(f"  Labels  : 0=normal  1=pre_syncope  2=syncope  3=recovery")
    print(f"\n  Tip: split train/test by user_id (LOSO) to avoid leakage!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()