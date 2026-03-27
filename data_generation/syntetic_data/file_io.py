import logging
import os
import sys
import pandas as pd
import numpy as np



def setup_logger(log_path="simulation.log"):
    logger = logging.getLogger("smartwatch_sim")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — full detail
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO only
    ch = logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_outputs(df, X, y, sessions, users, event_log,
                 csv_path="smartwatch_dataset.csv",
                 npz_path="smartwatch_windows.npz",
                 eventlog_path="event_log.csv",
                 logger=None):

    # Raw CSV
    df.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / 1e6
    if logger:
        logger.info(f"Saved raw CSV -> {csv_path}  ({size_mb:.1f} MB, {len(df):,} rows)")

    # Windowed numpy arrays
    np.savez_compressed(npz_path, X=X, y=y, sessions=sessions, users=users)
    size_mb = os.path.getsize(npz_path) / 1e6
    if logger:
        logger.info(f"Saved windows  -> {npz_path}  ({size_mb:.1f} MB)")

    # Event metadata
    if event_log:
        pd.DataFrame(event_log).to_csv(eventlog_path, index=False)
        if logger:
            logger.info(f"Saved event log -> {eventlog_path}  ({len(event_log)} events)")