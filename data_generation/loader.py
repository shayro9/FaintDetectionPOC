"""Load per-patient data from gold LOSO fold parquet files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_METADATA_COLS = [
    "subject_id",
    "window_start_sec",
    "window_end_sec",
    "consciousness_state",
    "has_loc_in_window",
    "high_risk",
    "target_class",
]


@dataclass
class Patient:
    id: str
    X_features: np.ndarray  # (N, F) feature matrix
    X_raw: np.ndarray       # (N, T, C) bronze signal windows; (N, 1, F) placeholder pre-pipeline
    y: np.ndarray           # (N,) binary labels
    time_to_loc: np.ndarray # (N,) seconds before LOC (positive = before, negative = after)


def _compute_time_to_loc(df: pd.DataFrame) -> np.ndarray:
    """Return seconds-before-LOC per window. Positive means the window precedes LOC onset."""
    loc_windows = df[df["consciousness_state"] == "LOC"]
    if loc_windows.empty:
        return np.zeros(len(df), dtype=np.float32)
    loc_onset = float(loc_windows["window_start_sec"].min())
    return (loc_onset - df["window_start_sec"].values).astype(np.float32)


def load_patients(data_path: str) -> list[Patient]:
    """Load one Patient per subject from gold fold test splits.

    Each fold's test.parquet contains all windows for the held-out subject, so
    iterating over fold directories recovers the full per-subject dataset.

    Args:
        data_path: Path to the ``data/`` directory (value of ``cfg["data"]["path"]``).

    Returns:
        List of Patient objects sorted by subject ID.
    """
    gold_dir = Path(data_path) / "gold"
    patients: list[Patient] = []

    for fold_dir in sorted(gold_dir.glob("fold_*")):
        test_file = fold_dir / "test.parquet"
        if not test_file.exists():
            continue

        df = pd.read_parquet(test_file)
        df = df.sort_values("window_start_sec").reset_index(drop=True)

        subject_id = str(int(df["subject_id"].iloc[0]))
        feature_cols = [c for c in df.columns if c not in _METADATA_COLS]

        X_features = df[feature_cols].values.astype(np.float32)
        y = df["high_risk"].values.astype(np.int32)
        time_to_loc = _compute_time_to_loc(df)

        # Load raw (N, T, C) signal array produced by gold_features.py.
        # Falls back to a feature-based placeholder if the pipeline hasn't been re-run yet.
        raw_path = fold_dir / "test_raw.npy"
        if raw_path.exists():
            X_raw = np.load(raw_path)  # (N, T, C=15)
        else:
            X_raw = X_features[:, np.newaxis, :]  # placeholder (N, 1, F)

        patients.append(Patient(
            id=subject_id,
            X_features=X_features,
            X_raw=X_raw,
            y=y,
            time_to_loc=time_to_loc,
        ))

    if not patients:
        raise FileNotFoundError(
            f"No fold directories found under {gold_dir}. "
            "Run gold_features.py to generate the gold layer first."
        )

    return patients
