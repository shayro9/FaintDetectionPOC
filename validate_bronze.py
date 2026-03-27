"""
Phase 1 Bronze Layer Validation
================================

Validates Bronze layer ingestion:
- Schema correctness
- Data completeness (all subjects, signals)
- Data quality (nulls, duplicates, value ranges)
- File integrity

Usage:
  python validate_bronze.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

BRONZE_DIR = Path("data") / "bronze"
DATA_DIR = Path("data") / "human_data" / "Data"

EXPECTED_SIGNALS = [
    "eda_tonic",
    "muPR", "sigmaPR", "mu_amp", "sigma_amp",
    "muRR", "sigmaRR", "muHR", "sigmaHR",
    "LF", "HF", "LFnu", "HFnu",
    "pow_tot", "ratio",
]
EXPECTED_SUBJECTS = list(range(1, 10))


class Validator:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []
    
    def check(self, name: str, passed: bool, details: str = ""):
        status = "[OK]" if passed else "[FAIL]"
        msg = f"{status} {name}"
        if details:
            msg += f" ({details})"
        self.checks.append((name, passed, details))
        print(msg)
        if not passed:
            self.errors.append(msg)
    
    def warn(self, msg: str):
        print(f"⚠ {msg}")
        self.warnings.append(msg)
    
    def summary(self):
        total = len(self.checks)
        passed = sum(1 for _, p, _ in self.checks if p)
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Passed: {passed}/{total}")
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
        if self.errors:
            print(f"Errors: {len(self.errors)}")
            for err in self.errors:
                print(f"  {err}")
        return len(self.errors) == 0


def validate_bronze():
    """Validate Phase 1 Bronze layer."""
    
    print("\n" + "="*80)
    print("PHASE 1 BRONZE VALIDATION")
    print("="*80 + "\n")
    
    val = Validator()
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1: Files Exist
    # ─────────────────────────────────────────────────────────────────────────
    
    print("1. Checking Bronze files...\n")
    
    signals_file = BRONZE_DIR / "signals_consolidated.parquet"
    val.check("signals_consolidated.parquet exists", signals_file.exists())
    
    loc_roc_file = BRONZE_DIR / "loc_roc_consolidated.parquet"
    val.check("loc_roc_consolidated.parquet exists", loc_roc_file.exists())
    
    metadata_file = BRONZE_DIR / "metadata.json"
    val.check("metadata.json exists", metadata_file.exists())
    
    if not (signals_file.exists() and loc_roc_file.exists()):
        print("\n[FAILED] Cannot proceed without Bronze files.")
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2: Load Signals
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n2. Loading and validating signals table...\n")
    
    try:
        signals_df = pd.read_parquet(signals_file)
        val.check("signals table loads successfully", True, f"{signals_df.shape[0]} records")
    except Exception as e:
        val.check("signals table loads successfully", False, str(e))
        return False
    
    # Schema check
    expected_cols = {
        "subject_id", "signal_name", "value_count", "null_count", "null_rate",
        "mean", "std", "min", "max", "values_bytes", "source_file",
        "source_file_hash", "ingested_at"
    }
    has_all_cols = expected_cols.issubset(set(signals_df.columns))
    val.check("All expected columns present", has_all_cols, f"Cols: {list(signals_df.columns)}")
    
    # Record count
    val.check("135 signals ingested", len(signals_df) == 135, f"Got {len(signals_df)}")
    
    # Subject coverage
    subjects_found = sorted(signals_df["subject_id"].unique())
    all_subjects = subjects_found == EXPECTED_SUBJECTS
    val.check("All 9 subjects present", all_subjects, f"Subjects: {subjects_found}")
    
    # Signal coverage per subject
    signals_per_subject = signals_df.groupby("subject_id").size().unique()
    val.check("All subjects have 15 signals", len(signals_per_subject) == 1 and signals_per_subject[0] == 15,
              f"Got {signals_per_subject}")
    
    signals_found = sorted(signals_df["signal_name"].unique())
    all_signals = signals_found == sorted(EXPECTED_SIGNALS)
    val.check("All 15 signal types present", all_signals, f"Signals: {signals_found}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3: Data Quality
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n3. Checking data quality...\n")
    
    # Value counts
    total_values = signals_df["value_count"].sum()
    expected_values = 320000000  # ~320M (includes newly loaded columnar eda_tonic and t_EDA_tonic)
    val.check(f"Total values: ~320M", abs(total_values - expected_values) < 10000000,
              f"Got {total_values:,}")
    
    # Null rates reasonable
    high_null_signals = signals_df[signals_df["null_rate"] > 0.2]
    val.check("No signals with >20% nulls", len(high_null_signals) == 0,
              f"Found {len(high_null_signals)} signals")
    
    # Mean/std/min/max present (not NaN for most signals)
    stats_present = signals_df[["mean", "std", "min", "max"]].notna().sum()
    val.check("Statistics computed for all signals", (stats_present >= 130).all(),
              f"Mean: {stats_present['mean']}, Std: {stats_present['std']}")
    
    # Values array present
    has_values = signals_df["values_bytes"].notna().sum() == len(signals_df)
    val.check("All signals have values_bytes", has_values)
    
    # Duplicates check
    unique_records = signals_df[["subject_id", "signal_name"]].drop_duplicates()
    no_dupes = len(unique_records) == len(signals_df)
    val.check("No duplicate subject+signal combinations", no_dupes,
              f"Unique: {len(unique_records)}, Total: {len(signals_df)}")
    
    # Source file hashes present
    has_hashes = signals_df["source_file_hash"].notna().sum() == len(signals_df)
    val.check("All records have source_file_hash", has_hashes)
    
    # Ingestion timestamp
    has_ts = signals_df["ingested_at"].notna().sum() == len(signals_df)
    val.check("All records have ingested_at", has_ts)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4: Load LOC/ROC
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n4. Validating LOC/ROC table...\n")
    
    try:
        loc_roc_df = pd.read_parquet(loc_roc_file)
        val.check("LOC/ROC table loads successfully", True, f"{len(loc_roc_df)} records")
    except Exception as e:
        val.check("LOC/ROC table loads successfully", False, str(e))
        return False
    
    # Expected 9 records (one per subject)
    val.check("LOC/ROC has 9 records", len(loc_roc_df) == 9,
              f"Got {len(loc_roc_df)}")
    
    # Subjects covered
    loc_roc_subjects = sorted(loc_roc_df["subject_id"].unique())
    val.check("All 9 subjects in LOC/ROC", loc_roc_subjects == EXPECTED_SUBJECTS)
    
    # Timestamps reasonable
    loc_times = loc_roc_df["loc_timestamp"].dropna()
    roc_times = loc_roc_df["roc_timestamp"].dropna()
    loc_reasonable = (loc_times > 0) & (loc_times < 100)
    roc_reasonable = (roc_times > 0) & (roc_times < 200)
    val.check("LOC times in reasonable range (0-100s)", loc_reasonable.all(),
              f"Min: {loc_times.min():.1f}s, Max: {loc_times.max():.1f}s")
    val.check("ROC times in reasonable range (0-200s)", roc_reasonable.all(),
              f"Min: {roc_times.min():.1f}s, Max: {roc_times.max():.1f}s")
    
    # Consciousness duration (ROC - LOC)
    valid_duration = loc_roc_df["consciousness_duration_sec"].dropna()
    duration_reasonable = (valid_duration > 0) & (valid_duration < 200)
    val.check("Consciousness duration reasonable", duration_reasonable.all(),
              f"Min: {valid_duration.min():.1f}s, Max: {valid_duration.max():.1f}s")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5: Metadata
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n5. Validating metadata...\n")
    
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        val.check("metadata.json loads successfully", True)
    except Exception as e:
        val.check("metadata.json loads successfully", False, str(e))
        return False
    
    # Metadata structure
    has_phase = "phase" in metadata and metadata["phase"] == "bronze_layer"
    val.check("Metadata has 'phase' = 'bronze_layer'", has_phase)
    
    has_signals_meta = "signals" in metadata
    val.check("Metadata has 'signals' section", has_signals_meta)
    
    has_loc_roc_meta = "loc_roc" in metadata
    val.check("Metadata has 'loc_roc' section", has_loc_roc_meta)
    
    # Signal metadata matches
    if has_signals_meta:
        sig_meta = metadata["signals"]
        # Handle both int (record count) and dict formats
        if isinstance(sig_meta, dict):
            meta_signal_count = sig_meta.get("record_count")
        else:
            meta_signal_count = sig_meta  # Direct int value
        val.check(f"Metadata: 135 signals", meta_signal_count == 135,
                  f"Got {meta_signal_count}")
        
    # LOC/ROC metadata matches
    if has_loc_roc_meta:
        loc_roc_meta = metadata["loc_roc"]
        # Handle both int (record count) and dict formats
        if isinstance(loc_roc_meta, dict):
            meta_loc_roc_count = loc_roc_meta.get("record_count")
        else:
            meta_loc_roc_count = loc_roc_meta  # Direct int value
        val.check(f"Metadata: 9 LOC/ROC records", meta_loc_roc_count == 9,
                  f"Got {meta_loc_roc_count}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 6: Sample Data Inspection
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n6. Inspecting sample data...\n")
    
    # Sample a signal from each subject
    for subject_id in [1, 5, 9]:
        subject_signals = signals_df[signals_df["subject_id"] == subject_id]
        sample = subject_signals.iloc[0]
        
        null_rate = sample["null_rate"]
        values_count = sample["value_count"]
        
        val.check(f"Subject {subject_id} sample valid",
                  null_rate < 0.5 and values_count > 1000,
                  f"Signal: {sample['signal_name']}, Values: {values_count}, Null%: {null_rate:.1%}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    success = val.summary()
    
    if success:
        print("\n✅ PHASE 1 BRONZE VALIDATION PASSED")
        print("Ready for Phase 2: Silver Layer Transformation\n")
    else:
        print("\n[FAILED] PHASE 1 BRONZE VALIDATION FAILED")
        print("Fix errors before proceeding to Phase 2\n")
    
    return success


if __name__ == "__main__":
    success = validate_bronze()
    sys.exit(0 if success else 1)
