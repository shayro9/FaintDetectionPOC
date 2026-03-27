"""
Phase 1: Bronze Layer Ingestion - FAST VERSION
===============================================

Ingests raw propofol study data into Parquet Bronze tables.
Key design: store raw signal values as arrays, NOT as individual records.

Usage:
  python bronze_load_fast.py

Output:
  data/bronze/
  ├── signals_consolidated.parquet
  ├── loc_roc_consolidated.parquet
  └── bronze_load.log
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json

import pandas as pd
import numpy as np

DATA_DIR = Path("data") / "human_data" / "Data"
BRONZE_DIR = Path("data") / "bronze"

SIGNAL_FILES = [
    "eda_tonic",
    "muPR", "sigmaPR", "mu_amp", "sigma_amp",
    "muRR", "sigmaRR", "muHR", "sigmaHR",
    "LF", "HF", "LFnu", "HFnu",
    "pow_tot", "ratio",
]

SUBJECTS = list(range(1, 10))


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.messages = []
    
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {level:8s} {msg}"
        print(line)
        self.messages.append(line)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            f.write('\n'.join(self.messages))


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_signal_values(file_path: Path) -> tuple:
    """Load signal CSV (single row or columnar format)."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Detect format: if first line has commas, it's comma-separated; else it's columnar
        if ',' in first_line:
            # Single-row format: comma-separated
            values_str = first_line.split(',')
        else:
            # Columnar format: newline-separated
            with open(file_path, 'r') as f:
                values_str = [line.strip() for line in f if line.strip()]
        
        values = np.array([float(v) if v != 'NaN' else np.nan for v in values_str])
        return values, len(values_str)
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None, 0


def run_bronze():
    """Execute Phase 1."""
    
    print("\n" + "="*80)
    print("  PHASE 1: BRONZE LAYER INGESTION")
    print("="*80 + "\n")
    
    log_file = BRONZE_DIR / "bronze_load.log"
    logger = Logger(log_file)
    
    logger.log("Phase 1 Bronze Layer Ingestion started")
    logger.log(f"Source: {DATA_DIR}")
    logger.log(f"Output: {BRONZE_DIR}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: INGEST SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.log("Starting signal ingestion...")
    signal_records = []
    total_values = 0
    
    for subject_id in SUBJECTS:
        logger.log(f"  Subject {subject_id}...")
        
        for signal_name in SIGNAL_FILES:
            files = list(DATA_DIR.glob(f"S{subject_id}_{signal_name}.csv"))
            if not files:
                continue
            
            file_path = files[0]
            values, total_count = load_signal_values(file_path)
            
            if values is None:
                continue
            
            # Stats (filter out inf and NaN)
            null_count = np.isnan(values).sum()
            null_rate = null_count / total_count if total_count > 0 else 0
            valid = values[~(np.isnan(values) | np.isinf(values))]
            
            # Store as one record (with array)
            record = {
                "subject_id": subject_id,
                "signal_name": signal_name,
                "value_count": int(total_count),
                "null_count": int(null_count),
                "null_rate": float(null_rate),
                "mean": float(np.mean(valid)) if len(valid) > 0 else None,
                "std": float(np.std(valid)) if len(valid) > 0 else None,
                "min": float(np.min(valid)) if len(valid) > 0 else None,
                "max": float(np.max(valid)) if len(valid) > 0 else None,
                "values_bytes": values.tobytes(),  # Store as bytes
                "source_file": file_path.name,
                "source_file_hash": compute_file_hash(file_path),
                "ingested_at": datetime.now().isoformat(),
            }
            signal_records.append(record)
            total_values += total_count
    
    logger.log(f"Loaded {len(signal_records)} signals ({total_values:,} values total)\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: INGEST LOC/ROC
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.log("Starting LOC/ROC ingestion...")
    loc_roc_records = []
    
    loc_roc_file = DATA_DIR / "LOC_ROC.csv"
    if loc_roc_file.exists():
        with open(loc_roc_file, 'r') as f:
            for subject_id, line in enumerate(f, start=1):
                if subject_id > 9:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                values_str = line.split(',')
                if len(values_str) >= 2:
                    loc_time = float(values_str[0]) if values_str[0] != 'NaN' else None
                    roc_time = float(values_str[1]) if values_str[1] != 'NaN' else None
                    
                    record = {
                        "subject_id": subject_id,
                        "loc_timestamp": loc_time,
                        "roc_timestamp": roc_time,
                        "consciousness_duration_sec": (roc_time - loc_time) if (loc_time and roc_time) else None,
                    }
                    loc_roc_records.append(record)
    
    logger.log(f"Loaded {len(loc_roc_records)} LOC/ROC records\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: SAVE BRONZE TABLES
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.log("Saving Bronze tables...")
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save signals
        signals_df = pd.DataFrame(signal_records)
        signals_parquet = BRONZE_DIR / "signals_consolidated.parquet"
        signals_df.to_parquet(signals_parquet, engine='pyarrow', compression='snappy', index=False)
        logger.log(f"  Saved signals: {len(signals_df)} records")
        
        # Save LOC/ROC
        loc_roc_df = pd.DataFrame(loc_roc_records)
        loc_roc_parquet = BRONZE_DIR / "loc_roc_consolidated.parquet"
        loc_roc_df.to_parquet(loc_roc_parquet, engine='pyarrow', compression='snappy', index=False)
        logger.log(f"  Saved LOC/ROC: {len(loc_roc_df)} records")
        
        # Metadata
        metadata = {
            "phase": "bronze_layer",
            "created_at": datetime.now().isoformat(),
            "signals": len(signal_records),
            "total_values": total_values,
            "loc_roc": len(loc_roc_records),
        }
        metadata_file = BRONZE_DIR / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        logger.log(f"ERROR: {e}", "ERROR")
        logger.save()
        return False
    
    # Summary
    logger.log("\n" + "="*80)
    logger.log("PHASE 1 COMPLETE")
    logger.log("="*80)
    logger.log(f"Signals: {len(signal_records)} ingested")
    logger.log(f"Values: {total_values:,} total")
    logger.log(f"LOC/ROC: {len(loc_roc_records)} records")
    logger.log(f"\nReady for Phase 2: Silver Layer")
    
    logger.save()
    print(f"\nLog: {log_file}")
    
    return True


if __name__ == "__main__":
    success = run_bronze()
    sys.exit(0 if success else 1)
