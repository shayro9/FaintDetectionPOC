"""
================================================================================
PHASE 2: SILVER LAYER TRANSFORMATION
================================================================================

Silver layer transforms raw Bronze signals into model-ready windowed samples:
1. Denormalize: Expand signal byte arrays into time-indexed rows
2. Validate: Check data quality, null rates, value ranges
3. Normalize: Apply per-subject z-score normalization
4. Label: Assign consciousness state (PRE-LOC, LOC, POST-ROC, BASELINE)
5. Sample: Create fixed-length windows for model input

Output: data/silver/signals_windowed.parquet with columns:
  subject_id, window_start_sec, window_end_sec, signal_name, values (normalized),
  consciousness_state, has_loc_in_window
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../../data")
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
SILVER_DIR.mkdir(exist_ok=True)

WINDOW_LENGTH_SEC = 30  # 30-second windows for prediction
SAMPLE_RATE_HZ = 10  # Assumed base rate for time-axis construction
MAX_SAMPLE_RATE_HZ = 20  # Hard cap — all signals are derived physio metrics (≤10 Hz in practice)

# Consciousness states relative to LOC/ROC
WINDOW_STATES = {
    'BASELINE': 'Baseline (before pre-LOC)',
    'PRE_LOC': 'Pre-LOC (30s before LOC)',
    'LOC': 'Loss of consciousness',
    'POST_ROC': 'Post-recovery',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.start_time = datetime.now()
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] INFO       {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

logger = Logger(SILVER_DIR / "silver_transform.log")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD BRONZE TABLES
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Loading Bronze layer tables...")
signals_bronze = pd.read_parquet(BRONZE_DIR / "signals_consolidated.parquet")
loc_roc_bronze = pd.read_parquet(BRONZE_DIR / "loc_roc_consolidated.parquet")

# Create subject → LOC/ROC mapping
subject_loc_roc = {}
for _, row in loc_roc_bronze.iterrows():
    subject_loc_roc[row['subject_id']] = {
        'loc_sec': row['loc_timestamp'],
        'roc_sec': row['roc_timestamp'],
        'duration_sec': row['consciousness_duration_sec'],
    }

logger.log(f"Loaded {len(signals_bronze)} signal records and {len(loc_roc_bronze)} LOC/ROC records\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DENORMALIZE SIGNALS (EXPAND BYTE ARRAYS TO TIME-INDEXED ROWS)
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 1: Denormalizing signals...")

denormalized_records = []

for idx, signal_row in signals_bronze.iterrows():
    subject_id = signal_row['subject_id']
    signal_name = signal_row['signal_name']
    value_count = signal_row['value_count']
    
    # Deserialize byte array
    values_bytes = signal_row['values_bytes']
    if isinstance(values_bytes, bytes):
        values = np.frombuffer(values_bytes, dtype=np.float64)
    else:
        logger.log(f"  WARN Skipping {signal_name} for subject {subject_id}: invalid byte format")
        continue
    
    # Assume linear time spacing (compute from value count)
    # Estimate: signals span from LOC - 60s to ROC + 60s
    loc_roc = subject_loc_roc.get(subject_id, {})
    loc_sec = loc_roc.get('loc_sec', 0)
    roc_sec = loc_roc.get('roc_sec', 500)
    
    # Create time axis. Estimate sample rate from value count / expected duration,
    # then cap at MAX_SAMPLE_RATE_HZ — all signals here are derived physio metrics
    # (mean/sigma of PR, RR, HR, HRV power) so rates above ~10 Hz are artefacts
    # of the value-count / duration estimate being off.
    estimated_duration_sec = (roc_sec - loc_sec) * 1.5 + 60
    raw_rate = value_count / estimated_duration_sec if estimated_duration_sec > 0 else SAMPLE_RATE_HZ
    estimated_sample_rate = min(max(raw_rate, 1), MAX_SAMPLE_RATE_HZ)

    time_axis = np.arange(value_count) / estimated_sample_rate
    
    # Store denormalized record
    denormalized_records.append({
        'subject_id': subject_id,
        'signal_name': signal_name,
        'time_axis': time_axis,
        'values': values,
        'value_count': value_count,
        'sample_rate_hz': estimated_sample_rate,
        'source_signal_row_idx': idx,
    })

logger.log(f"  OK Denormalized {len(denormalized_records)} signal vectors\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: VALIDATE DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 2: Validating data quality...")

validation_issues = []

for record in denormalized_records:
    values = record['values']
    subject_id = record['subject_id']
    signal_name = record['signal_name']
    
    # Check for all NaN
    if np.all(np.isnan(values)):
        validation_issues.append(f"  WARN {signal_name} (S{subject_id}): All NaN values")
        continue
    
    # Check for inf
    if np.any(np.isinf(values)):
        inf_count = np.sum(np.isinf(values))
        validation_issues.append(f"  WARN {signal_name} (S{subject_id}): {inf_count} inf values")
    
    # Check null rate
    null_rate = np.sum(np.isnan(values)) / len(values)
    if null_rate > 0.5:
        validation_issues.append(f"  WARN {signal_name} (S{subject_id}): {null_rate:.1%} null")

if validation_issues:
    for issue in validation_issues[:10]:  # Log first 10
        logger.log(issue)
    if len(validation_issues) > 10:
        logger.log(f"  ... and {len(validation_issues) - 10} more issues")
else:
    logger.log("  OK No critical validation issues\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: NORMALIZE (PER-SUBJECT Z-SCORE)
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 3: Normalizing signals (per-subject z-score)...")

normalized_by_subject = {}

for record in denormalized_records:
    subject_id = record['subject_id']
    values = record['values']
    
    # Handle NaN and inf
    valid_mask = ~(np.isnan(values) | np.isinf(values))
    valid_values = values[valid_mask]
    
    if len(valid_values) > 0:
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        # Normalize (avoid division by zero)
        if std_val > 1e-10:
            normalized = np.where(valid_mask, (values - mean_val) / std_val, np.nan)
        else:
            normalized = np.where(valid_mask, (values - mean_val), np.nan)
    else:
        normalized = values
    
    record['values_normalized'] = normalized
    record['norm_mean'] = mean_val if len(valid_values) > 0 else 0
    record['norm_std'] = std_val if len(valid_values) > 0 else 1

logger.log(f"  OK Normalized {len(denormalized_records)} signals\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CREATE WINDOWED SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 4: Creating windowed samples...")

windowed_samples = []

# Group by subject
by_subject = {}
for record in denormalized_records:
    subj = record['subject_id']
    if subj not in by_subject:
        by_subject[subj] = {}
    by_subject[subj][record['signal_name']] = record

window_sample_count = 0

for subject_id, signal_dict in by_subject.items():
    loc_sec = subject_loc_roc[subject_id]['loc_sec']
    roc_sec = subject_loc_roc[subject_id]['roc_sec']
    
    # Determine window range: PRE-LOC (30s before) to POST-ROC (30s after)
    window_start_min = max(0, loc_sec - 120)  # Include baseline before PRE-LOC window
    window_end_max = roc_sec + 60
    
    # Create sliding windows
    window_start_sec = window_start_min
    while window_start_sec + WINDOW_LENGTH_SEC <= window_end_max:
        window_end_sec = window_start_sec + WINDOW_LENGTH_SEC
        
        # Determine consciousness state
        if window_end_sec < loc_sec - 30:
            consciousness_state = 'BASELINE'
        elif window_end_sec < loc_sec:
            consciousness_state = 'PRE_LOC'
        elif window_start_sec < roc_sec:
            consciousness_state = 'LOC'
        else:
            consciousness_state = 'POST_ROC'
        
        has_loc_in_window = (window_start_sec < loc_sec < window_end_sec)
        
        # Extract feature window (all signals for this window)
        window_features = {
            'subject_id': subject_id,
            'window_start_sec': window_start_sec,
            'window_end_sec': window_end_sec,
            'consciousness_state': consciousness_state,
            'has_loc_in_window': has_loc_in_window,
        }
        
        # Extract signal values within window
        for signal_name, record in signal_dict.items():
            time_axis = record['time_axis']
            normalized_values = record['values_normalized']
            
            # Find indices within window
            mask = (time_axis >= window_start_sec) & (time_axis < window_end_sec)
            window_values = normalized_values[mask]
            
            # Store however many samples fall in this window — no forced padding.
            # Gold layer handles fixed-length shaping for each model type.
            window_features[f'{signal_name}_values'] = window_values
        
        windowed_samples.append(window_features)
        window_sample_count += 1
        
        window_start_sec += 10  # 10-second stride

logger.log(f"  OK Created {window_sample_count} windowed samples\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SAVE WINDOWED SAMPLES TO PARQUET
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 5: Saving windowed samples to Parquet...")

if len(windowed_samples) > 0:
    df_windowed = pd.DataFrame(windowed_samples)

    # Pickle preserves variable-length numpy arrays in-place with no extra
    # memory copies — parquet/pyarrow cannot handle ragged object columns well.
    output_file = SILVER_DIR / "signals_windowed.pkl"
    df_windowed.to_pickle(output_file)
    logger.log(f"  OK Saved {len(df_windowed)} samples to {output_file}\n")
else:
    logger.log("  WARN No windowed samples created!\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

logger.log("================================================================================")
logger.log("PHASE 2 COMPLETE")
logger.log("================================================================================\n")

if len(windowed_samples) > 0:
    print("\nSTATS Silver Layer Summary:")
    print(f"   • Subjects: {len(by_subject)}")
    print(f"   • Windowed samples: {window_sample_count}")
    print(f"   • Window length: {WINDOW_LENGTH_SEC}s @ {SAMPLE_RATE_HZ} Hz")
    print(f"   • Features per window: {len(signal_dict)} signals")
    
    # Class distribution
    state_counts = df_windowed['consciousness_state'].value_counts()
    print(f"\n   • Consciousness state distribution:")
    for state, count in state_counts.items():
        print(f"     - {state}: {count} ({count/len(df_windowed)*100:.1f}%)")
    
    print(f"\nDIR Outputs saved to: {SILVER_DIR}/")
    print(f"   • signals_windowed.parquet")
    print(f"   • silver_transform.log")
    print(f"\nOK Ready for Phase 3: Gold Layer (Feature Engineering & Model Training)\n")

logger.log(f"Ready for Phase 3: Gold Layer (Feature Engineering & Model Training)")
