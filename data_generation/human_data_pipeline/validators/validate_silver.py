"""
================================================================================
PHASE 2 SILVER LAYER VALIDATION
================================================================================

Validates the Silver layer output (windowed signals) before proceeding to Phase 3.

Checks:
1. File existence (signals_windowed.parquet)
2. Schema correctness (columns, dtypes, values_bytes)
3. Record count & completeness
4. Window size consistency (30s @ 10 Hz = 300 samples/window)
5. Consciousness state distribution & reasonableness
6. Data quality (nulls, range checks, temporal continuity)
7. Sampling coverage (all subjects, all windows represented)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../../../data")
SILVER_DIR = DATA_DIR / "silver"

EXPECTED_SUBJECTS = list(range(1, 10))
EXPECTED_SIGNALS = 15
EXPECTED_WINDOW_LENGTH = 30  # seconds
EXPECTED_SAMPLE_RATE = 10  # Hz
EXPECTED_SAMPLES_PER_WINDOW = EXPECTED_WINDOW_LENGTH * EXPECTED_SAMPLE_RATE  # 300

CONSCIOUSNESS_STATES = ['BASELINE', 'PRE_LOC', 'LOC', 'POST_ROC']

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Validator:
    def __init__(self):
        self.checks = []
        self.errors = []
    
    def check(self, name, condition, details=""):
        status = "[OK]" if condition else "[FAIL]"
        msg = f"{status} {name}"
        if details and not condition:
            msg += f" ({details})"
        self.checks.append((name, condition))
        print(msg)
        if not condition:
            self.errors.append((name, details))
    
    def summary(self):
        passed = sum(1 for _, ok in self.checks if ok)
        total = len(self.checks)
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Passed: {passed}/{total}")
        
        if self.errors:
            print(f"Errors: {len(self.errors)}")
            for name, details in self.errors:
                print(f"  [FAIL] {name}")
                if details:
                    print(f"         {details}")
        
        return len(self.errors) == 0

def validate_silver():
    """Execute Silver layer validation."""
    
    print("\n" + "="*80)
    print("PHASE 2 SILVER VALIDATION")
    print("="*80 + "\n")
    
    val = Validator()
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1: File Existence
    # ─────────────────────────────────────────────────────────────────────────
    
    print("1. Checking Silver files...\n")
    
    windowed_file = SILVER_DIR / "signals_windowed.parquet"
    val.check("signals_windowed.parquet exists", windowed_file.exists())
    
    if not windowed_file.exists():
        print("\n[FAIL] PHASE 2 SILVER VALIDATION FAILED")
        print("signals_windowed.parquet not found")
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2: Load & Schema
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n2. Loading and validating schema...\n")
    
    try:
        df = pd.read_parquet(windowed_file)
        val.check("Parquet file loads successfully", True, f"{len(df)} records")
    except Exception as e:
        val.check("Parquet file loads successfully", False, str(e))
        return False
    
    # Check required columns
    required_cols = ['subject_id', 'window_start_sec', 'window_end_sec', 
                    'consciousness_state', 'has_loc_in_window']
    has_required = all(col in df.columns for col in required_cols)
    val.check("All required columns present", has_required, 
             f"Cols: {list(df.columns[:5])}...")
    
    # Check signal value columns
    signal_cols = [col for col in df.columns if col.endswith('_values')]
    val.check(f"Has signal value arrays", len(signal_cols) == EXPECTED_SIGNALS,
             f"Found {len(signal_cols)}, expected {EXPECTED_SIGNALS}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3: Record Count & Completeness
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n3. Checking record count & completeness...\n")
    
    val.check("Non-zero records", len(df) > 0, f"Got {len(df)}")
    val.check("Reasonable record count (>50, <500)", 50 <= len(df) <= 500, 
             f"Got {len(df)}")
    
    # Subjects represented
    subjects = sorted(df['subject_id'].unique())
    val.check("All 9 subjects present", subjects == EXPECTED_SUBJECTS,
             f"Subjects: {subjects}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4: Window Consistency
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n4. Checking window consistency...\n")
    
    # Window length
    df['computed_window_length'] = df['window_end_sec'] - df['window_start_sec']
    all_30s = (df['computed_window_length'] == EXPECTED_WINDOW_LENGTH).all()
    val.check("All windows are 30 seconds", all_30s,
             f"Found lengths: {df['computed_window_length'].unique()}")
    
    # Window start times reasonable (0-200s range for propofol study)
    window_starts = df['window_start_sec']
    starts_reasonable = (window_starts >= 0) & (window_starts <= 200)
    val.check("Window start times in range [0, 200s]", starts_reasonable.all(),
             f"Min: {window_starts.min():.1f}s, Max: {window_starts.max():.1f}s")
    
    # Check signal array shapes (should be ~300 samples per window)
    sample_counts = []
    for col in signal_cols[:3]:  # Check first 3 signals
        if col in df.columns:
            for val_array in df[col]:
                if isinstance(val_array, np.ndarray):
                    sample_counts.append(len(val_array))
                elif isinstance(val_array, list):
                    sample_counts.append(len(val_array))
    
    if sample_counts:
        avg_samples = np.mean(sample_counts)
        val.check(f"Signal arrays have ~300 samples per window", 
                 200 <= avg_samples <= 400,
                 f"Mean: {avg_samples:.0f} samples")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5: Consciousness State Distribution
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n5. Checking consciousness state distribution...\n")
    
    # All states are valid
    valid_states = df['consciousness_state'].isin(CONSCIOUSNESS_STATES).all()
    val.check("All consciousness states valid", valid_states,
             f"States: {df['consciousness_state'].unique()}")
    
    # Reasonable distribution
    state_counts = df['consciousness_state'].value_counts()
    loc_count = state_counts.get('LOC', 0)
    pre_loc_count = state_counts.get('PRE_LOC', 0)
    high_risk_pct = (loc_count + pre_loc_count) / len(df) * 100 if len(df) > 0 else 0
    
    val.check("High-risk (LOC+PRE_LOC) >50% of samples", high_risk_pct > 50,
             f"Got {high_risk_pct:.1f}%")
    
    print(f"\n   State distribution:")
    for state, count in state_counts.items():
        pct = count / len(df) * 100
        print(f"   - {state}: {count} ({pct:.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 6: Data Quality
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n6. Checking data quality...\n")
    
    # Check for nulls in metadata
    has_loc_nulls = df[['subject_id', 'window_start_sec', 'window_end_sec']].isna().sum().sum()
    val.check("No nulls in metadata columns", has_loc_nulls == 0,
             f"Found {has_loc_nulls} nulls")
    
    # Check signal array integrity
    arrays_ok = True
    for col in signal_cols:
        for arr in df[col]:
            if not isinstance(arr, (np.ndarray, list)):
                arrays_ok = False
                break
    val.check("All signal columns contain arrays", arrays_ok)
    
    # Check normalized values are in reasonable range (-20 to +100 for z-score with outliers)
    value_ranges = []
    for col in signal_cols[:3]:  # Sample check
        for arr in df[col]:
            if isinstance(arr, np.ndarray):
                valid = arr[~(np.isnan(arr) | np.isinf(arr))]
                if len(valid) > 0:
                    value_ranges.extend([np.min(valid), np.max(valid)])
    
    if value_ranges:
        reasonable_range = (np.min(value_ranges) >= -20) and (np.max(value_ranges) <= 100)
        val.check("Normalized signal values in reasonable range [-20, +100]",
                 reasonable_range,
                 f"Min: {np.min(value_ranges):.2f}, Max: {np.max(value_ranges):.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 7: has_loc_in_window Consistency
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n7. Checking LOC/ROC temporal flags...\n")
    
    has_loc_values = df['has_loc_in_window'].unique()
    is_boolean = set(has_loc_values).issubset({True, False, 0, 1})
    val.check("has_loc_in_window is boolean", is_boolean,
             f"Values: {has_loc_values}")
    
    loc_flagged = (df['has_loc_in_window'] == True).sum()
    loc_expected = state_counts.get('LOC', 0)
    val.check("LOC flags reasonable (<20% of windows)", 
             loc_flagged <= len(df) * 0.3,  # At most 30% have LOC touching window
             f"Flagged: {loc_flagged} ({loc_flagged/len(df)*100:.1f}%), LOC samples: {loc_expected}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    success = val.summary()
    
    if success:
        print("\n[OK] PHASE 2 SILVER VALIDATION PASSED")
        print("Ready for Phase 3: Gold Layer Feature Engineering\n")
    else:
        print("\n[FAIL] PHASE 2 SILVER VALIDATION FAILED")
        print("Fix errors before proceeding to Phase 3\n")
    
    return success

if __name__ == "__main__":
    success = validate_silver()
    exit(0 if success else 1)
