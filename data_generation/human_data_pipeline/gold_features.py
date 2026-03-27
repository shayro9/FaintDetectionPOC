"""
================================================================================
PHASE 3: GOLD LAYER - FEATURE ENGINEERING & PREPARATION
================================================================================

Gold layer transforms windowed signals into machine-learning-ready features:
1. Flatten: Denest signal arrays into individual feature columns
2. Extract: Time-domain features (mean, std, min, max, energy, entropy)
3. Aggregate: Multi-signal features (correlations, ratios)
4. Validate: Check for NaN/inf and impute
5. Split: LOSO (Leave-One-Subject-Out) cross-validation splits
6. Export: Train/val/test sets ready for model training

Output: data/gold/features_loso_fold_*.parquet (one per fold)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../../data")
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
GOLD_DIR.mkdir(exist_ok=True)

WINDOW_LENGTH_SEC = 30
SAMPLE_RATE_HZ = 10
SAMPLES_PER_WINDOW = WINDOW_LENGTH_SEC * SAMPLE_RATE_HZ  # 300

# Target classes (for imbalance handling later)
TARGET_CLASSES = ['BASELINE', 'PRE_LOC', 'LOC', 'POST_ROC']

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
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

logger = Logger(GOLD_DIR / "gold_features.log")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_time_domain_features(signal_array):
    """Extract time-domain features from signal window."""
    # Handle NaN/inf
    valid = signal_array[~(np.isnan(signal_array) | np.isinf(signal_array))]
    
    if len(valid) == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'range': np.nan, 'energy': np.nan, 'entropy': np.nan,
            'rms': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
        }
    
    features = {
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
        'range': np.max(valid) - np.min(valid),
        'energy': np.sum(valid ** 2),
        'entropy': -np.sum(np.abs(valid) * np.log(np.abs(valid) + 1e-10)),
        'rms': np.sqrt(np.mean(valid ** 2)),
        'skewness': stats.skew(valid) if len(valid) > 2 else np.nan,
        'kurtosis': stats.kurtosis(valid) if len(valid) > 3 else np.nan,
    }
    
    return features

def compute_hrv_ratio_features(df_row):
    """Compute HRV ratio features (LF/HF, etc.)."""
    features = {}
    
    # LF/HF ratio (already in data as 'ratio')
    if 'ratio_mean' in df_row and pd.notna(df_row['ratio_mean']):
        features['lf_hf_ratio'] = df_row['ratio_mean']
    
    # Normalized LF/HF
    if 'LF_mean' in df_row and 'HF_mean' in df_row:
        lf = df_row.get('LF_mean', np.nan)
        hf = df_row.get('HF_mean', np.nan)
        if pd.notna(lf) and pd.notna(hf) and (lf + hf) > 0:
            features['lf_norm'] = lf / (lf + hf)
            features['hf_norm'] = hf / (lf + hf)
    
    return features

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD SILVER WINDOWED DATA
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Loading Silver layer windowed data...")
df_silver = pd.read_parquet(SILVER_DIR / "signals_windowed.parquet")
logger.log(f"Loaded {len(df_silver)} samples with {len(df_silver.columns)} columns\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FLATTEN SIGNAL ARRAYS INTO INDIVIDUAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 1: Flattening signal arrays and extracting time-domain features...")

# Find all signal columns (those ending with '_values')
signal_columns = [col for col in df_silver.columns if col.endswith('_values')]
logger.log(f"  Found {len(signal_columns)} signal columns")

# Extract time-domain features for each signal
for signal_col in signal_columns:
    signal_name = signal_col.replace('_values', '')
    
    # Extract features from each window
    features_list = []
    for idx, row in df_silver.iterrows():
        signal_array = row[signal_col]
        if isinstance(signal_array, np.ndarray):
            td_features = compute_time_domain_features(signal_array)
        else:
            td_features = compute_time_domain_features(np.array(signal_array))
        
        for feature_key, feature_val in td_features.items():
            features_list.append({
                'row_idx': idx,
                'signal': signal_name,
                'feature': feature_key,
                'value': feature_val,
            })
    
    # Pivot into wide format (one column per signal+feature)
    df_td_features = pd.DataFrame(features_list)
    df_td_pivot = df_td_features.pivot_table(
        index='row_idx',
        columns=['signal', 'feature'],
        values='value',
        aggfunc='first'
    )
    
    # Flatten multi-level column names
    df_td_pivot.columns = [f"{sig}_{feat}" for sig, feat in df_td_pivot.columns]
    
    if 'df_features' not in locals():
        df_features = df_silver[['subject_id', 'window_start_sec', 'window_end_sec', 
                                   'consciousness_state', 'has_loc_in_window']].copy()
    
    df_features = df_features.join(df_td_pivot, how='left')

logger.log(f"  OK - Created {len(df_features.columns) - 5} time-domain features\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: ADD DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 2: Computing derived features...")

# HRV ratios (if LF/HF available)
hrv_features_list = []
for idx, row in df_features.iterrows():
    hrv_feats = compute_hrv_ratio_features(row)
    hrv_feats['row_idx'] = idx
    hrv_features_list.append(hrv_feats)

if hrv_features_list:
    df_hrv = pd.DataFrame(hrv_features_list).set_index('row_idx')
    df_features = df_features.join(df_hrv, how='left')

logger.log(f"  OK - Added derived features\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 3: Handling missing values...")

# Count NaN before
nan_count_before = df_features.isna().sum().sum()
logger.log(f"  NaN values before imputation: {nan_count_before}")

# Forward fill per subject (temporal continuity within subject)
df_features = df_features.sort_values(['subject_id', 'window_start_sec']).reset_index(drop=True)
for subject_id in df_features['subject_id'].unique():
    mask = df_features['subject_id'] == subject_id
    feature_cols = [col for col in df_features.columns if col not in 
                   ['subject_id', 'window_start_sec', 'window_end_sec', 'consciousness_state', 'has_loc_in_window']]
    df_features.loc[mask, feature_cols] = df_features.loc[mask, feature_cols].ffill(limit=2)

# Backward fill for remaining NaN
for subject_id in df_features['subject_id'].unique():
    mask = df_features['subject_id'] == subject_id
    feature_cols = [col for col in df_features.columns if col not in 
                   ['subject_id', 'window_start_sec', 'window_end_sec', 'consciousness_state', 'has_loc_in_window']]
    df_features.loc[mask, feature_cols] = df_features.loc[mask, feature_cols].bfill(limit=2)

# Fill remaining with mean (per subject)
for col in df_features.columns:
    if df_features[col].dtype in [np.float64, np.float32]:
        df_features[col] = df_features.groupby('subject_id')[col].transform(
            lambda x: x.fillna(x.mean())
        )

# Fill any remaining NaN with 0
df_features = df_features.fillna(0)

nan_count_after = df_features.isna().sum().sum()
logger.log(f"  NaN values after imputation: {nan_count_after}")
logger.log(f"  OK - Handled missing values\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CREATE TARGET VARIABLE & ENCODE
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 4: Encoding target variable...")

# Map consciousness states to binary: PRE_LOC + LOC = high risk (1), else (0)
target_mapping = {
    'BASELINE': 0,
    'POST_ROC': 0,
    'PRE_LOC': 1,
    'LOC': 1,
}

df_features['high_risk'] = df_features['consciousness_state'].map(target_mapping)
df_features['target_class'] = df_features['consciousness_state']

logger.log(f"  Target class distribution:")
class_dist = df_features['target_class'].value_counts()
for cls, count in class_dist.items():
    logger.log(f"    - {cls}: {count} ({count/len(df_features)*100:.1f}%)")
logger.log(f"  High-risk (PRE_LOC + LOC): {df_features['high_risk'].sum()} ({df_features['high_risk'].mean()*100:.1f}%)\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CREATE LOSO FOLDS (LEAVE-ONE-SUBJECT-OUT)
# ─────────────────────────────────────────────────────────────────────────────

logger.log("Step 5: Creating LOSO (Leave-One-Subject-Out) folds...")

unique_subjects = sorted(df_features['subject_id'].unique())
num_folds = len(unique_subjects)

logger.log(f"  {num_folds} subjects -> {num_folds} LOSO folds\n")

loso_folds = {}

for fold_idx, test_subject in enumerate(unique_subjects):
    # Test: one subject out
    test_set = df_features[df_features['subject_id'] == test_subject].copy()
    
    # Train+Val: all other subjects
    train_val_set = df_features[df_features['subject_id'] != test_subject].copy()
    
    # Split train+val 80/20
    val_split_idx = int(len(train_val_set) * 0.8)
    train_set = train_val_set.iloc[:val_split_idx].copy()
    val_set = train_val_set.iloc[val_split_idx:].copy()
    
    loso_folds[fold_idx] = {
        'test_subject': test_subject,
        'train': train_set,
        'val': val_set,
        'test': test_set,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
    }
    
    logger.log(f"  Fold {fold_idx}: Test=S{test_subject} | Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: SAVE LOSO FOLDS & METADATA
# ─────────────────────────────────────────────────────────────────────────────

logger.log(f"\nStep 6: Saving LOSO folds...\n")

fold_metadata = {}

for fold_idx, fold_data in loso_folds.items():
    # Save train/val/test splits for this fold
    fold_dir = GOLD_DIR / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(exist_ok=True)
    
    # Identify feature columns (exclude metadata)
    metadata_cols = ['subject_id', 'window_start_sec', 'window_end_sec', 
                     'consciousness_state', 'has_loc_in_window', 'high_risk', 'target_class']
    feature_cols = [col for col in fold_data['train'].columns if col not in metadata_cols]
    
    # Save train/val/test with both features and metadata
    fold_data['train'].to_parquet(fold_dir / "train.parquet", index=False, compression='snappy')
    fold_data['val'].to_parquet(fold_dir / "val.parquet", index=False, compression='snappy')
    fold_data['test'].to_parquet(fold_dir / "test.parquet", index=False, compression='snappy')
    
    fold_metadata[fold_idx] = {
        'test_subject': int(fold_data['test_subject']),
        'train_size': fold_data['train_size'],
        'val_size': fold_data['val_size'],
        'test_size': fold_data['test_size'],
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
    }
    
    logger.log(f"  OK - Fold {fold_idx}: Train/val/test saved to fold_{fold_idx:02d}/")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE GLOBAL METADATA
# ─────────────────────────────────────────────────────────────────────────────

logger.log(f"\nStep 7: Saving metadata...\n")

metadata = {
    'pipeline': 'gold_features',
    'timestamp': datetime.now().isoformat(),
    'num_folds': num_folds,
    'total_samples': len(df_features),
    'window_length_sec': WINDOW_LENGTH_SEC,
    'sample_rate_hz': SAMPLE_RATE_HZ,
    'samples_per_window': SAMPLES_PER_WINDOW,
    'target_classes': TARGET_CLASSES,
    'class_distribution': class_dist.to_dict(),
    'high_risk_prevalence': float(df_features['high_risk'].mean()),
    'folds': fold_metadata,
}

with open(GOLD_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

logger.log(f"  OK - Metadata saved to metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

logger.log("\n" + "="*80)
logger.log("PHASE 3 COMPLETE")
logger.log("="*80 + "\n")

print("\nSTATS Gold Layer Summary:")
print(f"   * Total samples: {len(df_features)}")
print(f"   * Total features: {len(feature_cols)}")
print(f"   * LOSO folds: {num_folds}")
print(f"   * Subjects: {len(unique_subjects)}")
print(f"\n   * High-risk prevalence: {df_features['high_risk'].mean()*100:.1f}%")
print(f"   * Class balance:")
for cls, count in class_dist.items():
    pct = count / len(df_features) * 100
    print(f"     - {cls}: {count} ({pct:.1f}%)")

print(f"\nDIR Outputs saved to: {GOLD_DIR}/")
print(f"   * fold_00/ through fold_{num_folds-1:02d}/: Train/val/test splits")
print(f"   * metadata.json: Pipeline metadata & fold definitions")
print(f"   * gold_features.log: Detailed execution log")

print(f"\nOK Ready for Phase 4: Model Training")
logger.log(f"Ready for Phase 4: Model Training")
