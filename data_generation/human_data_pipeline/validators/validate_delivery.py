"""
================================================================================
FINAL GOLD LAYER DELIVERY VALIDATION & PACKAGING
================================================================================

Comprehensive pre-PR validation ensuring Gold layer is 100% production-ready.
Creates delivery checklist and data contract documentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../../../data")
GOLD_DIR = DATA_DIR / "gold"

DELIVERY_CHECKLIST = {
    "Data Integrity": [],
    "Schema Consistency": [],
    "No Data Leakage": [],
    "Documentation": [],
    "Reproducibility": [],
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_delivery():
    """Run comprehensive delivery validation."""
    
    print("\n" + "="*80)
    print("FINAL DELIVERY VALIDATION - GOLD LAYER")
    print("="*80 + "\n")
    
    all_pass = True
    
    # CHECK 1: Data Integrity
    print("1. DATA INTEGRITY CHECKS")
    print("-" * 80)
    
    fold_dirs = sorted([d for d in GOLD_DIR.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    # No missing folds
    check_1a = len(fold_dirs) == 9
    print(f"  [{'PASS' if check_1a else 'FAIL'}] All 9 folds present ({len(fold_dirs)}/9)")
    DELIVERY_CHECKLIST["Data Integrity"].append(("All 9 folds present", check_1a))
    
    # No null/NaN in critical columns
    null_check_pass = True
    for fold_idx, fold_dir in enumerate(fold_dirs):
        for split in ['train', 'val', 'test']:
            df = pd.read_parquet(fold_dir / f"{split}.parquet")
            critical_cols = ['subject_id', 'window_start_sec', 'window_end_sec', 'high_risk']
            nulls = df[critical_cols].isna().sum().sum()
            if nulls > 0:
                null_check_pass = False
                print(f"  [FAIL] Fold {fold_idx} {split}: {nulls} nulls in critical columns")
    
    if null_check_pass:
        print(f"  [PASS] No nulls in critical columns (all folds, all splits)")
    DELIVERY_CHECKLIST["Data Integrity"].append(("No nulls in critical columns", null_check_pass))
    all_pass = all_pass and null_check_pass
    
    # Feature arrays not corrupted
    arrays_ok = True
    for fold_dir in fold_dirs[:1]:  # Sample check
        df = pd.read_parquet(fold_dir / "train.parquet")
        for col in df.columns:
            if col.endswith('_values'):
                for val in df[col]:
                    if not isinstance(val, np.ndarray):
                        arrays_ok = False
                        break
    print(f"  [{'PASS' if arrays_ok else 'FAIL'}] Signal arrays properly serialized")
    DELIVERY_CHECKLIST["Data Integrity"].append(("Signal arrays properly serialized", arrays_ok))
    all_pass = all_pass and arrays_ok
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2: Schema Consistency
    print("\n2. SCHEMA CONSISTENCY CHECKS")
    print("-" * 80)
    
    # All folds have same schema
    schemas = {}
    schema_consistent = True
    for fold_idx, fold_dir in enumerate(fold_dirs):
        df_train = pd.read_parquet(fold_dir / "train.parquet")
        cols = set(df_train.columns)
        if fold_idx > 0 and cols != schemas['fold_0']:
            schema_consistent = False
        schemas[f'fold_{fold_idx}'] = cols
    
    print(f"  [{'PASS' if schema_consistent else 'FAIL'}] All folds have identical schema")
    DELIVERY_CHECKLIST["Schema Consistency"].append(("All folds have identical schema", schema_consistent))
    all_pass = all_pass and schema_consistent
    
    # Feature count matches plan
    sample_df = pd.read_parquet(fold_dirs[0] / "train.parquet")
    metadata_cols = ['subject_id', 'window_start_sec', 'window_end_sec', 
                    'consciousness_state', 'has_loc_in_window', 'high_risk', 'target_class']
    num_features = len([c for c in sample_df.columns if c not in metadata_cols])
    features_ok = num_features == 153
    print(f"  [{'PASS' if features_ok else 'FAIL'}] Feature count = 153 (got {num_features})")
    DELIVERY_CHECKLIST["Schema Consistency"].append(("Feature count = 153", features_ok))
    all_pass = all_pass and features_ok
    
    # Target variable properly encoded
    target_ok = True
    for fold_dir in fold_dirs:
        df = pd.read_parquet(fold_dir / "train.parquet")
        if 'high_risk' in df.columns:
            unique_vals = set(df['high_risk'].unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                target_ok = False
    print(f"  [{'PASS' if target_ok else 'FAIL'}] Target variable (high_risk) properly encoded (0/1)")
    DELIVERY_CHECKLIST["Schema Consistency"].append(("Target variable properly encoded", target_ok))
    all_pass = all_pass and target_ok
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3: No Data Leakage
    print("\n3. NO DATA LEAKAGE CHECKS (LOSO PROPERTY)")
    print("-" * 80)
    
    leakage_ok = True
    for fold_idx, fold_dir in enumerate(fold_dirs):
        df_train = pd.read_parquet(fold_dir / "train.parquet")
        df_val = pd.read_parquet(fold_dir / "val.parquet")
        df_test = pd.read_parquet(fold_dir / "test.parquet")
        
        train_subj = set(df_train['subject_id'].unique())
        val_subj = set(df_val['subject_id'].unique())
        test_subj = set(df_test['subject_id'].unique())
        
        # No overlap between train+val and test
        if len(train_subj & test_subj) > 0 or len(val_subj & test_subj) > 0:
            leakage_ok = False
            print(f"  [FAIL] Fold {fold_idx}: subject leakage detected")
    
    if leakage_ok:
        print(f"  [PASS] No subject leakage (LOSO property verified)")
    DELIVERY_CHECKLIST["No Data Leakage"].append(("No subject leakage (LOSO)", leakage_ok))
    all_pass = all_pass and leakage_ok
    
    # Balanced folds
    balanced_ok = True
    for fold_idx, fold_dir in enumerate(fold_dirs):
        df_train = pd.read_parquet(fold_dir / "train.parquet")
        high_risk_pct = df_train['high_risk'].mean() * 100
        if high_risk_pct < 70:
            balanced_ok = False
    print(f"  [{'PASS' if balanced_ok else 'FAIL'}] All folds have >70% high-risk samples")
    DELIVERY_CHECKLIST["No Data Leakage"].append(("All folds >70% high-risk", balanced_ok))
    all_pass = all_pass and balanced_ok
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4: Documentation
    print("\n4. DOCUMENTATION CHECKS")
    print("-" * 80)
    
    metadata_exists = (GOLD_DIR / "metadata.json").exists()
    print(f"  [{'PASS' if metadata_exists else 'FAIL'}] metadata.json exists")
    DELIVERY_CHECKLIST["Documentation"].append(("metadata.json exists", metadata_exists))
    all_pass = all_pass and metadata_exists
    
    log_exists = (GOLD_DIR / "gold_features.log").exists()
    print(f"  [{'PASS' if log_exists else 'FAIL'}] gold_features.log exists (execution trace)")
    DELIVERY_CHECKLIST["Documentation"].append(("Execution log exists", log_exists))
    all_pass = all_pass and log_exists
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5: Reproducibility
    print("\n5. REPRODUCIBILITY CHECKS")
    print("-" * 80)
    
    if metadata_exists:
        with open(GOLD_DIR / "metadata.json") as f:
            meta = json.load(f)
        
        timestamp_ok = 'timestamp' in meta
        print(f"  [{'PASS' if timestamp_ok else 'FAIL'}] Pipeline execution timestamp recorded")
        DELIVERY_CHECKLIST["Reproducibility"].append(("Execution timestamp recorded", timestamp_ok))
        
        params_ok = 'window_length_sec' in meta and 'sample_rate_hz' in meta
        print(f"  [{'PASS' if params_ok else 'FAIL'}] Pipeline parameters documented")
        DELIVERY_CHECKLIST["Reproducibility"].append(("Pipeline parameters documented", params_ok))
        
        all_pass = all_pass and timestamp_ok and params_ok
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    print("\n" + "="*80)
    print("DELIVERY CHECKLIST SUMMARY")
    print("="*80)
    
    total_checks = 0
    total_pass = 0
    
    for category, checks in DELIVERY_CHECKLIST.items():
        cat_pass = sum(1 for _, result in checks if result)
        cat_total = len(checks)
        total_pass += cat_pass
        total_checks += cat_total
        status = "PASS" if cat_pass == cat_total else "FAIL"
        print(f"\n{category}: {cat_pass}/{cat_total} [{status}]")
        for check_name, result in checks:
            check_status = "OK" if result else "FAIL"
            print(f"  [{check_status}] {check_name}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {total_pass}/{total_checks} CHECKS PASS")
    print("="*80)
    
    if all_pass:
        print("\n✓ GOLD DATA IS 100% PRODUCTION-READY FOR ML TEAM")
        print("\nDelivery Contents:")
        print("  • data/gold/fold_00/ through fold_08/")
        print("    ├── train.parquet (model training)")
        print("    ├── val.parquet (hyperparameter tuning)")
        print("    └── test.parquet (final evaluation)")
        print("  • data/gold/metadata.json (data contract)")
        print("  • data/gold/gold_features.log (execution trace)")
        print("\nNext Step: Create DATA_DELIVERY.md for ML team")
    else:
        print("\n✗ GOLD DATA HAS ISSUES - DO NOT DELIVER")
    
    return all_pass

if __name__ == "__main__":
    success = validate_delivery()
    exit(0 if success else 1)
