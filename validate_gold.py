"""
================================================================================
PHASE 3 GOLD LAYER VALIDATION
================================================================================

Validates the Gold layer output (LOSO folds) before model training.

Checks:
1. Fold structure (9 folds, each with train/val/test splits)
2. Sample counts per fold
3. Feature consistency across folds
4. LOSO property (no subject leakage)
5. Class distribution
6. Metadata consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
GOLD_DIR = DATA_DIR / "gold"

EXPECTED_NUM_FOLDS = 9
EXPECTED_SUBJECTS = list(range(1, 10))
EXPECTED_TOTAL_SAMPLES = 143
EXPECTED_FEATURES = 153

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
        if details:
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

def validate_gold():
    """Execute Gold layer validation."""
    
    print("\n" + "="*80)
    print("PHASE 3 GOLD LAYER VALIDATION")
    print("="*80 + "\n")
    
    val = Validator()
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 1: Fold Structure
    # ─────────────────────────────────────────────────────────────────────────
    
    print("1. Checking fold structure...\n")
    
    fold_dirs = sorted([d for d in GOLD_DIR.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    val.check("9 LOSO folds exist", len(fold_dirs) == EXPECTED_NUM_FOLDS, 
             f"Found {len(fold_dirs)}")
    
    # Check each fold has required files
    for fold_dir in fold_dirs:
        has_train = (fold_dir / "train.parquet").exists()
        has_val = (fold_dir / "val.parquet").exists()
        has_test = (fold_dir / "test.parquet").exists()
        
        fold_num = fold_dir.name
        val.check(f"{fold_num} has train/val/test", 
                 has_train and has_val and has_test,
                 f"Train:{has_train}, Val:{has_val}, Test:{has_test}")
    
    # Check metadata exists
    metadata_file = GOLD_DIR / "metadata.json"
    val.check("metadata.json exists", metadata_file.exists())
    
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            val.check("metadata.json is valid JSON", True)
            val.check("metadata has num_folds", "num_folds" in metadata, 
                     f"Keys: {list(metadata.keys())[:5]}")
        except Exception as e:
            val.check("metadata.json is valid JSON", False, str(e))
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 2: Sample Counts
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n2. Checking sample counts per fold...\n")
    
    fold_stats = []
    total_samples_across_folds = 0
    
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            train_df = pd.read_parquet(fold_dir / "train.parquet")
            val_df = pd.read_parquet(fold_dir / "val.parquet")
            test_df = pd.read_parquet(fold_dir / "test.parquet")
            
            n_train = len(train_df)
            n_val = len(val_df)
            n_test = len(test_df)
            total_fold = n_train + n_val + n_test
            
            fold_stats.append({
                'fold': fold_idx,
                'train': n_train,
                'val': n_val,
                'test': n_test,
                'total': total_fold,
            })
            
            total_samples_across_folds += total_fold
            
            # Check fold sums to expected total (approximately - might vary due to splits)
            val.check(f"Fold {fold_idx}: total ~143", 
                     130 <= total_fold <= 155,  # Allow variance
                     f"Train={n_train}, Val={n_val}, Test={n_test}, Total={total_fold}")
            
        except Exception as e:
            val.check(f"Fold {fold_idx} loads successfully", False, str(e))
    
    # Check total samples across all folds matches expectation
    val.check(f"Total samples across folds ~{EXPECTED_TOTAL_SAMPLES * EXPECTED_NUM_FOLDS}",
             abs(total_samples_across_folds - EXPECTED_TOTAL_SAMPLES * EXPECTED_NUM_FOLDS) < 100,
             f"Got {total_samples_across_folds}")
    
    print("\n   Sample distribution per fold:")
    for stat in fold_stats:
        print(f"   Fold {stat['fold']}: Train={stat['train']:3d}, Val={stat['val']:2d}, Test={stat['test']:2d}, Total={stat['total']:3d}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 3: Feature Consistency
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n3. Checking feature consistency...\n")
    
    all_features = None
    feature_consistency_ok = True
    
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            train_df = pd.read_parquet(fold_dir / "train.parquet")
            val_df = pd.read_parquet(fold_dir / "val.parquet")
            test_df = pd.read_parquet(fold_dir / "test.parquet")
            
            # Get feature columns (exclude metadata)
            metadata_cols = ['subject_id', 'window_start_sec', 'window_end_sec', 
                           'consciousness_state', 'has_loc_in_window', 'high_risk', 'target_class']
            features = [col for col in train_df.columns if col not in metadata_cols]
            
            if all_features is None:
                all_features = set(features)
            elif set(features) != all_features:
                feature_consistency_ok = False
            
            # Check schema matches across train/val/test
            val_features = [col for col in val_df.columns if col not in metadata_cols]
            test_features = [col for col in test_df.columns if col not in metadata_cols]
            
            schema_match = (set(features) == set(val_features) == set(test_features))
            val.check(f"Fold {fold_idx}: train/val/test have same features", 
                     schema_match,
                     f"Train={len(features)}, Val={len(val_features)}, Test={len(test_features)}")
            
        except Exception as e:
            val.check(f"Fold {fold_idx} feature check", False, str(e))
    
    val.check(f"All folds have consistent features", feature_consistency_ok)
    
    if all_features:
        val.check(f"Feature count matches plan ({EXPECTED_FEATURES})", 
                 len(all_features) == EXPECTED_FEATURES,
                 f"Got {len(all_features)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 4: LOSO Property (No Subject Leakage)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n4. Checking LOSO property (no subject leakage)...\n")
    
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            train_df = pd.read_parquet(fold_dir / "train.parquet")
            val_df = pd.read_parquet(fold_dir / "val.parquet")
            test_df = pd.read_parquet(fold_dir / "test.parquet")
            
            train_subjects = set(train_df['subject_id'].unique())
            val_subjects = set(val_df['subject_id'].unique())
            test_subjects = set(test_df['subject_id'].unique())
            
            # Test subject should be different from train+val
            test_subject = list(test_subjects)[0] if len(test_subjects) == 1 else None
            
            # Train and val should not contain test subject
            no_leakage = len(train_subjects & test_subjects) == 0 and len(val_subjects & test_subjects) == 0
            
            val.check(f"Fold {fold_idx} no subject leakage", no_leakage,
                     f"Train subj: {sorted(train_subjects)}, Val subj: {sorted(val_subjects)}, Test subj: {sorted(test_subjects)}")
            
        except Exception as e:
            val.check(f"Fold {fold_idx} LOSO check", False, str(e))
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 5: Class Distribution
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n5. Checking class distribution...\n")
    
    all_class_counts = {}
    
    for fold_idx, fold_dir in enumerate(fold_dirs):
        try:
            train_df = pd.read_parquet(fold_dir / "train.parquet")
            test_df = pd.read_parquet(fold_dir / "test.parquet")
            
            # Check high_risk column exists
            has_high_risk = 'high_risk' in train_df.columns
            val.check(f"Fold {fold_idx}: has high_risk column", has_high_risk)
            
            # Check target_class column exists
            has_target_class = 'target_class' in train_df.columns
            val.check(f"Fold {fold_idx}: has target_class column", has_target_class)
            
            if has_high_risk:
                high_risk_pct = train_df['high_risk'].mean() * 100
                val.check(f"Fold {fold_idx}: high-risk prevalence >70%", 
                         high_risk_pct > 70,
                         f"Got {high_risk_pct:.1f}%")
            
            if has_target_class:
                class_dist = train_df['target_class'].value_counts()
                all_class_counts[f"fold_{fold_idx}"] = class_dist.to_dict()
        
        except Exception as e:
            val.check(f"Fold {fold_idx} class distribution check", False, str(e))
    
    print("\n   Class distribution (sample from fold 0):")
    if all_class_counts and 'fold_0' in all_class_counts:
        for cls, count in all_class_counts['fold_0'].items():
            print(f"   - {cls}: {count}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHECK 6: Metadata Completeness
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n6. Checking metadata completeness...\n")
    
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Check required keys
            required_keys = ['num_folds', 'total_samples', 'class_distribution', 'folds']
            for key in required_keys:
                val.check(f"Metadata has '{key}'", key in metadata)
            
            # Check folds metadata
            if 'folds' in metadata:
                folds_meta = metadata['folds']
                val.check(f"Metadata folds count matches", 
                         len(folds_meta) == EXPECTED_NUM_FOLDS,
                         f"Got {len(folds_meta)}")
                
                # Sample fold metadata
                if '0' in folds_meta or 0 in folds_meta:
                    fold_key = '0' if '0' in folds_meta else 0
                    sample_fold_meta = folds_meta[fold_key]
                    has_test_subject = 'test_subject' in sample_fold_meta
                    has_num_features = 'num_features' in sample_fold_meta
                    val.check(f"Sample fold metadata complete", 
                             has_test_subject and has_num_features)
        
        except Exception as e:
            val.check(f"Metadata completeness check", False, str(e))
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    success = val.summary()
    
    print("\n" + "="*80)
    print("FOLD SUMMARY TABLE")
    print("="*80)
    df_fold_stats = pd.DataFrame(fold_stats)
    print(df_fold_stats.to_string(index=False))
    print("="*80)
    
    if success:
        print("\n[OK] PHASE 3 GOLD LAYER VALIDATION PASSED")
        print("Gold layer is ready for Phase 4: Model Training\n")
    else:
        print("\n[FAIL] PHASE 3 GOLD LAYER VALIDATION FAILED")
        print("Fix errors before proceeding to Phase 4\n")
    
    return success

if __name__ == "__main__":
    success = validate_gold()
    exit(0 if success else 1)
