# Data Pipeline Plan Summary: LOC Prediction

## 📋 Executive Summary

**Goal:** Build a data pipeline to train a model predicting **Loss of Consciousness (LOC)** from propofol-induced anesthesia data.

**Data Source:** 9 subjects, real physiological data from anesthesia experiments
- Multiple signals: EDA, HRV, HR metrics
- Ground truth: LOC and ROC (recovery) timestamps

**Pipeline Architecture:** Bronze → Silver → Gold (Medallion Lake House)

---

## 🎯 1. PREDICTION TASK

### What are we predicting?
```
INPUT:  60-second window of physiological signals
OUTPUT: Will patient lose consciousness in the next 30 seconds?
        (Binary classification: 0 = No, 1 = Yes)
```

### Why 60 seconds + 30-second warning?
- **60 seconds**: Enough data to observe physiological trends (HR decline, HRV collapse, EDA changes)
- **30 seconds**: Realistic warning lead time for user intervention/alert

### Class Distribution (Expected)
- **Class 0 (Normal)**: 70% — No LOC risk
- **Class 1 (Pre-LOC)**: 25% — LOC occurring within 30s
- **Excluded**: 5% — Post-LOC (already unconscious, no value to model)

### Cross-Validation Strategy
**Leave-One-Subject-Out (LOSO):**
- Train on 8 subjects, test on 1
- Rotate across all 9 subjects → 9-fold CV
- Prevents subject-specific overfitting
- Mimics deployment: model must work on new patients

---

## 🔌 2. SIGNAL SELECTION

### Tier 1: Key Signals (All Included)

#### **EDA (Electrodermal Activity)** — Autonomic indicator
```
✓ eda_tonic          — baseline skin conductance
✓ muPR, sigmaPR      — pulse rate: mean, std
✓ mu_amp, sigma_amp  — pulse amplitude: mean, std
Why: EDA collapses during LOC; sensitive to ANS changes
```

#### **HRV (Heart Rate Variability)** — Anesthesia depth indicator
```
✓ muRR, sigmaRR      — RR interval: mean, std
✓ muHR, sigmaHR      — HR: mean, std
✓ LF, HF             — Low freq, high freq power
✓ LFnu, HFnu         — Normalized LF/HF
✓ pow_tot            — Total HRV power
✓ ratio              — LF/HF ratio
Why: HRV is most predictive of anesthesia depth; markers LOC transition
```

### Total Signals: **18 signals × 7 statistics = 126 raw features**

---

## 🏗️ 3. DATA PIPELINE (Bronze → Silver → Gold)

### Layer 0: RAW CSV Files
```
S1_muHR.csv → S1_muRR.csv → ... → S9_sigmaHR.csv
S1_LOC.csv → S1_ROC.csv (ground truth labels)
S1_events.csv (propofol stage timestamps)
```

### Layer 1: BRONZE (Raw Ingest)
```
✓ Load all CSVs → Delta Lake table
✓ Zero transformation (append-only)
✓ Add metadata: _ingested_at, _source_file, _subject_id
✓ Idempotent: run 2x → same row count

Schema:
  subject_id, signal_name, timestamp, value, _ingested_at, _source_file
```

### Layer 2: SILVER (Cleanse & Conform)
```
✓ Deduplicate: keep latest per (subject, signal, timestamp)
✓ Validate: timestamp ∈ [0, 3600), HR ∈ [30, 200]
✓ Interpolate: fill <5s gaps (linear), flag >60s gaps (drop)
✓ Normalize: per-subject z-score (mean=0, std=1)
✓ Flag outliers: IQR method + domain knowledge

Schema:
  subject_id, signal_name, timestamp, value (normalized), value_raw,
  is_interpolated, is_outlier, interpolation_gap_seconds
```

### Layer 3: GOLD (Feature Engineering & Ready for ML)
```
STEP 1: Create sliding windows
  60-second windows, 1-second stride
  Example: [0-60s], [1-61s], [2-62s], ..., [N-60s]

STEP 2: Aggregate features per window
  Raw features (126):     mean, std, min, max, median, p25, p75 per signal
  Derived features (20):  HR trend, HF trend, LF/HF ratio, stage context

STEP 3: Encode labels
  For each window:
    time_to_loc = LOC_timestamp - window_start
    label = 1 if time_to_loc ≤ 30s, else 0
    exclude if time_to_loc < 0 (post-LOC)

STEP 4: Create splits
  Per subject: temporal split 60/20/20 (train/val/test)
  No subject appears in multiple folds (LOSO)

Schema:
  subject_id, window_id, label, split_set,
  [126 raw features], [20 derived features]
  
Total: ~90,000 windows across 9 subjects
```

---

## 📊 4. FEATURE ENGINEERING (146 Total)

### Raw Features: 126
```
7 statistics × 18 signals:
  - mean, std, min, max, median, 25th percentile, 75th percentile

Applied to: muHR, sigmaHR, muRR, sigmaRR, LF, HF, LFnu, HFnu, pow_tot, ratio,
            eda_tonic, muPR, sigmaPR, mu_amp, sigma_amp, + 3 more
```

### Derived Features: 20
```
Physiological Markers:
  - HR_trend: (HR_end - HR_start) / window_duration
  - HF_trend: (HF_end - HF_start) / window_duration
  - LF_HF_ratio: LF / HF (autonomic balance)
  - LF_HF_trend: (LF/HF_end - LF/HF_start) / window_duration
  
Context:
  - propofol_stage: current anesthesia stage (1-10)
  - time_since_stage_onset: seconds into current stage
  
Total: 20 derived features
```

---

## ✅ 5. DATA QUALITY CHECKPOINTS

### Bronze Layer
- ✓ Row counts logged per signal
- ✓ Timestamp continuity verified
- ✓ Duplicate detection (alert if same subject+signal+timestamp appears 2x)

### Silver Layer
- ✓ Null rate < 5% per signal
- ✓ Outlier rate < 2%
- ✓ Interpolation coverage < 10%
- ✓ Normalization verified: mean ≈ 0, std ≈ 1 per signal per subject

### Gold Layer
- ✓ No NaN values in 146 features
- ✓ Label distribution: 70% class 0, 25% class 1, 5% excluded (±5%)
- ✓ Window counts per subject logged
- ✓ LOSO split validation: no subject leakage
- ✓ Feature statistics: min/max/mean per feature (detect skew)

---

## 📅 6. IMPLEMENTATION TIMELINE

| Phase | Tasks | Days | Duration | Output |
|-------|-------|------|----------|--------|
| **0: EDA** | Profile signals, visualize | Day 0 | 4h | eda_report.md + plots |
| **1: Bronze** | Load raw data, validate | Day 1 | 6h | Bronze Delta table + load script |
| **2: Silver** | Dedup, normalize, flag QA | Day 2-3 | 8h | Silver Delta table + transform script |
| **3: Gold** | Windows, features, labels, splits | Day 3-4 | 8h | X_windowed.npz, y_labels, splits.json |
| **4: Validation** | Config, docs, E2E test, version | Day 4 | 4h | README, config.yaml, v1.0 tag |
| **TOTAL** | | **4 days** | **30h** | **Ready for Model Training** |

---

## 📦 7. DELIVERABLES (Ready for Phase 2: Model Training)

### Data Files
```
data/gold/
├── X_windowed.npz              ← Feature matrix (90k × 146)
├── y_labels.npz                ← Labels (90k,)
├── splits.json                 ← Train/val/test indices + subject mapping
├── feature_engineering_params.json  ← Normalization params, feature metadata
└── summary_statistics.json      ← Label distribution, feature ranges
```

### Code & Config
```
data_pipeline/
├── load_raw_data.py            ← Bronze ingest script
├── silver_transform.py         ← Cleanse & normalize
├── gold_feature_engineering.py ← Windows, features, labels
├── data_pipeline_config.yaml   ← All parameters (reproducible)
└── run_pipeline.sh            ← End-to-end execution
```

### Documentation
```
README_DATA_PIPELINE.md         ← Architecture, feature definitions, QA checks
data_pipeline_plan.md           ← Detailed design (this file's source)
eda_report.md                   ← Exploratory analysis + findings
```

---

## 🚨 8. KEY RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Small N=9 subjects** | High variance | Conservative CV strategy; confidence intervals; augment if needed |
| **Data quality issues** | Data leakage, poor model | EDA first; quality gates at each layer; robust preprocessing |
| **Class imbalance (70/25)** | Model bias | Class weights during training; stratified sampling; monitor F1 separately |
| **Temporal leakage** | Invalid model | Enforce temporal splits; validate no future signals in features |
| **Feature explosion** | Overfitting | Use feature importance; start with top 30; add incrementally |

---

## 🎓 9. DECISIONS & RATIONALE

Why these choices?

```
Window Size (60s)           → Captures physiological trends + short warning lead
Warning Lead Time (30s)     → Realistic intervention window
Window Stride (1s)          → High coverage of LOC boundary; ~90k windows manageable
Features (146)              → 126 raw (distribution statistics) + 20 derived (physiological)
Normalization (per-subject) → Accounts for individual baselines
LOSO CV                     → Only 9 subjects; prevents subject-specific overfitting
Temporal Split (60/20/20)   → Model sees full anesthesia progression
```

---

## 🔗 NEXT STEPS

1. **Review this plan** — Any changes to feature selection, window size, or splits?
2. **Start Phase 0 (EDA)** — Download data, profile signals, generate visualizations
3. **Execute Phases 1-4** — Follow the 4-day timeline; track todos in SQL
4. **Deliver Gold Layer** — Hand off `X_windowed.npz` + metadata to ML Team (Phase 2)

---

## 📄 LINKED DOCUMENTS

- **Full Plan**: `data_pipeline_plan.md` (detailed architecture, code patterns, reproducibility)
- **Sprint Plan**: `plan.md` (overall 1-week MVP timeline, all 4 phases)
- **SQL Todos**: Track 14 data pipeline tasks with dependencies

---

**Data Engineer**: Ready to build this pipeline. Shall we start with Phase 0 (EDA)?
