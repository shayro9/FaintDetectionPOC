# Data Pipeline Visual Reference

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Real Propofol Data (9 Subjects)            │
│  S1_muHR.csv, S1_muRR.csv, ..., S9_sigmaHR.csv | LOC/ROC timestamps  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   PHASE 0: EDA      │  (4 hours)
                    │  - Profile signals  │
                    │  - Visualize        │
                    │  - Plan features    │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  BRONZE LAYER        │                  │  DATA QUALITY PLAN   │
│  (Raw Ingest)        │                  │  - Null thresholds   │
│  6 hours             │                  │  - Outlier detection │
│                      │                  │  - Schema validation │
│ ✓ Load CSVs          │                  │  - Interpolation     │
│ ✓ Zero transforms    │                  └──────────────────────┘
│ ✓ Append-only        │
│ ✓ Add metadata       │
│                      │
│ Output:              │
│  raw_timeseries      │
│  schema: subject_id, │
│  signal_name,        │
│  timestamp, value    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  SILVER LAYER        │
│  (Cleanse)           │
│  8 hours             │
│                      │
│ ✓ Dedup              │
│ ✓ Validate           │
│ ✓ Interpolate gaps   │
│ ✓ Normalize          │
│   (z-score)          │
│ ✓ Flag outliers      │
│                      │
│ Output:              │
│  signals (normalized)│
│  is_interpolated:    │
│  is_outlier boolean  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GOLD LAYER          │
│  (Features)          │
│  8 hours             │
│                      │
│ STEP 1: Windows      │
│  60s window,         │
│  stride=1s           │
│  ~90k windows        │
│                      │
│ STEP 2: Features     │
│  126 raw features:   │
│  (7 stats ×          │
│   18 signals)        │
│  20 derived:         │
│  (trends, ratios,    │
│   context)           │
│  Total: 146 features │
│                      │
│ STEP 3: Labels       │
│  time_to_LOC =       │
│  LOC_ts - win_start  │
│                      │
│  if ≤30s → label=1   │
│  if >30s → label=0   │
│  if <0 → excluded    │
│                      │
│  Distribution:       │
│  70% class 0         │
│  25% class 1         │
│  5% excluded         │
│                      │
│ STEP 4: Splits       │
│  Temporal: 60/20/20  │
│  LOSO CV: per subj   │
│  No leakage          │
│                      │
│ Output:              │
│  X (N × 146)         │
│  y (N,) binary       │
│  splits.json (LOSO)  │
│  normalization_      │
│  params.json         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  VALIDATION          │
│  & RELEASE           │
│  4 hours             │
│                      │
│ ✓ Config file        │
│ ✓ Docs (README)      │
│ ✓ E2E test           │
│ ✓ Git version        │
│                      │
│ Output:              │
│  v_data_2026_03_25   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│  READY FOR ML TEAM               │
│  (Phase 2: Model Training)       │
│                                  │
│ Files:                           │
│  • X_windowed.npz                │
│  • y_labels.npz                  │
│  • splits.json (LOSO indices)    │
│  • feature_engineering_          │
│    params.json                   │
│  • summary_statistics.json       │
│                                  │
│ Metadata:                        │
│  • Feature names (146)           │
│  • Normalization params          │
│  • Label distribution            │
│  • Per-subject split mapping     │
│                                  │
│ Total: ~90k samples × 146 feat   │
└──────────────────────────────────┘
```

---

## Feature Engineering Flow

```
┌─ Signal Processing ─────────────────────────────────────────┐
│                                                              │
│  18 Raw Signals                    Per 60-second Window:     │
│  ├─ muHR, sigmaHR                  │                        │
│  ├─ muRR, sigmaRR                  ├─ mean(signal)          │
│  ├─ LF, HF, LFnu, HFnu             ├─ std(signal)           │
│  ├─ pow_tot, ratio                 ├─ min(signal)           │
│  ├─ eda_tonic                      ├─ max(signal)           │
│  ├─ muPR, sigmaPR                  ├─ median(signal)        │
│  ├─ mu_amp, sigma_amp              ├─ p25(signal)           │
│  └─ [stage_context]                └─ p75(signal)           │
│                                                              │
│  7 statistics × 18 signals = 126 RAW FEATURES              │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─ Domain Features ───┴────────────────────────────────────────┐
│                                                              │
│  Physiological Markers (Autonomic + HRV)                     │
│  ├─ HR_trend: (HR_end - HR_start) / window_duration          │
│  ├─ HF_trend: (HF_end - HF_start) / window_duration         │
│  ├─ LF_HF_ratio: LF / HF (autonomic balance)                │
│  ├─ LF_HF_trend: (LF/HF_end - LF/HF_start) / duration       │
│  ├─ EDA_tonic_change: (EDA_end - EDA_start)                 │
│  │                                                           │
│  Context Features                                           │
│  ├─ propofol_stage: current stage (1-10)                    │
│  ├─ time_since_stage_onset: seconds into stage              │
│  ├─ experiment_time_pct: % of total experiment elapsed      │
│  └─ [5 more exploratory features]                           │
│                                                              │
│  20 DERIVED FEATURES                                        │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────┐
        │  TOTAL: 146 FEATURES    │
        │  ├─ 126 Raw (stats)     │
        │  └─  20 Derived (domain)│
        └─────────────────────────┘
```

---

## Data Quality Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    QUALITY GATES                              │
└──────────────────────────────────────────────────────────────┘

BRONZE VALIDATION:
  ├─ Row counts logged per signal
  ├─ Timestamp ranges [0, 3600]
  ├─ Duplicate detection
  └─ Type inference verified
       │
       ▼ (Accept/Reject: Continue if <1% errors)

SILVER QUALITY CHECKS:
  ├─ Null rate per signal < 5%
  │  └─ If >5%: flag, analyze, decide
  ├─ Outlier rate < 2%
  │  └─ If >2%: review domain bounds
  ├─ Interpolation coverage < 10%
  │  └─ If >10%: investigate gap patterns
  ├─ Normalization: mean ≈ 0, std ≈ 1
  │  └─ Verify per signal per subject
  └─ Flag quality_score per signal
       │
       ▼ (Accept/Investigate: All checks must pass)

GOLD FEATURE VALIDATION:
  ├─ No NaN values in 146 features
  │  └─ Alert: feature missing → block
  ├─ Label distribution:
  │  │  Class 0: 70% ± 5%
  │  │  Class 1: 25% ± 5%
  │  │  Excluded: 5% ± 3%
  │  └─ Alert if skewed
  ├─ Feature statistics:
  │  │  Min/Max/Mean per feature
  │  │  Detect outliers/skew
  │  └─ Flag unusual distributions
  ├─ LOSO validation:
  │  │  No subject in multiple folds
  │  │  No temporal leakage
  │  └─ Verify splits.json
  └─ Final row count: ~90,000 windows
       │
       ▼ (PASS: Ready for model training)
```

---

## Timeline (4 Days, 30 Hours)

```
DAY 0 (4h):    EDA
  ├─ Download CSVs
  ├─ Profile signals
  ├─ Visualize
  └─ eda_report.md ✓

DAY 1 (6h):    BRONZE
  ├─ Load raw data
  ├─ Test idempotency
  ├─ Validate schema
  └─ raw_timeseries table ✓

DAY 2-3 (8h):  SILVER
  ├─ Dedup, validate, interpolate
  ├─ Normalize (z-score per subject)
  ├─ Flag outliers
  ├─ Quality checks
  └─ signals table ✓

DAY 3-4 (8h):  GOLD
  ├─ Create windows (60s, stride=1s)
  ├─ Compute 146 features
  ├─ Encode labels (pre-LOC)
  ├─ Create splits (temporal 60/20/20)
  ├─ LOSO CV mapping
  └─ X/y/splits/metadata ✓

DAY 4 (4h):    RELEASE
  ├─ Config file (YAML)
  ├─ README + docs
  ├─ E2E test (Bronze→Silver→Gold)
  ├─ Git version tag
  └─ Ready for Phase 2 ✓

                TOTAL: 30 HOURS → GO TRAIN MODEL
```

---

## Success Checklist

```
DATA QUALITY:
  ✅ Zero data leakage (LOSO CV)
  ✅ Zero temporal leakage (temporal splits)
  ✅ Null rate < 5%
  ✅ Outlier rate < 2%
  ✅ NaN in features = 0
  ✅ Label distribution 70/25/5 (±5%)

REPRODUCIBILITY:
  ✅ Config file captures all parameters
  ✅ Pipeline idempotent (run 2x → same output)
  ✅ Code version tagged
  ✅ All scripts in Git

HANDOFF:
  ✅ X_windowed.npz loads without errors
  ✅ Feature names, subject IDs, split info clear
  ✅ Normalization params stored (for inference)
  ✅ README documents schema + data contract

READY FOR ML TEAM → Phase 2: Model Training
```

---

## Risk Mitigation

```
RISK                          MITIGATION
────────────────────────────────────────────────────────────
Small N=9 subjects            → LOSO CV, confidence intervals
Data quality issues           → EDA first, quality gates per layer
Class imbalance 70/25         → Class weights during training
Temporal leakage              → Strict temporal split enforcement
Feature explosion (146→?)     → Tree feature importance + pruning
Signal gaps/outliers          → Interpolation strategy + flagging
Propofol stage identification → Manual inspection of event timestamps
```

---

## Files Generated

```
data_pipeline/
├── load_raw_data.py              ← Bronze ingestion
├── silver_transform.py           ← Cleanse & normalize
├── gold_feature_engineering.py   ← Windows, features, labels, splits
├── data_pipeline_config.yaml     ← All parameters (v1.0)
├── run_pipeline.sh               ← End-to-end execution
│
├── data/
│   ├── bronze/
│   │   ├── raw_timeseries         ← Delta table
│   │   └── bronze_validation.log
│   │
│   ├── silver/
│   │   ├── signals                ← Delta table (normalized)
│   │   └── quality_report.json
│   │
│   ├── gold/
│   │   ├── X_windowed.npz         ← 90k × 146 features
│   │   ├── y_labels.npz           ← 90k × 1 binary labels
│   │   ├── splits.json            ← LOSO CV indices
│   │   ├── feature_engineering_params.json
│   │   ├── summary_statistics.json
│   │   └── pipeline_run_2026_03_25.log
│   │
│   └── eda/
│       ├── eda_report.md
│       ├── signal_profiles.csv
│       ├── loc_roc_distribution.png
│       └── signal_examples.png
│
└── docs/
    ├── README_DATA_PIPELINE.md
    ├── data_pipeline_plan.md (detailed)
    ├── quick_ref.md
    └── SIGNAL_DEFINITIONS.md
```
