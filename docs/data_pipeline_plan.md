# Data Pipeline Plan: LOC Prediction Model

## 1. PREDICTION TASK DEFINITION

### 1.1 Problem Statement
**Predict Loss of Consciousness (LOC) onset** using real physiological signals from propofol-induced anesthesia experiments.

### 1.2 Supervised Learning Task Formulation

```
REGRESSION → BINARY CLASSIFICATION (recommended)

INPUT:  60-second window of physiological signals (sampled at variable Hz)
OUTPUT: Binary label: 
        - Class 0: Normal (no LOC in next 30s)
        - Class 1: Pre-LOC (LOC occurs within next 30s)

LABEL ENGINEERING:
  For each timestamp t in the data:
    if (LOC_time - t) ≤ 30 seconds:
      label = 1 (pre-LOC, warning window)
    elif (LOC_time - t) > 120 seconds:
      label = 0 (normal, safe zone)
    else:
      label = -1 (uncertain, exclude from training)

LABEL DISTRIBUTION (expected):
  - Class 0 (Normal):   ~70% of samples
  - Class 1 (Pre-LOC):  ~25% of samples (30s warning window)
  - Excluded:           ~5% of samples (uncertain zone)
```

### 1.3 Data Split Strategy
```
LEAVE-ONE-SUBJECT-OUT (LOSO) Cross-Validation:
  Train:     8 subjects (80% of physiological diversity)
  Validate:  1 subject (held-out evaluation)
  Rotate across all 9 subjects → 9-fold CV

RATIONALE:
  - Prevents data leakage (no subject appears in multiple folds)
  - Mimics deployment: model must work on new patients
  - Acceptable for small dataset (N=9 subjects)

TEMPORAL SPLIT (within each subject):
  - First 60% of experiment timeline → training
  - Next 20% → validation
  - Last 20% → test
  (Rationale: propofol stages evolve; model sees full progression)
```

### 1.4 Success Metrics
```
PRIMARY METRIC: Sensitivity (Recall) for Class 1
  Target: ≥ 70% (catch 70% of pre-LOC episodes)
  Why: False negatives = missed warnings (dangerous)
  Acceptable false positive rate: ≤ 30% (alert fatigue vs. safety)

SECONDARY METRICS:
  - Specificity (True Negative Rate): ≥ 80%
  - ROC-AUC: ≥ 0.75
  - F1-Score (Class 1): ≥ 0.65
  
CROSS-VALIDATION PERFORMANCE:
  - Mean ± Std across 9 folds
  - Per-subject performance breakdown (understand failure modes)
```

---

## 2. SIGNAL SELECTION & FEATURE ENGINEERING

### 2.1 Available Signals (from 9 subjects)

#### **Tier 1: High-Priority Signals** (include all)
```
EDA (Electrodermal Activity):
  ✓ S#_eda_tonic.csv + S#_t_EDA_tonic       — tonic EDA baseline
  ✓ S#_muPR.csv, S#_sigmaPR.csv             — pulse rate: mean, std
  ✓ S#_mu_amp.csv, S#_sigma_amp.csv         — pulse amplitude: mean, std
  Why: EDA sensitive to autonomic nervous system changes during anesthesia

HRV (Heart Rate Variability):
  ✓ S#_muRR.csv, S#_sigmaRR.csv             — RR interval: mean, std
  ✓ S#_muHR.csv, S#_sigmaHR.csv             — HR: mean, std
  ✓ S#_LF.csv, S#_HF.csv                    — low freq, high freq power
  ✓ S#_LFnu.csv, S#_HFnu.csv                — normalized LF/HF
  ✓ S#_pow_tot.csv                          — total HRV power
  ✓ S#_ratio.csv                            — LF/HF ratio
  Why: HRV highly correlated with depth of anesthesia; discriminates LOC phases
```

#### **Tier 2: Contextual** (use if available)
```
Experiment Metadata:
  ✓ S#_events (timestamps of propofol stages)
    → encodes expected anesthesia progression
    → helps model learn stage-specific LOC risk
```

#### **Tier 3: Not Needed**
```
  ✗ EDA_temp_amp_#.csv (aggregated covariates for regression; not time-series)
```

### 2.2 Feature Engineering Strategy

#### **Raw Features** (direct from signals)
```
Time-Domain Features per 60-second window:
  - Mean, Std, Min, Max of each signal
  - Median, 25th percentile, 75th percentile
  - Range (max - min)
  - Coefficient of variation (std / mean)
  
Example: For S#_muHR.csv:
  mean_HR, std_HR, min_HR, max_HR, median_HR, range_HR, cv_HR

Total raw features: ~18 signals × 7 stats = 126 features
```

#### **Derived Features** (engineered)
```
Rate-of-Change (ROC):
  - HR trend: (HR_t - HR_t-30s) / 30
  - HRV trend: (HF_t - HF_t-30s) / 30
  Why: Anesthesia causes rapid HR/HRV decline before LOC

Autonomic Balance:
  - LF/HF ratio (sympathetic vs parasympathetic)
  - LFnu/HFnu trend
  Why: Parasympathetic dominance correlates with LOC

EDA Reactivity:
  - Tonic level (baseline)
  - Phasic amplitude changes (spikes in pulse rate)
  Why: EDA collapse is signature of LOC

Propofol Stage (context):
  - Time since stage onset (seconds)
  - Current stage number (1-10 encoded)
  Why: LOC risk increases with stage progression
```

#### **Total Feature Set**
```
Raw:     126 features
Derived:  20 features (ROC, autonomic balance, EDA, stage)
─────────────
Total:   ~146 features

Dimensionality reduction strategy:
  - Use tree-based feature importance (XGBoost) to prune
  - Target: keep top 30-40 features (reduce noise, improve generalization)
```

### 2.3 Signal Quality & Handling

```
MISSING DATA STRATEGY:
  Signal                    Frequency  Handling
  ─────────────────────────────────────────────────
  HR/HRV (muHR, sigmaHR)   ~1-4 Hz    → Interpolate (linear)
  EDA (muPR, mu_amp)       ~1-4 Hz    → Forward-fill if <5s gap
  HRV (LF, HF)            ~1-4 Hz    → Drop window if >10% missing
  
OUTLIER DETECTION:
  - IQR method: flag values > Q3 + 3×IQR
  - Domain knowledge: HR > 200 or < 30 → suspect (but keep for now)
  - Action: flag in metadata; use robust scaling during training

NORMALIZATION:
  - Per-subject z-score normalization (mean=0, std=1)
  - Rationale: Subject-specific baselines vary; model should learn changes
  - Applied at Silver layer; Gold layer preserves normalized values
```

---

## 3. DATA PIPELINE ARCHITECTURE

### 3.1 Medallion Architecture (Bronze → Silver → Gold)

```
                    ┌─────────────────────┐
                    │   RAW DATA FILES    │
                    │  (S#_*.csv format)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    BRONZE LAYER     │ ← Append-only, raw ingest
                    │  (data/bronze/)     │   No transformation
                    │                     │   Schema: subject, timestamp, signal_value
                    │  - raw_timeseries   │
                    │  - raw_locs         │
                    │  - raw_events       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │  Parse & Load  │  │  Validate       │  │  Handle Nulls   │
    │  (PySpark)     │  │  Schema         │  │  & Outliers     │
    │                │  │                 │  │                 │
    │  Silver Layer  │  │  Silver Layer   │  │  Silver Layer   │
    │  (data/silver/)│  │  (data/silver/) │  │  (data/silver/) │
    └─────────────┬──┘  └────────┬────────┘  └────────┬────────┘
                  │               │                    │
                  └───────────────┼────────────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  SILVER LAYER      │ ← Cleansed, conformed
                        │  (data/silver/)    │   Deduplicated
                        │                    │   Schema per subject:
                        │ - silver_signals   │   timestamp, signal_name,
                        │   (long format)    │   value, quality_flag
                        │ - silver_locs      │
                        │ - silver_events    │
                        │ - silver_metadata  │
                        └─────────┬──────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌────────▼────────┐
    │  Create 60s    │  │  Compute       │  │  Encode Labels  │
    │  Windows       │  │  Derived       │  │  (pre-LOC)      │
    │  (stride=1s)   │  │  Features      │  │                 │
    │                │  │  (ROC, LF/HF)  │  │                 │
    │  Gold Layer    │  │                │  │  Gold Layer     │
    │  (data/gold/)  │  │  Gold Layer    │  │  (data/gold/)   │
    └─────────┬──────┘  │  (data/gold/)  │  └────────┬────────┘
              │         └────────────────┘           │
              │                                      │
              └──────────────────┬───────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │   GOLD LAYER     │ ← ML-ready features
                        │  (data/gold/)    │   Labels prepared
                        │                  │   Train/val/test split
                        │ - X_windowed     │   (by subject & time)
                        │ - y_labels       │
                        │ - metadata       │
                        │ - splits.json    │
                        └──────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │  MODEL TRAINING  │
                        │  (phase 2)       │
                        └──────────────────┘
```

### 3.2 Pipeline Layer Details

#### **BRONZE: Raw Data Ingestion**

```
Input: CSV files from propofol study
  - S1_muHR.csv, S1_muRR.csv, ..., S9_sigmaHR.csv
  - S1_LOC.csv, S2_LOC.csv, ..., S9_ROC.csv
  - S1_events.csv, ..., S9_events.csv

Process:
  1. Load each CSV → infer schema (pandas initially, then Spark)
  2. Add metadata:
     - _ingested_at: timestamp
     - _source_file: filename (S1_muHR.csv)
     - _subject_id: extracted from filename (S1 → 1)
  3. Append to Delta Lake table (idempotent: dedup on subject_id + signal_name + timestamp)

Output Schema:
  subject_id: int
  signal_name: string (muHR, sigmaHR, LF, etc.)
  timestamp: double (seconds since experiment start)
  value: double
  _ingested_at: timestamp
  _source_file: string

Storage:
  - Table: data/bronze/raw_timeseries (Delta format)
  - Partitioning: subject_id, signal_name (for pruning)
  - Retention: 90 days (for audit/replay)
```

#### **SILVER: Cleanse, Conform, Deduplicate**

```
Input: Bronze layer

Process:
  1. DEDUPLICATE: Group by (subject_id, signal_name, timestamp)
     → Keep latest value if duplicates exist
  2. VALIDATE SCHEMA:
     - timestamp ≥ 0 and < 3600 (max 1 hour per experiment)
     - value within biological bounds (e.g., HR ∈ [30, 200])
     - Flag outliers but don't drop
  3. INTERPOLATE MISSING DATA:
     - For each subject × signal: sort by timestamp
     - Linear interpolation for gaps < 5 seconds
     - Forward-fill for gaps < 2 seconds
     - Drop values if gap > 60 seconds (too much uncertainty)
  4. NORMALIZE PER SUBJECT:
     - Calculate mean, std for each signal (per subject)
     - z-score normalize: (value - mean) / std
     - Store normalization params for inference
  5. ADD QUALITY FLAGS:
     - is_interpolated: boolean
     - is_outlier: boolean
     - interpolation_gap_seconds: double
     
Output Schema:
  subject_id: int
  signal_name: string
  timestamp: double
  value: double (normalized)
  value_raw: double (before normalization)
  is_interpolated: boolean
  is_outlier: boolean
  interpolation_gap_seconds: double
  _updated_at: timestamp

Storage:
  - Table: data/silver/signals (Delta format, partitioned by subject_id)
  - Quality checks: row-count & schema validation
  - Retention: permanent (audit trail)
```

#### **GOLD: Feature Engineering & Labels**

```
Input: Silver signals + LOC/ROC timestamps

Process:

  STEP 1: CREATE WINDOWS
    - For each subject, create 60-second sliding windows
    - Stride: 1 second (high overlap, but captures all potential LOC boundaries)
    - Window index: window_id = subject_id + window_start_timestamp
    
    Example:
      Subject 1, LOC at 450s:
      Window 1: [0s, 60s)      → label = 0 (normal)
      Window 2: [1s, 61s)      → label = 0
      ...
      Window 385: [385s, 445s) → label = 1 (LOC at 450s, within 30s warning)
      Window 386: [386s, 446s) → label = 1
      Window 387: [387s, 447s) → label = 1 ← LOC occurs during window
      ...

  STEP 2: AGGREGATE FEATURES PER WINDOW
    - For each window [t_start, t_start+60s]:
      * Filter Silver signals where timestamp ∈ [t_start, t_start+60)
      * Compute statistics:
        - mean(muHR), std(muHR), min(muHR), max(muHR), ...
        - (repeat for all 18 signals)
      * Compute derived features:
        - HR trend: (HR at t_start+60 - HR at t_start) / 60
        - LF/HF ratio trend
        - Time since current propofol stage onset
    
  STEP 3: ENCODE LABELS
    - For each window, lookup LOC_timestamp for subject
    - Calculate time_to_loc = LOC_timestamp - window_start_timestamp
    - Assign label:
      * 0 if time_to_loc > 30 seconds (safe zone)
      * 1 if time_to_loc ≤ 30 seconds and ≥ 0 (pre-LOC warning window)
      * -1 if time_to_loc < 0 (post-LOC, exclude)
    - Remove rows with label = -1
    
  STEP 4: TEMPORAL SPLIT
    - Per subject, sort by window_start_timestamp
    - Split:
      * Train: first 60% of experiment
      * Val: next 20%
      * Test: last 20%
    - Add column: split_set (train/val/test)

Output Schema (per window):
  subject_id: int
  window_id: string (subject_1_385000ms)
  window_start_timestamp: double
  label: int (0=normal, 1=pre_locs)
  split_set: string (train/val/test)
  
  [126 raw features]:
  mean_muHR, std_muHR, min_muHR, ... (7 stats × 18 signals)
  
  [20 derived features]:
  hr_trend, hf_trend, lf_hf_ratio, lf_hf_trend, 
  propofol_stage, time_since_stage_onset, ...

Storage:
  - Format: CSV (data/gold/X_windowed.csv) + NPZ (data/gold/X_windowed.npz)
  - NPZ structure:
    {
      'X': shape (N_samples, 128, 6) — if raw signals, NOT aggregated
      'X_features': shape (N_windows, 146) — aggregated features
      'y': shape (N_windows,) — labels (0/1)
      'subject_ids': shape (N_windows,) — for LOSO CV
      'splits': shape (N_windows,) — train/val/test
      'feature_names': list of 146 feature names
    }
  - Metadata: data/gold/feature_engineering_params.json
    {
      "subject_normalization": {1: {muHR: {mean, std}}, ...},
      "propofol_stage_times": {1: [0, 120, 240, ...], ...},
      "feature_names": [...],
      "label_definition": "1 = LOC within 30s"
    }
```

### 3.3 Data Quality & Validation

```
QUALITY CHECKPOINTS:

Bronze Layer:
  ✓ Row count: "Ingested 50,000 rows for S1_muHR.csv"
  ✓ Schema validation: All columns match expected types
  ✓ Duplicate check: Flag if same (subject, signal, timestamp) appears >1x
  ✓ Timestamp continuity: Min/max/count logged per signal

Silver Layer:
  ✓ Null rate per signal: Alert if >5% nulls
  ✓ Outlier rate: Alert if >2% values flagged as outliers
  ✓ Interpolation rate: Monitor; alert if >10% of values interpolated
  ✓ Normalization validation: Mean ≈ 0, Std ≈ 1 per signal per subject

Gold Layer:
  ✓ Label distribution: "Class 0: 70%, Class 1: 25%, Excluded: 5%"
  ✓ No NaNs in features: All 146 features have valid values
  ✓ Window count per subject: "Subject 1: 380 windows, Subject 2: 375, ..."
  ✓ Split distribution: "Train: 200 windows (60%), Val: 67 (20%), Test: 67 (20%)"
  ✓ Leakage check: No subject appears in multiple splits
  ✓ Feature statistics: Min/max/mean per feature logged (detect skew)
```

### 3.4 Reproducibility & Documentation

```
REPRODUCIBLE PIPELINE:
  
  Config File: data_pipeline_config.yaml
  ─────────────────────────────────────────
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  signals:
    - muHR, sigmaHR, muRR, sigmaRR
    - LF, HF, LFnu, HFnu, pow_tot, ratio
    - eda_tonic, muPR, sigmaPR, mu_amp, sigma_amp
  
  window_size_seconds: 60
  window_stride_seconds: 1
  locs_warning_window_seconds: 30
  
  quality_thresholds:
    max_null_rate: 0.05
    max_outlier_rate: 0.02
    max_interpolation_rate: 0.10
    hr_bounds: [30, 200]
  
  splits:
    train_ratio: 0.60
    val_ratio: 0.20
    test_ratio: 0.20
    strategy: temporal_per_subject
  
  version: v1.0
  created_date: 2026-03-25
  
  PIPELINE SCRIPT: data_pipeline_main.py
  ─────────────────────────────────────────
  Entry point: pipeline = DataPipelineOrchestrator(config_file)
              pipeline.run()
  
  Output: data/gold/X_windowed.npz, y_labels, splits.json, params.json
  Logs: data/pipeline_run_YYYY_MM_DD_HH_MM_SS.log
  
  VERSIONING:
  - Git tag each pipeline run: v_data_2026_03_25_windowed_146feat
  - Commit config + code → ensures reproducibility
```

---

## 4. IMPLEMENTATION ROADMAP

### Phase 0: Exploratory Data Analysis (Day 0, 4 hours)
**Tasks:**
- [ ] Download all 9 subjects' CSV files
- [ ] Profile each signal: range, nulls, sampling rate, duration
- [ ] Visualize 1-2 subjects: HR, HF/LF, EDA tonic over time; mark LOC/ROC
- [ ] Identify data quality issues (missing intervals, outliers)
- [ ] Estimate total samples post-windowing (~N subjects × ~10,000 seconds × 1Hz stride = 90k windows)

**Deliverable:** `data/eda_report.md` + 3-5 visualizations

---

### Phase 1: Bronze Layer (Day 1, 6 hours)
**Tasks:**
- [ ] Create `load_raw_data.py`: Iterate CSV files → create Bronze Delta table
- [ ] Test idempotency: run twice, verify row count doesn't double
- [ ] Validate schema & type inference
- [ ] Log row counts, timestamp ranges per signal

**Deliverable:** `data/bronze/raw_timeseries` Delta table + load script + validation log

---

### Phase 2: Silver Layer (Day 2-3, 8 hours)
**Tasks:**
- [ ] Create `silver_transform.py`: Dedup, validate, interpolate, normalize
- [ ] Per-subject normalization: compute/store params for inference later
- [ ] Quality checks: null rate, outlier rate, interpolation coverage
- [ ] Test on Subject 1; verify Silver table matches expected schema

**Deliverable:** `data/silver/signals` Delta table + transform script + quality report

---

### Phase 3: Gold Layer (Day 3-4, 8 hours)
**Tasks:**
- [ ] Create `gold_feature_engineering.py`: windowing, feature aggregation, labeling
- [ ] Generate 60-second windows with 1s stride
- [ ] Compute 146 features per window (raw + derived)
- [ ] Encode labels: pre-LOC (0/1), post-LOC excluded
- [ ] Temporal split: train/val/test per subject
- [ ] Export to NPZ + CSV + metadata JSON

**Deliverable:** `data/gold/X_windowed.npz`, `y_labels.npz`, `splits.json`, feature metadata + feature engineering script

---

### Phase 4: Validation & Docs (Day 4, 4 hours)
**Tasks:**
- [ ] Write `data_pipeline_config.yaml` with all parameters
- [ ] Create `README_DATA_PIPELINE.md`: architecture, quality checks, reproducibility
- [ ] Run end-to-end test: Bronze → Silver → Gold; verify shapes match expected
- [ ] Generate summary statistics: label distribution, feature ranges, null counts
- [ ] Commit all to Git; tag version `v_data_2026_03_25`

**Deliverable:** Complete pipeline + docs + version tag + ready for model training

---

## 5. EXPECTED OUTPUTS (Ready for Model Training)

```
data/gold/
├── X_windowed.npz                    ← Feature matrix
│   └── Keys: X (N, 146), feature_names (list), subject_ids, splits
├── y_labels.npz                      ← Labels
│   └── Keys: y (N,) with values 0/1
├── splits.json                       ← Train/val/test indices + subject mapping
├── feature_engineering_params.json   ← Normalization params, feature metadata
├── summary_statistics.json           ← Label distribution, feature stats
└── pipeline_run_2026_03_25.log      ← Full pipeline execution log

Key Stats (expected):
  - Total windows: ~90,000 (9 subjects × ~10k seconds × 1 Hz)
  - Class 0 (Normal):   ~63,000 (70%)
  - Class 1 (Pre-LOC):  ~22,500 (25%)
  - Excluded (Post-LOC): ~4,500 (5%)
  - Features:           146 (126 raw + 20 derived)
  - Train/Val/Test:     60% / 20% / 20% per subject
```

---

## 6. KEY DECISIONS & RATIONALE

| Decision | Choice | Why |
|----------|--------|-----|
| Prediction window | 60 seconds | Balance: long enough to capture trends; short enough for warning lead time |
| Warning lead time | 30 seconds | Typical reaction time for user + intervention; can adjust down to 15s for higher sensitivity |
| Window stride | 1 second | Captures all LOC boundary conditions; manageable dataset size (~90k windows) |
| Feature aggregation | 7 stats (mean, std, min, max, median, p25, p75) | Captures distribution shape; reduces dimensionality vs. raw signals |
| Normalization | Per-subject z-score | Subjects have different baselines; model learns changes relative to each subject's baseline |
| CV strategy | LOSO (Leave-One-Subject-Out) | Only 9 subjects; LOSO prevents subject-specific overfitting; mimics deployment (new patient) |
| Temporal split | 60/20/20 | Realistic: model must see full anesthesia progression (induction → deeper → recovery) |
| Derived features | ROC, LF/HF, stage context | Domain knowledge: physiological changes most predictive of LOC |

---

## 7. RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Small N=9 subjects → high variance | High | Use conservative splits; augment with synthetic data if needed; report confidence intervals |
| Data quality issues (gaps, outliers) | High | EDA first (Phase 0); flag quality metrics; consider robust models (isolation forest for anomalies) |
| Class imbalance (70/25%) | Medium | Use stratified sampling; class weights during training; monitor precision/recall separately |
| Temporal data leakage | Critical | Enforce temporal splits; train on early epochs, test on late epochs; validate no future info in features |
| LOC timing precision | Medium | Assume LOC time is ground truth; sensitivity analysis: what if ±5s error in LOC timestamp? |
| Feature explosion (146→?) | Low | Use tree feature importance to prune; start with top 30 features; add others only if improves CV score |

---

## 8. SUCCESS CRITERIA (Data Layer)

✅ **Data Quality**
- [ ] Zero data leakage: LOSO splits verified
- [ ] Null rate < 5% per signal
- [ ] No duplicate windows (subject + timestamp unique)
- [ ] Normalization parameters reproducible (stored in metadata)

✅ **Feature Engineering**
- [ ] 146 features generated consistently across all windows
- [ ] No NaN values in final feature matrix
- [ ] Label distribution: 70/25/5 (±5%)
- [ ] Feature ranges inspected; outliers documented

✅ **Reproducibility**
- [ ] Full pipeline runs end-to-end without errors
- [ ] Output identical if run twice (idempotent)
- [ ] Config file captures all parameters
- [ ] Commit hash linked to output version

✅ **Handoff to ML Team**
- [ ] NPZ files load in Python without errors
- [ ] Feature names, subject IDs, split assignments clear
- [ ] README documents feature definitions & data contract
- [ ] Normalization parameters available for inference pipeline
