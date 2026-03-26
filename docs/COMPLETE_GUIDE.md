# 🏥 Complete Data Engineering Plan: LOC Prediction

## Three Simple Questions → Complete Answer

### ❓ 1. What Prediction Task?

**Binary Classification:**
```
INPUT:   60-second window of 18 physiological signals
OUTPUT:  Will patient lose consciousness in the next 30 seconds?
         YES (1) → Pre-LOC warning zone (alert user!)
         NO  (0) → Safe, continue monitoring
```

**Why 60 seconds + 30 seconds?**
- 60s = captures physiological trends (HR decline, HRV collapse, EDA changes)
- 30s = realistic time for user intervention/emergency response

**Data Source:**
- 9 real subjects
- Propofol-induced anesthesia (controlled LOC events)
- Ground truth: exact LOC timestamps

---

### ❓ 2. Which Signals to Use?

**18 Signals Selected** (divided into 3 categories):

```
EDA (Electrodermal Activity) — Autonomic Nervous System
├─ eda_tonic                 (baseline skin conductance)
├─ muPR, sigmaPR             (pulse rate: mean, std)
└─ mu_amp, sigma_amp         (pulse amplitude: mean, std)
   → EDA collapses during LOC; most sensitive to autonomic changes

HRV (Heart Rate Variability) — Anesthesia Depth Indicator
├─ muRR, sigmaRR             (RR interval: mean, std)
├─ muHR, sigmaHR             (HR: mean, std)
├─ LF, HF                    (Low/high frequency power)
├─ LFnu, HFnu                (Normalized LF/HF)
├─ pow_tot                   (Total HRV power)
└─ ratio                     (LF/HF ratio)
   → HRV most predictive of anesthesia depth; markers LOC transition

Context — Experimental Stage Information
├─ propofol_stage            (1-10, induction → recovery)
└─ time_since_stage_onset    (seconds into current stage)
   → Helps model learn stage-specific LOC risk
```

**Why these?**
- EDA: Direct measure of autonomic collapse
- HRV: Strongest predictor of anesthesia depth
- Context: Explains temporal patterns (LOC risk increases with stage progression)

---

### ❓ 3. How to Organize Data Pipeline?

**Medallion Architecture: Bronze → Silver → Gold**

```
┌─ RAW CSVs ─────────────────────────┐
│ S1_muHR.csv, ..., S9_sigmaHR.csv   │
└────────────┬────────────────────────┘
             │
    ┌────────▼────────┐
    │ PHASE 1: BRONZE │
    │ (Raw Ingest)    │
    │ • Load CSVs     │
    │ • Zero changes  │
    │ • Append-only   │
    │ • Add metadata  │
    │ Time: 6 hours   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ PHASE 2: SILVER │
    │ (Cleanse)       │
    │ • Deduplicate   │
    │ • Normalize     │
    │ • Interpolate   │
    │ • Flag outliers │
    │ • Quality checks│
    │ Time: 8 hours   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ PHASE 3: GOLD   │
    │ (Features)      │
    │ • 60s windows   │
    │ • 146 features  │
    │ • Labels        │
    │ • LOSO splits   │
    │ Time: 8 hours   │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │PHASE 4: RELEASE │
    │ • Config        │
    │ • Docs          │
    │ • E2E test      │
    │ • Version tag   │
    │ Time: 4 hours   │
    └────────┬────────┘
             │
    ┌────────▼──────────────┐
    │ READY FOR ML TEAM     │
    │ (Phase 2: Training)   │
    └───────────────────────┘
```

---

## 🧬 Feature Engineering: 146 Total

### Raw Features (126)
```
For each of 18 signals, compute 7 statistics:
├─ mean (central tendency)
├─ std (variability)
├─ min (floor value)
├─ max (peak value)
├─ median (robust average)
├─ p25 (25th percentile)
└─ p75 (75th percentile)

Applied to: muHR, sigmaHR, muRR, sigmaRR, LF, HF, LFnu, HFnu, 
            pow_tot, ratio, eda_tonic, muPR, sigmaPR, mu_amp, sigma_amp, + 3 more

Total: 7 × 18 = 126 raw features per window
```

### Derived Features (20)
```
Physiological Markers (understanding LOC transitions):
├─ HR_trend              (is heart rate declining? → LOC signal)
├─ HF_trend              (is parasympathetic dominant? → LOC signal)
├─ LF_HF_ratio           (autonomic balance shifting? → LOC signal)
├─ LF_HF_trend           (ratio changing over window? → LOC signal)
├─ EDA_tonic_change      (EDA collapsing? → LOC signal)
└─ [additional exploratory features]

Context Markers (when is LOC most likely?):
├─ propofol_stage        (deeper stages = higher LOC risk)
├─ time_since_stage      (longer time = may approach next stage)
└─ experiment_progress   (% of total experiment elapsed)

Total: 20 derived features per window
```

**TOTAL: 126 + 20 = 146 features per 60-second window**

---

## 📊 Labels & Cross-Validation

### How Labels Are Created
```
For each 60-second window:
  time_to_loc = LOC_timestamp - window_start_timestamp

  if time_to_loc ≤ 30s AND ≥ 0:    label = 1 (WARNING — LOC coming!)
  elif time_to_loc > 30s:          label = 0 (SAFE — no LOC risk)
  else (time_to_loc < 0):          excluded (POST-LOC — useless)

Expected Distribution:
  Class 0 (Safe):       ~70% (~63,000 samples)
  Class 1 (Warning):    ~25% (~22,500 samples)
  Excluded (Post-LOC):  ~5% (~4,500 samples)
```

### LOSO Cross-Validation (Leave-One-Subject-Out)
```
Why LOSO? Only 9 subjects → need strategy to prevent subject-specific overfitting

Process:
  Fold 1: Train on S2-S9 (80% data) → Test on S1 (20% data, never seen before)
  Fold 2: Train on S1,S3-S9          → Test on S2
  ...
  Fold 9: Train on S1-S8             → Test on S9

Result: 9-fold cross-validation
  • No subject appears in multiple folds (prevents leakage)
  • Model tested on completely new subjects (realistic deployment scenario)

Temporal Split (within each fold):
  Train: First 60% of subject's timeline (induction phase)
  Val:   Next 20% (deepening phase)
  Test:  Last 20% (recovery phase)
  
  Rationale: Model must see full anesthesia progression; prevents temporal leakage
```

---

## ✅ Quality Assurance: 3-Layer Validation

### BRONZE Layer Quality
```
✓ Row counts logged per signal
✓ Timestamp ranges verified (0 to 3600 seconds)
✓ Type inference validated
✓ Duplicate detection (alert if same subject+signal+timestamp appears 2x)
```

### SILVER Layer Quality
```
✓ Null rate < 5% per signal (alert if higher)
✓ Outlier rate < 2% (domain bounds: HR ∈ [30, 200], etc.)
✓ Interpolation coverage < 10% (flag if >10% filled values)
✓ Normalization verification: mean ≈ 0, std ≈ 1 per signal per subject
✓ Quality score attached to each signal
```

### GOLD Layer Quality
```
✓ No NaN values in 146 features (critical blocker)
✓ Label distribution: 70±5% / 25±5% / 5±3% (alert if skewed)
✓ Feature statistics logged (detect outliers/skew in distributions)
✓ LOSO split validation: confirm no subject appears in multiple folds
✓ Window count per subject logged (~10k windows per subject)
✓ Feature engineering parameters reproducible (stored & versioned)
```

---

## 📅 4-Day Implementation Timeline

```
DAY 0 (Phase 0: EDA)               │  4 hours  │ eda_report.md
└─ Download CSVs                   │           │ + visualizations
  ├─ Profile signals (range, nulls)
  ├─ Visualize HR/HF/LF/EDA over time
  ├─ Mark LOC/ROC on plots
  └─ Document findings & issues

DAY 1 (Phase 1: Bronze)            │  6 hours  │ raw_timeseries
└─ Load raw data                   │           │ Delta table
  ├─ Iterate CSV files
  ├─ Create Bronze Delta table
  ├─ Test idempotency (run 2x, verify no duplicates)
  └─ Log validation metrics

DAY 2-3 (Phase 2: Silver)          │  8 hours  │ signals table
└─ Cleanse & normalize             │           │ (normalized)
  ├─ Deduplicate per subject+signal+timestamp
  ├─ Validate bounds (HR, SpO2, etc.)
  ├─ Interpolate <5s gaps (linear)
  ├─ Normalize per-subject z-score (mean=0, std=1)
  ├─ Flag outliers + quality scores
  └─ Run quality checks (<5% nulls, <2% outliers)

DAY 3-4 (Phase 3: Gold)            │  8 hours  │ X/y/splits NPZ
└─ Feature engineering             │           │ + metadata JSON
  ├─ STEP 1: Create 60s windows (stride=1s) → ~90k windows
  ├─ STEP 2: Aggregate 126 raw features per window
  ├─ STEP 3: Compute 20 derived features (trends, ratios)
  ├─ STEP 4: Encode labels (pre-LOC classification)
  ├─ STEP 5: Create temporal splits (60/20/20 per subject)
  ├─ STEP 6: Map LOSO CV folds (no leakage)
  └─ STEP 7: Export to NPZ + metadata JSON

DAY 4 (Phase 4: Release)           │  4 hours  │ Config + docs
└─ Docs & version                  │           │ + v1.0 tag
  ├─ Write data_pipeline_config.yaml
  ├─ Create README_DATA_PIPELINE.md
  ├─ Run end-to-end test
  ├─ Verify output shapes & counts
  ├─ Git version tag: v_data_2026_03_25
  └─ Ready for ML Team!

─────────────────────────────────────────────────────────────
TOTAL: 30 hours (4 days) → Ready for Phase 2 (Model Training)
```

---

## 📦 Deliverables (Ready for ML Team)

### Data Files
```
data/gold/
├── X_windowed.npz
│   └─ Keys: X (90k × 146), feature_names, subject_ids, splits
│
├── y_labels.npz
│   └─ Keys: y (90k,) with values 0/1
│
├── splits.json
│   └─ LOSO CV fold assignments + subject mapping
│
├── feature_engineering_params.json
│   └─ Normalization parameters (mean/std per signal per subject)
│
└── summary_statistics.json
    └─ Label distribution, feature ranges, quality metrics
```

### Code
```
data_pipeline/
├── load_raw_data.py          (Bronze ingestion)
├── silver_transform.py       (Cleanse & normalize)
├── gold_feature_engineering.py (Windows, features, labels, splits)
├── data_pipeline_config.yaml (All parameters, reproducible)
└── run_pipeline.sh           (One-command execution)
```

### Documentation
```
├── README_DATA_PIPELINE.md   (How to use + reproduce)
├── eda_report.md             (Exploratory analysis findings)
├── signal_definitions.md     (What each signal means)
└── data_pipeline_config.yaml (Version 1.0, all parameters)
```

---

## 🎯 Success Criteria (Data Layer)

**Before handing to ML Team, verify:**

✅ **Data Quality**
- [ ] Zero data leakage (LOSO CV verified, subjects don't cross folds)
- [ ] Zero temporal leakage (features don't include future information)
- [ ] Null rate < 5% per signal
- [ ] Outlier rate < 2%
- [ ] NaN in features = 0 (critical blocker)

✅ **Feature Engineering**
- [ ] 146 features generated for each window
- [ ] Label distribution: 70/25/5 (±5%)
- [ ] Per-subject normalization parameters stored for inference

✅ **Reproducibility**
- [ ] Pipeline runs end-to-end without errors
- [ ] Output identical if run twice (idempotent)
- [ ] Config file captures all parameters
- [ ] Code versioned in Git

✅ **Handoff**
- [ ] NPZ files load in Python without errors
- [ ] Feature names documented (146 names provided)
- [ ] Subject IDs and split assignments clear
- [ ] Normalization parameters available for inference pipeline

---

## 🚀 Getting Started

**Step 1: Choose Your Entry Point**
- Want quick overview? → Read `SUMMARY.md` or `quick_ref.md`
- Want executive summary? → Read `data_summary.md`
- Want implementation details? → Read `data_pipeline_plan.md`
- Want complete reference? → Read `README.md`

**Step 2: Track Progress**
- Query SQL todos table to see 14 tasks + 12 dependencies
- Update task status as phases complete

**Step 3: Start Phase 0 (EDA)**
- Download all 9 subject CSV files
- Profile each signal (range, nulls, sampling rate, duration)
- Create visualization report
- Document any data quality issues found

**Step 4: Follow Timeline**
- Days 1-4: Execute phases sequentially (Bronze → Silver → Gold → Release)
- Each phase has clear acceptance criteria
- Quality gates must pass before proceeding

**Step 5: Deliver Outputs**
- Provide NPZ files + metadata to ML Team
- They can immediately start Phase 2: Model Training
- Your data pipeline is reproducible for future iterations

---

**You're ready to build this pipeline. Start with Phase 0 (EDA) whenever ready.**
