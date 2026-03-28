# 🏥 FaintDetection MVP: Complete Data Pipeline Plan

## Executive Summary

**Objective:** Build a production-grade data pipeline to prepare **Loss of Consciousness (LOC) prediction data** from 9 real anesthesia subjects for machine learning model training.

**Timeline:** 4 days (30 hours) → Ready for Phase 2: Model Training  
**Approach:** Medallion Lake House Architecture (Bronze → Silver → Gold)  
**Data Sources:** 9 subjects, 18 physiological signals (EDA + HRV metrics)  
**Deliverable:** 90k windowed samples × 146 features with LOSO cross-validation splits

---

## 📊 The Prediction Problem

### What We're Building
```
INPUT:  60-second window of physiological signals
OUTPUT: Binary classification — Will patient lose consciousness in next 30 seconds?
        Class 0: Safe (no LOC risk)
        Class 1: Warning (LOC occurring within 30s)
```

### Why This Matters
- **30-second lead time** = realistic window for user intervention
- **Real physiological signals** = propofol-induced anesthesia (controlled LOC events)
- **Ground truth labels** = exact LOC timestamps (no guessing)
- **Small but real dataset** = 9 subjects (constraints drive thoughtful engineering)

---

## 📋 What You're Getting

### Plan Documents
1. **`data_pipeline_plan.md`** (24 KB) — Full technical spec with code patterns, data contracts, reproducibility strategy
2. **`data_summary.md`** (9 KB) — Executive summary with task definition, signals, architecture, timeline
3. **`quick_ref.md`** (5 KB) — One-page reference: signals, features, splits, success metrics
4. **`visual_reference.md`** (11 KB) — Architecture diagrams, feature flows, quality gates, checklist

### SQL Todos
- **14 tasks** across 4 phases with **12 dependencies**
- Track from Phase 0 (EDA) through Phase 4 (Release)
- Queryable status + dependency management

### Reference Materials
- **`plan.md`** — Original sprint plan (1-week MVP overview)

---

## 🔑 Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Prediction window** | 60 seconds | Enough for trend observation; realistic warning lead time |
| **Warning lead time** | 30 seconds | User reaction time + intervention window |
| **Signals** | 18 (EDA + HRV + context) | Autonomic & HRV most predictive of LOC transition |
| **Features** | 146 (126 raw + 20 derived) | 7 statistics per signal + physiological trends + propofol context |
| **Normalization** | Per-subject z-score | Accounts for individual baselines; model learns changes |
| **CV Strategy** | LOSO (Leave-One-Subject-Out) | Small N=9; prevents subject-specific overfitting |
| **Split Strategy** | Temporal 60/20/20 | Model sees full anesthesia progression; prevents temporal leakage |
| **Architecture** | Bronze→Silver→Gold | Reproducible, observable, quality-gated pipeline |

---

## 🏗️ Pipeline Architecture (3 Layers)

### LAYER 1: BRONZE (Raw Ingest)
```
Input:   CSV files (S1_muHR.csv, ..., S9_sigmaHR.csv)
Process: Load → add metadata → append-only Delta table
Output:  Immutable raw data (schema: subject_id, signal_name, timestamp, value)
QA:      Row counts, timestamp ranges, duplicate detection
Time:    6 hours
```

### LAYER 2: SILVER (Cleanse & Conform)
```
Input:   Bronze raw signals
Process: Dedup → Validate → Interpolate gaps → Normalize (z-score) → Flag outliers
Output:  Cleansed, normalized signals (per-subject baselines)
QA:      Null rate <5%, outlier rate <2%, interpolation <10%
Time:    8 hours
```

### LAYER 3: GOLD (Features & Labels)
```
Input:   Silver normalized signals + LOC/ROC timestamps
Process: 
  1. Create 60s windows (stride=1s) → ~90k windows
  2. Aggregate features: 7 stats × 18 signals = 126 raw
  3. Compute derived: HR trend, HF trend, LF/HF ratio, stage context = 20 derived
  4. Encode labels: Pre-LOC (1 if time_to_LOC ≤ 30s, 0 else)
  5. Split: Temporal 60/20/20 per subject (LOSO CV)
Output:  X (90k × 146 features), y (90k binary labels), splits (LOSO indices)
QA:      No NaN, label dist 70/25/5, LOSO leakage check
Time:    8 hours
```

---

## 📊 Signals & Features

### Available Signals (18 Total)

**EDA (Electrodermal Activity)** — Autonomic nervous system
- eda_tonic (baseline)
- muPR, sigmaPR (pulse rate: mean, std)
- mu_amp, sigma_amp (pulse amplitude: mean, std)

**HRV (Heart Rate Variability)** — Anesthesia depth
- muRR, sigmaRR (RR interval: mean, std)
- muHR, sigmaHR (HR: mean, std)
- LF, HF (Low/high frequency power)
- LFnu, HFnu (Normalized LF/HF)
- pow_tot (Total HRV power)
- ratio (LF/HF ratio)

### Features Generated (146 Total)

**Raw Features (126)**
- 7 statistics per signal: mean, std, min, max, median, p25, p75
- Applied across 18 signals
- Captures signal distribution + central tendency

**Derived Features (20)**
- HR trend, HF trend, LF/HF ratio, LF/HF trend (physiological changes)
- Propofol stage, time_since_stage_onset (context)
- Additional exploratory features (EDA reactivity, autonomic balance)

---

## 🏷️ Labels & Stratification

### Label Definition
```
For each 60-second window:
  time_to_loc = LOC_timestamp - window_start_timestamp
  
  if time_to_loc ≤ 30s AND ≥ 0:    label = 1 (pre-LOC, warning zone)
  elif time_to_loc > 30s:           label = 0 (normal, safe)
  else (time_to_loc < 0):           excluded (post-LOC, useless)

Expected Distribution:
  Class 0 (Normal):    ~70% (~63k samples)
  Class 1 (Pre-LOC):   ~25% (~22.5k samples)
  Excluded (Post-LOC): ~5% (~4.5k samples)
```

### Cross-Validation Strategy: LOSO
```
Leave-One-Subject-Out (9-fold):
  Fold 1: Train on S2-S9, Test on S1
  Fold 2: Train on S1,S3-S9, Test on S2
  ...
  Fold 9: Train on S1-S8, Test on S9

Within each fold:
  Train: 60% of subject's timeline
  Val:   20%
  Test:  20%

Rationale:
  - No subject appears in multiple folds (no leakage)
  - Temporal split mimics real deployment (model sees progression)
  - Only 9 subjects → LOSO necessary
```

---

## ✅ Quality Assurance Strategy

### Quality Gates (Per Layer)

**BRONZE Layer**
- ✓ Row count logs
- ✓ Timestamp continuity
- ✓ Duplicate detection
- ✓ Type inference validation

**SILVER Layer**
- ✓ Null rate < 5% per signal (alert if higher)
- ✓ Outlier rate < 2% (domain bounds enforced)
- ✓ Interpolation < 10% (monitor gap patterns)
- ✓ Normalization verified (mean ≈ 0, std ≈ 1 per subject per signal)
- ✓ Quality score per signal

**GOLD Layer**
- ✓ No NaN values in 146 features
- ✓ Label distribution: 70±5% / 25±5% / 5±3%
- ✓ Feature statistics logged (detect skew/outliers)
- ✓ LOSO validation (no subject leakage)
- ✓ Window count per subject logged

### Data Quality Report
All metrics exported to `summary_statistics.json` + per-layer quality logs.

---

## 📅 Implementation Timeline (4 Days)

| Phase | Day(s) | Duration | Tasks | Output |
|-------|--------|----------|-------|--------|
| **Phase 0: EDA** | 0 | 4h | Profile signals, visualize, plan features | eda_report.md + plots |
| **Phase 1: Bronze** | 1 | 6h | Load raw data, validate schema, test idempotency | raw_timeseries Delta table |
| **Phase 2: Silver** | 2-3 | 8h | Dedup, normalize, interpolate, quality checks | signals Delta table (normalized) |
| **Phase 3: Gold** | 3-4 | 8h | Windows (60s, stride=1s), compute 146 features, encode labels, splits | X/y/splits/params NPZ files |
| **Phase 4: Release** | 4 | 4h | Config, docs, E2E test, Git version tag | Complete pipeline + v1.0 |
| | | **30h** | | **Ready for Model Training** |

---

## 📦 Deliverables (Ready for Phase 2)

### Data Files
```
data/gold/
├── X_windowed.npz              ← Feature matrix (90k × 146)
├── y_labels.npz                ← Labels (90k × 1) binary
├── splits.json                 ← LOSO CV indices + subject mapping
├── feature_engineering_params.json  ← Normalization params, feature metadata
└── summary_statistics.json      ← Label distribution, feature ranges, quality metrics
```

### Code & Config
```
data_pipeline/
├── load_raw_data.py            ← Bronze ingestion script
├── silver_transform.py         ← Cleanse & normalize
├── gold_feature_engineering.py ← Windows, features, labels, splits
├── data_pipeline_config.yaml   ← All parameters (reproducible)
└── run_pipeline.sh             ← End-to-end execution
```

### Documentation
```
├── README_DATA_PIPELINE.md     ← Architecture, feature definitions, QA process
├── eda_report.md               ← Exploratory analysis + findings
├── signal_definitions.md       ← EDA/HRV signal explanations
└── pipeline_config.yaml        ← Version 1.0, all parameters captured
```

---

## 🚨 Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **Small N=9 subjects** | High | High | LOSO CV + confidence intervals; augment with synthetic if needed |
| **Data quality issues** | High | High | EDA first phase; quality gates at each layer; robust preprocessing |
| **Class imbalance (70/25)** | Medium | Medium | Class weights during model training; stratified sampling; monitor F1 separately |
| **Temporal data leakage** | Critical | Critical | Enforce temporal splits; validate no future information in features |
| **Feature explosion (146→?)** | Low | Medium | Tree feature importance pruning; start with top 30; add incrementally |
| **Signal gaps/outliers** | Medium | Medium | Interpolation strategy + outlier flagging; quality thresholds enforced |
| **Propofol timing precision** | Low | Medium | Manual inspection of event timestamps; sensitivity analysis on ±5s error |

---

## 🎓 Success Criteria (Data Layer)

✅ **Data Quality**
- [ ] Zero data leakage verified (LOSO CV splits)
- [ ] Zero temporal leakage (features don't include future information)
- [ ] Null rate < 5% per signal
- [ ] Outlier rate < 2%
- [ ] No NaN values in final feature matrix

✅ **Feature Engineering**
- [ ] 146 features generated consistently
- [ ] Label distribution: 70/25/5 (±5%)
- [ ] Per-subject normalization parameters stored for inference

✅ **Reproducibility**
- [ ] Pipeline runs end-to-end without errors
- [ ] Output identical if run twice (idempotent)
- [ ] Config file captures all parameters
- [ ] Code + config version tagged in Git

✅ **Handoff to ML Team**
- [ ] NPZ files load in Python without errors
- [ ] Feature names, subject IDs, split assignments clear
- [ ] README documents feature definitions + data contracts
- [ ] Normalization parameters available for inference pipeline

---

## 📚 Document Navigation

### For Quick Overview
→ Start with **`quick_ref.md`** (1-page cheat sheet)

### For Complete Understanding
→ Read **`data_summary.md`** (executive summary) + **`visual_reference.md`** (diagrams)

### For Implementation Details
→ Refer to **`data_pipeline_plan.md`** (full technical spec with code patterns)

### For Task Tracking
→ Check SQL todos table (14 tasks, 12 dependencies)

### For Overall Sprint Context
→ See **`plan.md`** (1-week MVP sprint plan)

---

## 🚀 Next Steps

1. **Review this plan** — Any questions or changes needed?
2. **Confirm signals + features** — Are these the right predictors for LOC?
3. **Approve window size + lead time** — 60s + 30s warning acceptable?
4. **Start Phase 0 (EDA)** — Download data, profile signals, generate visualizations
5. **Execute Phases 1-4** — Follow the 4-day timeline; track todos in SQL
6. **Deliver Gold Layer** — Hand off NPZ + metadata to ML Team for model training

---

## 💬 Questions?

- **What's in each signal?** → See `quick_ref.md` or `data_summary.md`
- **How are features computed?** → See `data_pipeline_plan.md` (STEP 2 in Gold Layer)
- **What's the quality bar?** → See `visual_reference.md` (Quality Gates section)
- **How do I run the pipeline?** → See `README_DATA_PIPELINE.md` (to be created during Phase 4)

---

**Status:** ✅ Plan Complete  
**Ready to:** Execute Phase 0 (EDA) → Begin building data infrastructure  
**Estimated Completion:** 4 days from start → Ready for ML team
