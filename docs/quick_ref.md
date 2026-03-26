# Quick Reference: LOC Prediction Data Pipeline

## 🎯 The Task
**Predict Loss of Consciousness (LOC)** from propofol anesthesia data  
**Input:** 60-sec window of 18 physiological signals  
**Output:** Binary: will patient lose consciousness in next 30s?

---

## 📊 Signals (18 Total)

| Category | Signals | Count |
|----------|---------|-------|
| **EDA** | eda_tonic, muPR, sigmaPR, mu_amp, sigma_amp | 5 |
| **HRV** | muRR, sigmaRR, muHR, sigmaHR, LF, HF, LFnu, HFnu, pow_tot, ratio | 10 |
| **Context** | propofol_stage, time_since_stage | 2 |
| **Meta** | subject_id, timestamp | 1 |

---

## 🏗️ Pipeline: 3 Layers

```
BRONZE (Raw)           → SILVER (Clean)        → GOLD (Features)
├─ Load CSVs           ├─ Dedup                ├─ 60s windows (stride=1s)
├─ Zero transforms     ├─ Normalize            ├─ Aggregate: 7 stats/signal
├─ Append-only         ├─ Interpolate gaps     ├─ Encode labels (0/1)
└─ Add metadata        ├─ Flag outliers        ├─ Split: 60/20/20 temporal
                       └─ Quality checks       └─ LOSO CV strategy
```

---

## 📈 Features Generated: 146 Total

| Type | Count | Method |
|------|-------|--------|
| **Raw** | 126 | 7 stats × 18 signals (mean, std, min, max, median, p25, p75) |
| **Derived** | 20 | Trends (HR, HF), ratios (LF/HF), context (stage, time_in_stage) |

---

## 🏷️ Labels

```
time_to_loc = LOC_timestamp - window_start_timestamp

if time_to_loc ≤ 30s:    label = 1 (pre-LOC, warn!)
elif time_to_loc > 30s:  label = 0 (normal)
else:                    excluded (post-LOC, useless)

Distribution:
  Class 0: ~70% (safe)
  Class 1: ~25% (warning window)
  Excluded: ~5% (post-LOC)
```

---

## 📁 Data Split Strategy

```
LEAVE-ONE-SUBJECT-OUT (LOSO) Cross-Validation:
  Fold 1: Train on S2-S9, Test on S1
  Fold 2: Train on S1,S3-S9, Test on S2
  ...
  Fold 9: Train on S1-S8, Test on S9

Within each fold:
  Train: 60% of subject's timeline (early epochs)
  Val:   20%
  Test:  20% (late epochs — realistic deployment scenario)

Why temporal split?
  → Model must see full anesthesia progression (induction → deeper → recovery)
  → Prevents temporal leakage
```

---

## ✅ Quality Gates

| Layer | Gate | Threshold | Action |
|-------|------|-----------|--------|
| **Bronze** | Duplicates | 0 | Alert & investigate |
| **Silver** | Null rate | <5% per signal | Interpolate or exclude |
| **Silver** | Outlier rate | <2% | Flag in metadata |
| **Silver** | Interpolation | <10% | Monitor; exclude if >20% |
| **Gold** | NaN in features | 0 | Block release |
| **Gold** | Label distribution | 70/25/5 (±5%) | Verify; alert if skewed |
| **Gold** | LOSO leakage | 0 subjects in multiple folds | Verify splits |

---

## 📅 4-Day Timeline

| Day | Phase | Deliverable | Hours |
|-----|-------|-------------|-------|
| **0** | EDA | Profile + visualize signals | 4 |
| **1** | Bronze | Load raw data → Delta table | 6 |
| **2-3** | Silver | Cleanse, normalize, validate | 8 |
| **3-4** | Gold | Features, labels, splits, export | 8 |
| **4** | Release | Config, docs, E2E test, version | 4 |

**Total: 30 hours → Ready for model training**

---

## 📦 Outputs

```
data/gold/
├── X_windowed.npz          (90k samples × 146 features)
├── y_labels.npz            (90k × 1, binary)
├── splits.json             (LOSO CV indices + subject mapping)
├── feature_engineering_params.json (normalization params)
└── summary_statistics.json (label dist, feature ranges)

Metadata:
├── feature_names.txt       (146 feature names)
├── signal_definitions.md   (EDA/HRV signal explanations)
└── pipeline_config.yaml    (all parameters, version)
```

---

## 🚀 Success Metrics (Data Layer)

✅ **Zero Data Leakage**: LOSO verified, temporal splits enforced  
✅ **Quality**: Null <5%, outlier <2%, NaN = 0  
✅ **Reproducibility**: Config + code deterministic; idempotent pipeline  
✅ **Handoff**: NPZ files load in Python without errors  

---

## ⚠️ Key Risks

| Risk | Mitigation |
|------|-----------|
| Small N=9 | LOSO CV; confidence intervals; augment if needed |
| Data quality | EDA first; quality gates at each layer |
| Class imbalance | Class weights during training; monitor F1 |
| Temporal leakage | Enforce temporal splits; validate features |

---

## 💡 Key Decisions

```
Window: 60s         → Enough for trends; 30s warning is realistic
Signals: 18         → EDA (autonomic) + HRV (depth) + context
Features: 146       → 126 raw + 20 derived (domain knowledge)
Normalization: Z-score per subject → Accounts for individual baselines
CV: LOSO → Mimics deployment (new patients); prevents overfitting
Split: Temporal 60/20/20 → Model sees full anesthesia progression
```

---

**Questions?** See `data_pipeline_plan.md` for full details.
