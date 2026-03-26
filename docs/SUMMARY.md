# ✅ DATA PIPELINE PLAN COMPLETE

## 📊 Summary

I've created a **comprehensive data pipeline plan** to prepare Loss of Consciousness (LOC) prediction data for model training.

### What You Get

**5 Planning Documents** (76 KB total):
- `README.md` — Complete navigation guide + executive summary
- `data_pipeline_plan.md` — Full technical spec (code patterns, reproducibility)
- `data_summary.md` — Prediction task + signals + features
- `quick_ref.md` — 1-page cheat sheet
- `visual_reference.md` — ASCII diagrams + architecture flows

**SQL Task Tracking**:
- 14 tasks across 5 phases
- 12 dependencies (ensures proper execution)
- Track status: pending → in_progress → done

---

## 🎯 The Plan at a Glance

### Prediction Task
```
INPUT:  60-second window of physiological signals (18 total: EDA + HRV + context)
OUTPUT: Binary classification — Will patient lose consciousness in next 30 seconds?

Class Distribution:
  • 70% Class 0 (Safe — no LOC in next 30s)
  • 25% Class 1 (Warning — LOC occurring within 30s)
  •  5% Excluded (Post-LOC, useless for training)
```

### Pipeline (Bronze → Silver → Gold)

**BRONZE** (6 hrs)
- Load raw CSVs
- Add metadata
- Append-only, zero transforms

**SILVER** (8 hrs)
- Deduplicate
- Normalize (z-score per subject)
- Interpolate <5s gaps
- Flag outliers
- Quality checks

**GOLD** (8 hrs)
- Create 60s windows (stride=1s) → ~90k windows
- Compute 146 features (126 raw + 20 derived)
- Encode labels (pre-LOC)
- Create LOSO CV splits (temporal 60/20/20)
- Export to NPZ + metadata

**RELEASE** (4 hrs)
- Config file (YAML)
- Documentation
- E2E test
- Git version tag

---

## 📊 Signals & Features

### 18 Signals Selected

**EDA (5)**
- eda_tonic, muPR, sigmaPR, mu_amp, sigma_amp

**HRV (10)**
- muRR, sigmaRR, muHR, sigmaHR, LF, HF, LFnu, HFnu, pow_tot, ratio

**Context (2)**
- propofol_stage, time_since_stage_onset

**Meta (1)**
- subject_id, timestamp

### 146 Features Generated

**Raw (126)**: 7 statistics × 18 signals
- mean, std, min, max, median, p25, p75

**Derived (20)**: Physiological markers + context
- HR trend, HF trend, LF/HF ratio, stage context, etc.

---

## 🏷️ Cross-Validation Strategy

**LOSO (Leave-One-Subject-Out)**
- Train on 8 subjects, test on 1
- Rotate across all 9 subjects → 9-fold CV
- No subject appears in multiple folds (prevents overfitting)
- Mimics deployment (model works on new patients)

**Temporal Splits (within each fold)**
- Train: 60% of subject's timeline
- Val: 20%
- Test: 20%
- Rationale: Model sees full anesthesia progression (induction → deeper → recovery)

---

## ✅ Quality Gates

| Layer | Gate | Threshold |
|-------|------|-----------|
| **Bronze** | Duplicates | 0 |
| **Silver** | Null rate | < 5% per signal |
| **Silver** | Outlier rate | < 2% |
| **Silver** | Interpolation | < 10% |
| **Gold** | NaN in features | 0 |
| **Gold** | Label distribution | 70/25/5 (±5%) |
| **Gold** | LOSO leakage | 0 subjects in multiple folds |

---

## 📅 Timeline

| Phase | Day(s) | Hours | Output |
|-------|--------|-------|--------|
| Phase 0: EDA | 0 | 4 | eda_report.md + plots |
| Phase 1: Bronze | 1 | 6 | raw_timeseries table |
| Phase 2: Silver | 2-3 | 8 | signals table (normalized) |
| Phase 3: Gold | 3-4 | 8 | X/y/splits/params (NPZ) |
| Phase 4: Release | 4 | 4 | Config, docs, v1.0 tag |
| **TOTAL** | **4 days** | **30h** | **Ready for ML** |

---

## 📦 Deliverables (Ready for Model Training)

```
data/gold/
├── X_windowed.npz              (90k × 146 features)
├── y_labels.npz                (90k × 1 binary)
├── splits.json                 (LOSO CV indices)
├── feature_engineering_params.json (normalization params)
└── summary_statistics.json     (label dist, feature ranges, QA metrics)
```

---

## 🔑 Key Decisions

| Decision | Why |
|----------|-----|
| **60-second window** | Captures physiological trends + realistic warning lead time |
| **30-second lead time** | User reaction/intervention window |
| **18 signals** | EDA (autonomic) + HRV (anesthesia depth) + context |
| **146 features** | 126 raw (distribution stats) + 20 derived (physiological trends) |
| **Per-subject normalization** | Accounts for individual baselines |
| **LOSO CV** | Only 9 subjects; prevents subject overfitting |
| **Temporal splits** | Model sees full progression; prevents temporal leakage |

---

## 🚨 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Small N=9 | LOSO CV + confidence intervals |
| Data quality | EDA first + quality gates per layer |
| Class imbalance | Class weights during training |
| Temporal leakage | Enforce temporal splits + validate |
| Feature explosion | Tree importance pruning |

---

## 📖 How to Use These Documents

**Want a quick overview?**
→ Read `quick_ref.md` (1 page)

**Need executive summary + visuals?**
→ Read `data_summary.md` + `visual_reference.md`

**Implementing the pipeline?**
→ Refer to `data_pipeline_plan.md` (full technical spec)

**Complete reference?**
→ See `README.md`

**Tracking tasks?**
→ Check SQL todos (14 tasks, 12 dependencies)

---

## 🎬 Next Steps

1. ✅ **Review & Approve** — Any changes to signals, features, or splits?
2. 🚀 **Start Phase 0 (EDA)** — Download data, profile signals, visualize
3. 📈 **Execute Phases 1-4** — Follow 4-day timeline, track in SQL todos
4. 📦 **Deliver Gold Layer** — Hand off NPZ + metadata to ML Team for model training

---

## ✨ STATUS: PLAN COMPLETE & READY TO EXECUTE

**Data Engineer is ready to build this pipeline.**

All planning documents are in: `C:\Users\shayr\.copilot\session-state\72850df1-0c98-4102-ba01-bd51000b70a4\`

Shall we start Phase 0 (EDA)?
