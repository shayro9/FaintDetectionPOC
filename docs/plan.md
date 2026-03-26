# FaintDetection MVP Sprint Plan (1-Week Timeline)

## Problem Statement
Build a real-time wearable fainting prediction MVP that demonstrates a working model capable of predicting fainting events. Launch within 1 week with minimal real-world data.

## Approach
**Phased execution** with parallel workstreams to maximize 1-week timeline:
- **Phase 1 (Days 1-2)**: Data pipeline completion & feature engineering
- **Phase 2 (Days 2-3)**: Model training & baseline evaluation
- **Phase 3 (Days 4-5)**: Real-time inference pipeline & edge optimization
- **Phase 4 (Days 5-7)**: Integration, testing, documentation & demo prep

## Key Assumptions
1. "Small dataset of real people" = supplemented with synthetic data (already in progress)
2. MVP = offline model trained, not running on wearable yet (iteration #2)
3. Success = model predicts syncope events with measurable accuracy
4. Real-time execution = batch inference on wearable, not continuous streaming
5. Constraints: Limited real data → accept lower accuracy; focus on working pipeline

---

## Sprint Breakdown

### **PHASE 1: DATA & FEATURES (Days 1-2)**

#### Task 1.1: Complete Data Generation Pipeline
- **User story**: As a ML engineer, I need synthetic + real data windowed and ready for model training
- **Deliverable**: `smartwatch_windows.npz` with (X, y, user_ids, session_ids)
- **Acceptance criteria**:
  - [ ] Data loader runs without errors
  - [ ] Output shape verified: (N_samples, 128_timesteps, 6_channels)
  - [ ] Class balance logged (normal/pre_syncope/syncope/recovery)
  - [ ] Event log shows planted fainting timestamps
- **Effort**: 4-6 hours
- **Owner**: Data Lead

#### Task 1.2: Feature Validation & Augmentation
- **User story**: As a data scientist, I need to ensure features are interpretable and informative
- **Deliverable**: Feature importance analysis + augmentation (if needed)
- **Acceptance criteria**:
  - [ ] Visualizations of 3-5 key features across classes (HR, SpO2, accel)
  - [ ] Time-domain features extracted (mean, std, min, max per window)
  - [ ] Frequency-domain features extracted (if applicable for accel/gyro)
  - [ ] Documentation: feature_engineering.md
- **Effort**: 3-4 hours
- **Owner**: Data Lead

#### Task 1.3: Train/Test Split Strategy
- **User story**: As a responsible ML engineer, I need to prevent data leakage
- **Deliverable**: LOSO (Leave-One-Subject-Out) CV split ready
- **Acceptance criteria**:
  - [ ] Train/val/test split by user_id (no user appears in multiple sets)
  - [ ] Stratification applied (balanced class distribution per fold)
  - [ ] Splits saved to `data/splits.json`
  - [ ] Documented in README
- **Effort**: 2-3 hours
- **Owner**: Data Lead

**Phase 1 Dependencies**: None (parallel start)
**Phase 1 Blockers**: None identified
**Go/No-Go Gate**: All 1.x tasks complete before Phase 2

---

### **PHASE 2: MODEL & TRAINING (Days 2-4)**

#### Task 2.1: Baseline Model Implementation
- **User story**: As a product manager, I need a working model to demonstrate predictions
- **Deliverable**: Trained model + predictions on test set
- **Acceptance criteria**:
  - [ ] Model type chosen (e.g., 1D CNN, LSTM, or XGBoost on hand-crafted features)
  - [ ] Model training script: `model/train.py`
  - [ ] Training completes in <30 minutes on available hardware
  - [ ] Model saved to `model/best_model.pkl` or `.h5`
  - [ ] No errors on inference
- **Effort**: 6-8 hours
- **Owner**: ML Engineer (primary)
- **Notes**: Start simple (tree-based or shallow CNN); iterate if time permits

#### Task 2.2: Model Evaluation & Metrics
- **User story**: As a medical device team, I need interpretable performance metrics
- **Deliverable**: Evaluation report with metrics + confusion matrix
- **Acceptance criteria**:
  - [ ] Metrics logged: Accuracy, Precision, Recall, F1, AUC-ROC per class
  - [ ] **Key metric**: Sensitivity (recall) for syncope class ≥ 70% (fewer missed events)
  - [ ] Confusion matrix + ROC curves saved to `results/evaluation/`
  - [ ] Cross-validation results documented (LOSO CV score)
  - [ ] Report: `results/evaluation_report.md`
- **Effort**: 4-5 hours
- **Owner**: ML Engineer

#### Task 2.3: Error Analysis & Interpretability
- **User story**: As a data scientist, I need to understand failure modes
- **Deliverable**: Root cause analysis of misclassifications
- **Acceptance criteria**:
  - [ ] Top 5 misclassified examples identified
  - [ ] Feature importance / SHAP analysis (if model supports)
  - [ ] Documented: `results/error_analysis.md`
  - [ ] Recommendations for next iteration
- **Effort**: 3-4 hours
- **Owner**: ML Engineer

**Phase 2 Dependencies**: Phase 1 complete
**Phase 2 Blockers**: Model training time or hardware limitations
**Go/No-Go Gate**: Model achieves ≥70% syncope recall on test set; otherwise, pivot to simpler model

---

### **PHASE 3: INFERENCE & DEPLOYMENT (Days 4-5)**

#### Task 3.1: Real-Time Inference Wrapper
- **User story**: As a wearable app developer, I need a lightweight inference engine
- **Deliverable**: `inference.py` that processes incoming sensor windows
- **Acceptance criteria**:
  - [ ] Function signature: `predict(sensor_window) → class, confidence`
  - [ ] Model loading cached (not reloaded per prediction)
  - [ ] Inference latency <500ms per window on target hardware (wearable)
  - [ ] Handles edge cases (missing channels, short sequences)
  - [ ] Unit tests: `test/test_inference.py`
- **Effort**: 4-5 hours
- **Owner**: ML Ops / Backend Lead

#### Task 3.2: Edge Model Optimization (Optional)
- **User story**: As a wearable platform engineer, I need a deployable model
- **Deliverable**: Model quantized / pruned if needed
- **Acceptance criteria**:
  - [ ] Model size <10 MB (if tree-based, already small; if DL, quantize)
  - [ ] Inference latency verified on target device
  - [ ] Performance drop <5% vs. original model
  - [ ] Saved as deployable artifact (ONNX, TFLite, etc.)
- **Effort**: 3-4 hours (optional; defer if time critical)
- **Owner**: ML Ops

#### Task 3.3: Alert/Notification Logic
- **User story**: As a user, I need clear alerts when fainting risk is detected
- **Deliverable**: Alert thresholding & formatting logic
- **Acceptance criteria**:
  - [ ] Thresholds defined per class (e.g., pre_syncope triggers "caution", syncope triggers "alert")
  - [ ] Alert formatting: timestamp + confidence + recommended action
  - [ ] Mock alert output tested: `test/test_alerts.py`
  - [ ] Documented in README
- **Effort**: 2-3 hours
- **Owner**: Product / Backend

**Phase 3 Dependencies**: Phase 2 complete (model trained)
**Phase 3 Blockers**: Hardware-specific latency issues
**Go/No-Go Gate**: Inference latency <500ms; alerts functional

---

### **PHASE 4: INTEGRATION & DEMO (Days 5-7)**

#### Task 4.1: End-to-End Pipeline Testing
- **User story**: As a QA engineer, I need to verify the full flow
- **Deliverable**: Integration test suite
- **Acceptance criteria**:
  - [ ] Test: Synthetic data → preprocessing → model → alert (no errors)
  - [ ] Test: Real data sample (if available)
  - [ ] All edge cases handled (empty data, corrupted input)
  - [ ] Tests: `test/test_e2e.py`
  - [ ] CI/CD pipeline configured (GitHub Actions or equivalent)
- **Effort**: 4-5 hours
- **Owner**: QA / DevOps

#### Task 4.2: Documentation & Demo Preparation
- **User story**: As a stakeholder, I need to understand what was built and why
- **Deliverable**: Comprehensive documentation + demo script
- **Acceptance criteria**:
  - [ ] README updated: architecture, setup, usage
  - [ ] Model card created: `MODEL_CARD.md` (dataset, performance, limitations)
  - [ ] Demo script ready: `demo.py` (runs inference on sample data, visualizes predictions)
  - [ ] Architecture diagram: `docs/architecture.png`
  - [ ] Limitations & future work documented
- **Effort**: 4-6 hours
- **Owner**: Tech Lead / Product

#### Task 4.3: Demo Execution & Feedback
- **User story**: As a product lead, I need to demonstrate to stakeholders
- **Deliverable**: Live demo showing model predictions
- **Acceptance criteria**:
  - [ ] Demo runs without errors (end-to-end: sample data → predictions → alerts)
  - [ ] Results clearly visualized (confusion matrix, sample predictions)
  - [ ] Q&A prepared: limitations, next steps, risks
  - [ ] Feedback captured & recorded for iteration planning
- **Effort**: 2-3 hours
- **Owner**: Product Lead

#### Task 4.4: Retrospective & Sprint Closure
- **User story**: As a team lead, I need to capture lessons learned
- **Deliverable**: Sprint retrospective + iteration backlog
- **Acceptance criteria**:
  - [ ] Team retrospective held: what went well, what didn't, why
  - [ ] Backlog created for iteration #2 (improvements, real wearable deployment, etc.)
  - [ ] Known issues & blockers documented
  - [ ] Sprint metrics logged (velocity, cycle time, etc.)
- **Effort**: 1-2 hours
- **Owner**: Scrum Master / Tech Lead

**Phase 4 Dependencies**: Phase 3 complete (inference ready)
**Phase 4 Blockers**: Documentation gaps or critical bugs from testing
**Go/No-Go Gate**: All acceptance criteria met; demo executable and bug-free

---

## Success Criteria (MVP Definition)

### Functional
- ✅ Model successfully predicts syncope class with ≥70% recall
- ✅ End-to-end pipeline: data → preprocessing → model → alert
- ✅ Inference latency <500ms per sample
- ✅ Demo runs live without errors

### Documentation
- ✅ README with setup and usage instructions
- ✅ Model card with performance metrics and limitations
- ✅ Architecture diagram and data flow
- ✅ Known limitations and future roadmap

### Quality
- ✅ No data leakage (LOSO CV strategy)
- ✅ Test coverage: model inference, alerts, E2E pipeline
- ✅ Reproducible results (seed set, parameters logged)

---

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Small dataset → poor model performance | High | Critical | Use synthetic data; lower accuracy threshold; accept ensemble / bootstrap approaches |
| Real-world data quality issues | Medium | High | Exploratory data analysis upfront; robust preprocessing |
| Model training time exceeds budget | Medium | Medium | Start with simple model (XGBoost); scale up only if time permits |
| Hardware latency on wearable | Medium | High | Early latency testing (Task 3.1); quantization/pruning ready (Task 3.2) |
| Stakeholder expectations misalignment | Low | Critical | Clear communication: this is MVP, limited by data; show roadmap for iteration #2 |

---

## Capacity & Allocation

**Assumed Team**: 2-3 engineers + 1 PM (1 week, ~30-40 hours/person)

### Role Breakdown
- **Data Lead** (1 engineer, 15-20 hours): Tasks 1.1–1.3
- **ML Engineer** (1 engineer, 18-24 hours): Tasks 2.1–2.3, 3.1
- **ML Ops / Backend** (0.5 engineer, 5-10 hours): Task 3.2–3.3
- **QA / DevOps** (0.5 engineer, 4-8 hours): Task 4.1
- **Tech Lead / Product** (1 engineer + 1 PM, 8-12 hours): Tasks 4.2–4.4

### Parallelization
- **Days 1-2**: Phase 1 (all data tasks in parallel)
- **Days 2-4**: Phase 2 (training while data finalized)
- **Days 4-5**: Phase 3 (parallel: inference, optimization, alerts)
- **Days 5-7**: Phase 4 (integration, testing, docs, demo)

---

## Deliverables Checklist

### Code
- [ ] `data_generation/main.py` produces train/test splits
- [ ] `model/train.py` trains and evaluates model
- [ ] `inference.py` handles real-time predictions
- [ ] `test/` directory with unit + integration tests
- [ ] `demo.py` for stakeholder demo

### Data
- [ ] `data/smartwatch_windows.npz` with windowed features
- [ ] `data/splits.json` with LOSO CV configuration
- [ ] `results/evaluation_report.md` with metrics
- [ ] `results/error_analysis.md` with failure modes

### Documentation
- [ ] `README.md` (updated with full architecture & usage)
- [ ] `MODEL_CARD.md` (performance, limitations, future work)
- [ ] `docs/architecture.png` (system diagram)
- [ ] `CHANGELOG.md` (sprint accomplishments)

### Artifacts
- [ ] Trained model: `model/best_model.pkl` or `.h5`
- [ ] Confusion matrices + ROC curves
- [ ] Feature importance visualizations

---

## Timeline Overview

```
Day 1-2: Data preparation + feature engineering
├─ Task 1.1: Data pipeline
├─ Task 1.2: Feature engineering
└─ Task 1.3: Train/test split

Day 2-4: Model development
├─ Task 2.1: Baseline model training
├─ Task 2.2: Evaluation & metrics
└─ Task 2.3: Error analysis

Day 4-5: Inference & deployment
├─ Task 3.1: Inference wrapper
├─ Task 3.2: Edge optimization (optional)
└─ Task 3.3: Alert logic

Day 5-7: Integration & launch
├─ Task 4.1: E2E testing
├─ Task 4.2: Documentation
├─ Task 4.3: Demo
└─ Task 4.4: Retrospective

Friday EOD: MVP LAUNCH ✅
```

---

## Next Iteration (Post-MVP)

1. **Deploy to real wearable hardware** (iOS/Android integration)
2. **Collect real-world data** for continuous model improvement
3. **Reduce false positives** (current: ~30% FP acceptable; target: <10% for production)
4. **Add real-time streaming** (current: batch; target: continuous 5s windows with rolling alerts)
5. **Regulatory & safety** (CE/FDA if applicable; clinical validation study)
