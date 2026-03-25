# 📋 PHASE 0: READY TO EXECUTE

## Current Status
✅ **EDA script prepared**: `eda_phase0.py`  
⏳ **Waiting for**: Data zip file to complete downloading (`behavioral-and-autonomic-dynamics...zip`)

---

## ⏱️ What to Do When Download Completes

### Step 1: Extract ZIP File
```powershell
# Navigate to data folder
cd C:\Users\shayr\PycharmProjects\FaintDetectionPOC\data

# Extract (Replace the .crdownload file)
Expand-Archive -Path "behavioral-and-autonomic-dynamics-during-propofol-induced-unconsciousness-1.0.zip.crdownload" `
                -DestinationPath "." -Force
```

Or if the extension changes to `.zip` after download completes:
```powershell
Expand-Archive -Path "behavioral-and-autonomic-dynamics-during-propofol-induced-unconsciousness-1.0.zip" `
                -DestinationPath "." -Force
```

### Step 2: Run EDA Script
```powershell
# Navigate to project root
cd C:\Users\shayr\PycharmProjects\FaintDetectionPOC

# Run EDA
python eda_phase0.py
```

---

## 📊 What the EDA Script Does

**Inputs**: 
- S1_muHR.csv through S9_sigmaHR.csv (18 signals × 9 subjects)
- S1_LOC.csv, S1_ROC.csv (ground truth loss/recovery of consciousness)

**Outputs**:
```
data/eda/
├── signal_profiles.csv       ← Per-signal statistics (mean, std, min, max, null rate)
├── loc_roc_distribution.csv  ← LOC/ROC timestamps analysis
├── subject_summary.csv       ← Per-subject overview
├── eda_findings.txt          ← Text report (detailed findings)
├── signal_overview.png       ← Bar chart of signal means per subject
└── loc_roc_timeline.png      ← LOC/ROC timeline visualization
```

**Key Metrics Calculated**:
- Signal count, length, null rate per subject
- LOC/ROC timing (when consciousness was lost/recovered)
- Duration of unconsciousness
- Data quality indicators

---

## ✅ Success Criteria (Phase 0)

Phase 0 is complete when:
- [ ] All 6 output files are created in `data/eda/`
- [ ] `eda_findings.txt` contains signal profiles
- [ ] Signal visualizations display correctly
- [ ] No errors in console output

---

## 🚀 Next: Phase 1 (Bronze Layer)

Once Phase 0 finishes, we'll proceed to Phase 1: **Bronze Layer Ingestion**
- Create `load_raw_data.py`
- Load CSVs into Bronze Delta table
- Test idempotency
- Validate schema

---

## 📝 Current TODO Status

```sql
SELECT id, title, status FROM todos WHERE id IN ('data-eda', 'data-bronze-load');
```

Status will update to `done` when Phase 0 EDA completes successfully.

---

**⏳ Waiting for download... Check back once file extraction is complete!**
