"""
Phase 0: Exploratory Data Analysis (EDA) for Propofol Study
============================================================

Profiles 9 subjects' physiological signals during propofol-induced anesthesia.
Generates: signal profiles, LOC/ROC distribution, sample visualizations, findings report.

Usage:
  python eda_phase0.py

Output:
  data/eda/
  ├── signal_profiles.csv          (per-signal statistics per subject)
  ├── loc_roc_distribution.csv     (LOC/ROC timing analysis)
  ├── subject_summary.csv          (per-subject overview)
  ├── eda_findings.txt             (text report)
  ├── signal_overview.png          (sample signals visualization)
  └── loc_roc_timeline.png         (LOC/ROC timeline)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("../../data") / "human_data" / "Data"
EDA_OUTPUT_DIR = Path("../../data") / "eda"

# Signals to profile (18 total, organized by type)
SIGNALS = [
    # EDA (5 signals)
    ("eda_tonic", "EDA"),
    ("muPR", "EDA"), ("sigmaPR", "EDA"),
    ("mu_amp", "EDA"), ("sigma_amp", "EDA"),
    # HRV (10 signals)
    ("muRR", "HRV"), ("sigmaRR", "HRV"),
    ("muHR", "HRV"), ("sigmaHR", "HRV"),
    ("LF", "HRV"), ("HF", "HRV"),
    ("LFnu", "HRV"), ("HFnu", "HRV"),
    ("pow_tot", "HRV"), ("ratio", "HRV"),
]

# Propofol study has 9 subjects
SUBJECTS = list(range(1, 10))

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def create_output_dir():
    """Create output directory."""
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_OUTPUT_DIR


def find_subjects():
    """Find all subject directories (S1, S2, ..., S9)."""
    # Propofol study has 9 subjects (S1-S9)
    subjects = []
    for i in SUBJECTS:
        # Check if any signal file exists for this subject
        pattern = f"S{i}_muHR.csv"
        files = list(DATA_DIR.glob(pattern))
        if files:
            subjects.append(i)
    return sorted(subjects) if subjects else SUBJECTS


def load_signal(subject_id, signal_name):
    """Load signal CSV for a subject (handles comma-separated row format)."""
    pattern = f"S{subject_id}_{signal_name}.csv"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    try:
        df = pd.read_csv(files[0], header=None)
        # Handle both: single row with multiple values OR column format
        values = df.values.flatten()
        result_df = pd.DataFrame({"value": values})
        return result_df
    except Exception as e:
        print(f"  ⚠️  Error loading {pattern}: {e}")
        return None


def load_timestamps(subject_id, signal_name):
    """Load timestamps for a signal."""
    # Common timestamp files
    ts_patterns = [
        f"S{subject_id}_t_{signal_name}.csv",
        f"S{subject_id}_t_HRV.csv",
        f"S{subject_id}_t_EDA.csv",
        f"S{subject_id}_t_EDA_tonic.csv",
    ]
    
    for pattern in ts_patterns:
        files = list(DATA_DIR.glob(pattern))
        if files:
            try:
                ts = pd.read_csv(files[0], header=None)
                ts.columns = ["timestamp"]
                return ts
            except:
                pass
    return None


def load_loc_roc(subject_id):
    """Load LOC (Loss of Consciousness) and ROC (Recovery) timestamps."""
    # The data has a master LOC_ROC.csv file with all subjects
    loc_roc_file = DATA_DIR / "LOC_ROC.csv"
    
    if not loc_roc_file.exists():
        return None, None
    
    try:
        loc_roc_data = pd.read_csv(loc_roc_file, header=None)
        # Subject IDs are 1-indexed; LOC_ROC rows correspond to S1, S2, ..., S9
        if subject_id - 1 < len(loc_roc_data):
            row = loc_roc_data.iloc[subject_id - 1]
            loc_time = row[0] if pd.notna(row[0]) else None
            roc_time = row[1] if pd.notna(row[1]) else None
            return loc_time, roc_time
    except Exception as e:
        print(f"  ⚠️  Error loading LOC/ROC for subject {subject_id}: {e}")
    
    return None, None


def profile_signal(subject_id, signal_name):
    """Profile a single signal - fast streaming approach for large files."""
    pattern = f"S{subject_id}_{signal_name}.csv"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    
    try:
        file_path = files[0]
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Fast approach: parse raw values without pandas overhead for single-row files
        with open(file_path, 'r') as f:
            line = f.readline().strip()
        
        # Parse comma-separated values efficiently
        values_str = line.split(',')
        
        # Convert to numeric, skip NaN
        values_clean = []
        total_count = len(values_str)
        nan_count = 0
        
        for v in values_str:
            if v == 'NaN':
                nan_count += 1
            else:
                try:
                    values_clean.append(float(v))
                except ValueError:
                    nan_count += 1
        
        values_arr = np.array(values_clean)
        
        return {
            "subject": subject_id,
            "signal": signal_name,
            "count": total_count,
            "non_null_count": len(values_clean),
            "mean": float(np.mean(values_arr)) if len(values_arr) > 0 else np.nan,
            "std": float(np.std(values_arr)) if len(values_arr) > 0 else np.nan,
            "min": float(np.min(values_arr)) if len(values_arr) > 0 else np.nan,
            "max": float(np.max(values_arr)) if len(values_arr) > 0 else np.nan,
            "median": float(np.median(values_arr)) if len(values_arr) > 0 else np.nan,
            "null_count": nan_count,
            "null_rate": nan_count / total_count if total_count > 0 else 0,
            "file_size_mb": file_size_mb,
        }
    except Exception as e:
        print(f"  ⚠️  Error profiling {pattern}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_eda():
    """Execute Phase 0 EDA."""
    
    print("\n" + "="*80)
    print("  PHASE 0: EXPLORATORY DATA ANALYSIS (EDA)")
    print("  Profiling propofol study data (9 subjects)")
    print("="*80 + "\n")
    
    output_dir = create_output_dir()
    print(f"📁 Output directory: {output_dir}\n")
    
    # Step 1: Find subjects
    print("Step 1: Discovering subjects...")
    subjects = find_subjects()
    print(f"  ✅ Found {len(subjects)} subjects: {subjects}\n")
    
    if not subjects:
        print("  ❌ No subject data found in data/ directory")
        print("  Please ensure data files are extracted (S1_*.csv, S2_*.csv, etc.)")
        return False
    
    # Step 2: Profile each signal
    print("Step 2: Profiling signals...")
    signal_profiles = []
    for subject_id in subjects:
        print(f"  Subject {subject_id}...")
        for signal_name, _ in SIGNALS:
            profile = profile_signal(subject_id, signal_name)
            if profile is not None:
                signal_profiles.append(profile)
    
    if not signal_profiles:
        print("  ❌ No signal data found")
        return False
    
    signals_df = pd.DataFrame(signal_profiles)
    signals_df.to_csv(output_dir / "signal_profiles.csv", index=False)
    print(f"  ✅ Profiled {len(signals_df)} signals\n")
    
    # Step 3: Analyze LOC/ROC
    print("Step 3: Analyzing LOC/ROC timestamps...")
    loc_roc_data = []
    for subject_id in subjects:
        loc_time, roc_time = load_loc_roc(subject_id)
        loc_roc_data.append({
            "subject": subject_id,
            "LOC_timestamp": loc_time,
            "ROC_timestamp": roc_time,
            "LOC_to_ROC_duration": (roc_time - loc_time) if (loc_time and roc_time) else None,
        })
    
    loc_roc_df = pd.DataFrame(loc_roc_data)
    loc_roc_df.to_csv(output_dir / "loc_roc_distribution.csv", index=False)
    print(f"  ✅ Analyzed LOC/ROC for {len(loc_roc_df)} subjects\n")
    
    # Step 4: Generate subject summaries
    print("Step 4: Generating subject summaries...")
    subject_summaries = []
    for subject_id in subjects:
        subject_signals = signals_df[signals_df['subject'] == subject_id]
        
        if len(subject_signals) > 0:
            loc_row = loc_roc_df[loc_roc_df['subject'] == subject_id].iloc[0]
            
            subject_summaries.append({
                "subject": subject_id,
                "signal_count": len(subject_signals),
                "avg_signal_length": subject_signals['count'].mean(),
                "avg_null_rate": subject_signals['null_rate'].mean(),
                "LOC_timestamp": loc_row['LOC_timestamp'],
                "ROC_timestamp": loc_row['ROC_timestamp'],
                "consciousness_duration_sec": loc_row['LOC_to_ROC_duration'],
            })
    
    subject_df = pd.DataFrame(subject_summaries)
    subject_df.to_csv(output_dir / "subject_summary.csv", index=False)
    print(f"  ✅ Generated summaries for {len(subject_df)} subjects\n")
    
    # Step 5: Generate text report
    print("Step 5: Generating findings report...")
    with open(output_dir / "eda_findings.txt", "w") as f:
        f.write("EDA PHASE 0: PROPOFOL STUDY DATA PROFILING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total subjects: {len(subjects)}\n")
        f.write(f"Total signals profiled: {len(signals_df)}\n")
        f.write(f"Signals per subject: {signals_df.groupby('subject').size().iloc[0]} (avg)\n\n")
        
        f.write("SIGNAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(signals_df.to_string())
        f.write("\n\n")
        
        f.write("LOC/ROC ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(loc_roc_df.to_string())
        f.write("\n\n")
        
        f.write("SUBJECT SUMMARIES\n")
        f.write("-" * 80 + "\n")
        f.write(subject_df.to_string())
        f.write("\n\n")
        
        f.write("DATA QUALITY NOTES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average null rate: {signals_df['null_rate'].mean():.2%}\n")
        f.write(f"Max null rate: {signals_df['null_rate'].max():.2%}\n")
        f.write(f"Signals with nulls: {(signals_df['null_count'] > 0).sum()} / {len(signals_df)}\n\n")
        
        f.write("NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("Phase 1: Bronze layer ingestion (raw data loading)\n")
        f.write("Phase 2: Silver layer transformation (cleansing, normalization)\n")
        f.write("Phase 3: Gold layer features (windowing, feature engineering)\n")
        f.write("Phase 4: Release (documentation, versioning)\n")
    
    print(f"  ✅ Report saved to {output_dir / 'eda_findings.txt'}\n")
    
    # Step 6: Create visualizations
    print("Step 6: Creating visualizations...")
    
    try:
        # Visualization 1: Signal overview
        fig, axes = plt.subplots(len(subjects), 1, figsize=(12, 3 * len(subjects)))
        if len(subjects) == 1:
            axes = [axes]
        
        for idx, subject_id in enumerate(subjects):
            subject_signals = signals_df[signals_df['subject'] == subject_id]
            signal_names = subject_signals['signal'].values
            means = subject_signals['mean'].values
            
            axes[idx].bar(range(len(signal_names)), means)
            axes[idx].set_title(f"Subject {subject_id} - Signal Means")
            axes[idx].set_ylabel("Mean Value")
            axes[idx].set_xticks(range(len(signal_names)))
            axes[idx].set_xticklabels(signal_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "signal_overview.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved signal_overview.png\n")
    except Exception as e:
        print(f"  ⚠️  Could not create signal overview: {e}\n")
    
    try:
        # Visualization 2: LOC/ROC timeline
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_loc_roc = loc_roc_df[loc_roc_df['LOC_timestamp'].notna()]
        
        if len(valid_loc_roc) > 0:
            subjects_plot = valid_loc_roc['subject'].astype(str).values
            loc_times = valid_loc_roc['LOC_timestamp'].values
            roc_times = valid_loc_roc['ROC_timestamp'].values
            
            y_pos = np.arange(len(subjects_plot))
            
            # Plot LOC and ROC
            ax.scatter(loc_times, y_pos, color='red', s=100, label='LOC', marker='o')
            ax.scatter(roc_times, y_pos, color='green', s=100, label='ROC', marker='s')
            
            # Connect LOC to ROC
            for i, (loc, roc) in enumerate(zip(loc_times, roc_times)):
                if pd.notna(loc) and pd.notna(roc):
                    ax.plot([loc, roc], [i, i], 'k--', alpha=0.3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(subjects_plot)
            ax.set_xlabel("Timestamp (seconds)")
            ax.set_ylabel("Subject")
            ax.set_title("LOC/ROC Timeline (Loss and Recovery of Consciousness)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "loc_roc_timeline.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved loc_roc_timeline.png\n")
    except Exception as e:
        print(f"  ⚠️  Could not create LOC/ROC timeline: {e}\n")
    
    # Final Summary
    print("="*80)
    print("  PHASE 0 COMPLETE ✅")
    print("="*80)
    print(f"\n📊 EDA Summary:")
    print(f"   • Subjects: {len(subjects)}")
    print(f"   • Signals profiled: {len(signals_df)}")
    print(f"   • Average null rate: {signals_df['null_rate'].mean():.2%}")
    print(f"   • LOC/ROC records: {len(loc_roc_df)}")
    print(f"\n📁 Outputs saved to: {output_dir}/")
    print(f"   • signal_profiles.csv")
    print(f"   • loc_roc_distribution.csv")
    print(f"   • subject_summary.csv")
    print(f"   • eda_findings.txt")
    print(f"   • signal_overview.png")
    print(f"   • loc_roc_timeline.png")
    print("\n✅ Ready for Phase 1: Bronze Layer Ingestion\n")
    
    return True


if __name__ == "__main__":
    success = run_eda()
    exit(0 if success else 1)
