"""Evaluation metrics and plots for LOO cross-validation."""

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")

_WINDOW_STEP_S = 1.0  # assumed window stride in seconds (1 window = 1 second)


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    time_to_loc: np.ndarray,
    threshold: float,
    patient_id: str,
    has_loc: bool,
) -> dict:
    """Evaluate predictions for a single LOO fold."""
    result: dict = {"patient_id": patient_id, "has_loc": has_loc}

    if len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        result["auc"] = float("nan")

    if has_loc:
        fires_before_loc = (y_prob[:, 1] >= threshold) & (time_to_loc > 0)
        result["first_detection_time"] = (
            float(time_to_loc[fires_before_loc].max()) if fires_before_loc.any() else None
        )
    else:
        result["first_detection_time"] = None

    negative_mask = y_true == 0
    n_neg = negative_mask.sum()
    n_fp = ((y_prob[:, 1] >= threshold) & negative_mask).sum()
    total_neg_hours = (n_neg * _WINDOW_STEP_S) / 3600.0
    result["false_alarm_rate"] = float(n_fp / total_neg_hours) if total_neg_hours > 0 else 0.0

    return result


def compute_summary(fold_results: list[dict]) -> dict:
    """Aggregate fold dicts into mean/std of each metric."""
    aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
    detection_times = [r["first_detection_time"] for r in fold_results if r["first_detection_time"] is not None]
    false_alarm_rates = [r["false_alarm_rate"] for r in fold_results]

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc": float(np.std(aucs)) if aucs else float("nan"),
        "mean_detection_time": float(np.mean(detection_times)) if detection_times else float("nan"),
        "std_detection_time": float(np.std(detection_times)) if detection_times else float("nan"),
        "mean_false_alarm_rate": float(np.mean(false_alarm_rates)),
        "std_false_alarm_rate": float(np.std(false_alarm_rates)),
    }


def plot_time_to_event(fold_results: list[dict], bins: list[int]) -> matplotlib.figure.Figure:
    """Bar chart of sensitivity per time-to-LOC bin across LOC folds."""
    loc_results = [r for r in fold_results if r["has_loc"]]
    n_loc = len(loc_results)

    bin_labels = [">10min", "5-10min", "2-5min", "<2min"]
    sensitivities = [
        sum(
            1 for r in loc_results
            if r["first_detection_time"] is not None and r["first_detection_time"] >= t
        ) / n_loc
        for t in bins
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bin_labels, sensitivities)
    ax.set_xlabel("Time before LOC")
    ax.set_ylabel("Sensitivity (fraction detected)")
    ax.set_ylim(0, 1)
    ax.set_title("Detection Sensitivity by Time-to-Event Bin")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances_per_fold: list[np.ndarray],
    feature_names: list[str],
) -> matplotlib.figure.Figure:
    """Bar chart of mean ± std feature importance across folds."""
    importances = np.array(importances_per_fold)
    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)

    order = np.argsort(mean_imp)[::-1]
    fig, ax = plt.subplots(figsize=(max(10, len(feature_names) // 2), 5))
    x = np.arange(len(feature_names))
    ax.bar(x, mean_imp[order], yerr=std_imp[order], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance (mean ± std across folds)")
    plt.tight_layout()
    return fig
