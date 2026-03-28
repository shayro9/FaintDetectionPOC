"""Leave-One-Out cross-validation trainer for LOC prediction."""

from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.preprocessing import StandardScaler

from loc_prediction.models import BaseModel, RFModel, RNNModel, SVMModel
from loc_prediction.training import metrics


class Patient(Protocol):
    """Expected interface for patient data objects."""

    id: str
    X_features: np.ndarray  # (N, F) feature matrix
    X_raw: np.ndarray       # (N, T, 4) raw signal windows
    y: np.ndarray           # (N,) binary labels
    time_to_loc: np.ndarray # (N,) seconds before LOC (0 for non-event windows)


def _scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler on train, apply to both; handles 2-D and 3-D inputs."""
    scaler = StandardScaler()
    if X_train.ndim == 3:
        N_tr, T, C = X_train.shape
        N_te = X_test.shape[0]
        X_train = scaler.fit_transform(X_train.reshape(N_tr * T, C)).reshape(N_tr, T, C)
        X_test = scaler.transform(X_test.reshape(N_te * T, C)).reshape(N_te, T, C)
    else:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test


def _make_model(model_name: str, model_cfg: dict, pos_weight: float) -> BaseModel:
    """Instantiate the correct model class from name, config, and pos_weight."""
    if model_name == "svm":
        return SVMModel(model_cfg)
    if model_name == "rf":
        return RFModel(model_cfg)
    if model_name == "rnn":
        return RNNModel(model_cfg, pos_weight)
    raise ValueError(f"Unknown model: {model_name!r}")


class LOOTrainer:
    """Leave-One-Out cross-validation trainer."""

    def __init__(self, cfg: dict, wandb_run: Any) -> None:
        """Store full config and active wandb run."""
        self.cfg = cfg
        self.run = wandb_run

    def run(self, patients: list[Patient], model_name: str, model_cfg: dict) -> None:
        """Execute LOO cross-validation for one model and log all results."""
        n = len(patients)
        threshold = self.cfg["evaluation"]["threshold"]
        bins = self.cfg["evaluation"]["time_to_event_bins"]
        X_attr = "X_features" if model_cfg["input"] == "features" else "X_raw"

        fold_results: list[dict] = []
        importances_per_fold: list[np.ndarray] = []

        for i, test_patient in enumerate(patients):
            train_patients = [p for j, p in enumerate(patients) if j != i]
            train_ids = ", ".join(p.id for p in train_patients)
            print(f"Fold {i + 1}/{n} — Test: {test_patient.id} — Train: {train_ids}")

            X_train = np.concatenate([getattr(p, X_attr) for p in train_patients])
            y_train = np.concatenate([p.y for p in train_patients])
            X_test = getattr(test_patient, X_attr)

            X_train, X_test = _scale(X_train, X_test)

            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            pos_weight = n_neg / n_pos

            model = _make_model(model_name, model_cfg, pos_weight)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)

            has_loc = bool(test_patient.y.any())
            fold_result = metrics.evaluate_fold(
                test_patient.y, y_prob, test_patient.time_to_loc,
                threshold, test_patient.id, has_loc,
            )
            fold_results.append(fold_result)

            self.run.log({
                f"fold_{i}/auc": fold_result["auc"],
                f"fold_{i}/first_detection_time": fold_result["first_detection_time"] or float("nan"),
                f"fold_{i}/false_alarm_rate": fold_result["false_alarm_rate"],
            })

            imp = model.feature_importances()
            if imp is not None:
                importances_per_fold.append(imp)

        self._log_summary(fold_results, importances_per_fold, bins)

    def _log_summary(
        self,
        fold_results: list[dict],
        importances_per_fold: list[np.ndarray],
        bins: list[int],
    ) -> None:
        """Log summary metrics and plots to wandb."""
        summary = metrics.compute_summary(fold_results)
        self.run.log({
            "summary/mean_auc": summary["mean_auc"],
            "summary/std_auc": summary["std_auc"],
            "summary/mean_detection_time": summary["mean_detection_time"],
        })

        fig = metrics.plot_time_to_event(fold_results, bins)
        self.run.log({"summary/time_to_event_plot": wandb.Image(fig)})
        plt.close(fig)

        if importances_per_fold:
            n_features = importances_per_fold[0].shape[0]
            feature_names = [f"feature_{j}" for j in range(n_features)]
            fig = metrics.plot_feature_importance(importances_per_fold, feature_names)
            self.run.log({"summary/feature_importance": wandb.Image(fig)})
            plt.close(fig)
