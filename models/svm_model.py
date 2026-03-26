"""SVM classifier with internal StandardScaler pipeline."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base_model import BaseModel


class SVMModel(BaseModel):
    """SVM classifier with internal StandardScaler pipeline."""

    def __init__(self, cfg: dict) -> None:
        """Initialize SVM from config dict."""
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=cfg["C"],
                gamma=cfg["gamma"],
                kernel=cfg["kernel"],
                probability=True,
            )),
        ])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the SVM pipeline."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array."""
        return self.model.predict_proba(X)

    def feature_importances(self) -> None:
        """SVM does not support feature importances."""
        return None
