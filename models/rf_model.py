"""Random Forest classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RFModel(BaseModel):
    """Random Forest classifier."""

    def __init__(self, cfg: dict) -> None:
        """Initialize RF from config dict."""
        self.model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Random Forest."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array."""
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        """Return feature importances from the trained RF."""
        return self.model.feature_importances_
