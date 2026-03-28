"""Abstract base class for all LOC prediction models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base for all LOC prediction models."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on provided data."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array [P(neg), P(pos)]."""

    @abstractmethod
    def feature_importances(self) -> np.ndarray | None:
        """Return feature importances or None if unsupported."""
