"""GRU-based RNN model wrapping a LightningModule."""

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel


class GRULightning(L.LightningModule):
    """GRU sequence classifier as a LightningModule."""

    def __init__(self, cfg: dict, pos_weight: float) -> None:
        """Initialize GRU from config dict and pos_weight."""
        super().__init__()
        self.save_hyperparameters()
        self.lr = cfg["lr"]
        self.gru = nn.GRU(
            input_size=4,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg["hidden_size"], 1)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N,) from input (N, T, 4)."""
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute BCEWithLogitsLoss and log to wandb."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        """Return Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class RNNModel(BaseModel):
    """Wrapper around GRULightning implementing the BaseModel interface."""

    def __init__(self, cfg: dict, pos_weight: float) -> None:
        """Initialize RNN model from config dict and pos_weight."""
        self.cfg = cfg
        self.pos_weight = pos_weight
        self.module: GRULightning | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the GRU with PyTorch Lightning, logging to wandb."""
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

        self.module = GRULightning(self.cfg, self.pos_weight)
        trainer = L.Trainer(
            max_epochs=self.cfg["max_epochs"],
            logger=WandbLogger(),
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(self.module, loader)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array [1-p, p]."""
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.tensor(X, dtype=torch.float32))
            p = torch.sigmoid(logits).numpy()
        return np.column_stack([1.0 - p, p])

    def feature_importances(self) -> None:
        """RNN does not support feature importances."""
        return None
