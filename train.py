"""Entry point for LOC prediction training pipeline."""

import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from loc_prediction.training.loo_trainer import LOOTrainer

# Implemented in the data branch — loads per-patient windowed arrays
from data_generation.loader import load_patients


def main() -> None:
    """Load config, seed RNGs, and run LOO training for each enabled model."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    patients = load_patients(cfg["data"]["path"])

    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            continue

        run = wandb.init(
            project=cfg["experiment"]["wandb_project"],
            name=f"{model_name}_loo",
            config=cfg,
            reinit=True,
        )

        trainer = LOOTrainer(cfg, run)
        trainer.run(patients, model_name, model_cfg)

        run.finish()


if __name__ == "__main__":
    main()
