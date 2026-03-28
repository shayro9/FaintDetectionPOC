"""Entry point for LOC prediction training pipeline."""

import random
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from training.loo_trainer import LOOTrainer

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

    # Single group ID shared across all model runs so they appear together in wandb
    group_id = f"experiment_{uuid.uuid4().hex[:8]}"

    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            continue

        run = wandb.init(
            project=cfg["experiment"]["wandb_project"],
            name=f"{model_name}_loo",
            group=group_id,
            config=cfg,
            reinit="finish_previous",
        )

        trainer = LOOTrainer(cfg, run)
        trainer.run(patients, model_name, model_cfg)

        run.finish()


if __name__ == "__main__":
    main()
