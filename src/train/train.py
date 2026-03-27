"""
train.py — Training entrypoint for the EfficientNet-B4 damage classifier.

Usage:
    python src/train/train.py --config configs/train.yaml --paths-cfg configs/paths.yaml
    python src/train/train.py --config configs/train.yaml --paths-cfg configs/paths.yaml --fast-dev-run
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data.xbd_dataset import XBDDataset
from src.models.classifier import DamageClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── data helpers ──────────────────────────────────────────────────────────────

def _build_loaders(cfg: dict, paths_cfg: Path) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    tcfg = cfg["trainer"]

    train_ds = XBDDataset(
        paths_cfg=paths_cfg,
        split="train",
        include_tier3=dcfg["include_tier3"],
    )
    val_ds = XBDDataset(
        paths_cfg=paths_cfg,
        split="val",
        include_tier3=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        pin_memory=dcfg["pin_memory"],
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=dcfg["num_workers"],
        pin_memory=dcfg["pin_memory"],
        persistent_workers=True,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    return train_loader, val_loader


# ── callbacks ─────────────────────────────────────────────────────────────────

def _build_callbacks(cfg: dict, paths_cfg: Path) -> list:
    with paths_cfg.open() as fh:
        pcfg = yaml.safe_load(fh)
    ckpt_dir = Path(pcfg["training"]["checkpoints"])

    ccfg = cfg["checkpoint"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ccfg["filename"],
        monitor=ccfg["monitor"],
        mode=ccfg["mode"],
        save_top_k=ccfg["save_top_k"],
        every_n_epochs=ccfg["every_n_epochs"],
        save_last=True,
    )
    return [checkpoint_cb, LearningRateMonitor(logging_interval="epoch"), RichProgressBar()]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 damage classifier.")
    parser.add_argument("--config",    default="configs/train.yaml",  type=Path)
    parser.add_argument("--paths-cfg", default="configs/paths.yaml",  type=Path)
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1 train + 1 val batch for smoke-testing")
    parser.add_argument("--resume",    default=None, type=Path,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load .env (WANDB_API_KEY, etc.) — must happen before any wandb import
    env_file = Path("configs/.env")
    if env_file.exists():
        load_dotenv(env_file)
        logger.info("Loaded env from %s", env_file)
    else:
        logger.warning(".env not found at %s — W&B may fail if key not set", env_file)

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    pl.seed_everything(42, workers=True)

    train_loader, val_loader = _build_loaders(cfg, args.paths_cfg)
    model = DamageClassifier(cfg)

    wcfg = cfg["wandb"]
    wandb_logger = WandbLogger(
        project=wcfg["project"],
        entity=wcfg.get("entity"),
        tags=wcfg.get("tags", []),
        log_model=wcfg.get("log_model", False),
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)

    tcfg = cfg["trainer"]
    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        precision=tcfg["precision"],
        gradient_clip_val=tcfg["gradient_clip_val"],
        accumulate_grad_batches=tcfg["accumulate_grad_batches"],
        val_check_interval=tcfg["val_check_interval"],
        log_every_n_steps=tcfg["log_every_n_steps"],
        callbacks=_build_callbacks(cfg, args.paths_cfg),
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=False,   # slightly faster on A100
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
