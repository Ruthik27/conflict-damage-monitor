"""
eval.py — Evaluate a segmentation or change-detection checkpoint on the test set.

Loads any checkpoint that implements the 6-channel forward() interface
(DamageSegmentor or SiameseChangeDetector), runs sliding-window inference on
every test image, and prints per-class F1 / IoU.

Usage:
    PYTHONPATH=. python src/train/eval.py \
        --checkpoint /blue/.../checkpoints/last.ckpt \
        --model-type segmentor \
        --config configs/train_seg.yaml \
        --paths-cfg configs/paths.yaml

    PYTHONPATH=. python src/train/eval.py \
        --checkpoint /blue/.../checkpoints/unetpp-best.ckpt \
        --model-type change_detector \
        --config configs/train_change.yaml \
        --paths-cfg configs/paths.yaml \
        --split test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.xbd_seg_dataset import XBDSegDataset
from src.utils.metrics import DamageMetrics
from src.utils.tiling import sliding_window_inference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_model(checkpoint: Path, model_type: str, cfg: dict) -> torch.nn.Module:
    if model_type == "segmentor":
        from src.models.segmentor import DamageSegmentor
        model = DamageSegmentor.load_from_checkpoint(checkpoint, cfg=cfg)
    elif model_type == "change_detector":
        from src.models.change_detector import SiameseChangeDetector
        model = SiameseChangeDetector.load_from_checkpoint(checkpoint, cfg=cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'segmentor' or 'change_detector'.")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate damage segmentation checkpoint.")
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--model-type",  required=True, choices=["segmentor", "change_detector"])
    parser.add_argument("--config",      default="configs/train_seg.yaml", type=Path)
    parser.add_argument("--paths-cfg",   default="configs/paths.yaml",     type=Path)
    parser.add_argument("--split",       default="test", choices=["val", "test"])
    parser.add_argument("--tile-size",   default=512, type=int)
    parser.add_argument("--overlap",     default=128, type=int)
    args = parser.parse_args()

    env_file = Path("configs/.env")
    if env_file.exists():
        load_dotenv(env_file)

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = _load_model(args.checkpoint, args.model_type, cfg).to(device)
    num_classes = cfg["model"]["num_classes"]

    ds = XBDSegDataset(paths_cfg=args.paths_cfg, split=args.split, include_tier3=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    logger.info("Evaluating %d samples from %s split", len(ds), args.split)

    metrics = DamageMetrics(device=device)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference"):
            x, y = x.to(device), y.to(device)
            logits = sliding_window_inference(
                model, x,
                tile_size=args.tile_size,
                overlap=args.overlap,
                num_classes=num_classes,
            )
            preds = logits.argmax(dim=0)   # (H, W)
            metrics.update(preds, y.squeeze(0))

    print(metrics.summary())

    results = metrics.compute()
    logger.info("macro F1 : %.4f", results["macro_f1"])
    logger.info("macro IoU: %.4f", results["macro_iou"])


if __name__ == "__main__":
    main()
