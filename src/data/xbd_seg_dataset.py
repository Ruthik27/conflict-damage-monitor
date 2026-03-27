"""
XBDSegDataset — pixel-level damage segmentation dataset.

Directory layout expected under paths.data.raw/xbd/{split}/:
    images/   {stem}_pre_disaster.png  + {stem}_post_disaster.png
    targets/  {stem}_post_disaster_target.png   (uint8, values 0-4)

Mask classes:
    0  background (no building)
    1  no-damage
    2  minor-damage
    3  major-damage
    4  destroyed

Train  : random 512×512 crop from 1024×1024, joint spatial augmentation
Val/Test: full 1024×1024 (tiled at inference by src/utils/tiling.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TRAIN_SIZE = 512
NUM_SEG_CLASSES = 5


def _build_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(TRAIN_SIZE, TRAIN_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ],
        additional_targets={"mask": "mask"},
        is_check_shapes=False,
    )


def _build_eval_transforms() -> A.Compose:
    return A.Compose(
        [],  # return full 1024×1024 unchanged
        additional_targets={"mask": "mask"},
        is_check_shapes=False,
    )


class XBDSegDataset(Dataset):
    """PyTorch Dataset for xBD pixel-level damage segmentation.

    Each sample returns:
        image : (6, H, W) float32 tensor  — channels 0-2 pre, 3-5 post, [0, 1]
        mask  : (H, W)    int64 tensor    — class indices 0-4
    """

    def __init__(
        self,
        paths_cfg: str | Path = "configs/paths.yaml",
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        include_tier3: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.is_train = split == "train"
        xbd_root = self._resolve_xbd_root(Path(paths_cfg))

        split_dir_map = {"train": "train", "val": "hold", "test": "test"}
        split_dirs = [xbd_root / split_dir_map[split]]
        if self.is_train and include_tier3:
            t3 = xbd_root / "tier3"
            if t3.exists():
                split_dirs.append(t3)

        self._samples: list[tuple[Path, Path, Path]] = []
        for d in split_dirs:
            self._samples.extend(self._discover_triplets(d))

        logger.info("XBDSegDataset | split=%s | samples=%d", split, len(self._samples))

        if transforms is not None:
            self.transforms = transforms
        elif self.is_train:
            self.transforms = _build_train_transforms()
        else:
            self.transforms = _build_eval_transforms()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pre_path, post_path, mask_path = self._samples[idx]

        pre  = np.array(Image.open(pre_path).convert("RGB"))   # (H, W, 3) uint8
        post = np.array(Image.open(post_path).convert("RGB"))  # (H, W, 3) uint8
        mask = np.array(Image.open(mask_path))                  # (H, W)    uint8

        image = np.concatenate([pre, post], axis=2)  # (H, W, 6)
        out = self.transforms(image=image, mask=mask)
        image, mask = out["image"], out["mask"]

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor  = torch.from_numpy(mask.copy()).long()
        return image_tensor, mask_tensor

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_xbd_root(paths_cfg: Path) -> Path:
        with paths_cfg.open() as fh:
            cfg = yaml.safe_load(fh)
        return Path(cfg["data"]["raw"]) / "xbd"

    @staticmethod
    def _discover_triplets(split_dir: Path) -> list[tuple[Path, Path, Path]]:
        images_dir  = split_dir / "images"
        targets_dir = split_dir / "targets"

        if not images_dir.exists():
            logger.warning("images/ not found: %s", images_dir)
            return []
        if not targets_dir.exists():
            logger.warning("targets/ not found: %s", targets_dir)
            return []

        triplets: list[tuple[Path, Path, Path]] = []
        for pre in sorted(
            [*images_dir.glob("*_pre_disaster.png"), *images_dir.glob("*_pre_disaster.tif")]
        ):
            ext  = pre.suffix
            stem = pre.stem.replace("_pre_disaster", "")
            post = images_dir  / f"{stem}_post_disaster{ext}"
            mask = targets_dir / f"{stem}_post_disaster_target.png"

            if not post.exists():
                logger.debug("Missing post image for %s — skipping", stem)
                continue
            if not mask.exists():
                logger.debug("Missing target mask for %s — skipping", stem)
                continue

            triplets.append((pre, post, mask))

        return triplets
