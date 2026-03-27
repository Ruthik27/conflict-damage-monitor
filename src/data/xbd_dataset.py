"""
xBD Dataset loader for building damage classification.

Expected directory layout under paths.data.raw/xbd/:

    {split}/
        images/
            {disaster}_{id}_pre_disaster.tif
            {disaster}_{id}_post_disaster.tif
        labels/
            {disaster}_{id}_post_disaster.json

Post-disaster label JSON schema:
    {
        "features": {
            "xy": [
                {
                    "properties": {"subtype": "<damage_class>", "uid": "..."},
                    "wkt": "POLYGON (...)"
                },
                ...
            ]
        }
    }

Scene-level label: most severe damage class found among all building polygons
in the post-disaster JSON. Falls back to 0 (no-damage) when no valid labels
exist or when the JSON is absent.

Damage class mapping:
    0  no-damage
    1  minor-damage
    2  major-damage
    3  destroyed
    --  un-classified (skipped)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE: int = 512

DAMAGE_LABEL_MAP: dict[str, int] = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

# Maps the public split names to xBD directory names on disk.
SPLIT_DIR_MAP: dict[str, str] = {
    "train": "train",
    "val": "hold",
    "test": "test",
}


# ── augmentation pipelines ────────────────────────────────────────────────────

def _build_train_transforms() -> A.Compose:
    """Spatial + photometric augmentations applied during training only."""
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        ],
        is_check_shapes=False,
    )


def _build_eval_transforms() -> A.Compose:
    """Deterministic resize only — used for val and test splits."""
    return A.Compose(
        [A.Resize(IMAGE_SIZE, IMAGE_SIZE)],
        is_check_shapes=False,
    )


# ── dataset ───────────────────────────────────────────────────────────────────

class XBDDataset(Dataset):
    """PyTorch Dataset for the xBD building damage classification task.

    Each sample is a pair of pre/post-disaster GeoTIFF chips concatenated into
    a single (6, 512, 512) float32 tensor (channels 0-2 = pre, 3-5 = post) and
    a scalar integer label in [0, 3].

    Args:
        paths_cfg:      Path to configs/paths.yaml. Resolved relative to the
                        current working directory when not absolute.
        split:          Dataset partition — one of "train", "val", or "test".
        transforms:     Custom albumentations Compose pipeline.  When None the
                        default train or eval pipeline is used automatically.
        include_tier3:  Include xBD tier-3 images when split == "train".
                        Ignored for val/test splits.
    """

    def __init__(
        self,
        paths_cfg: str | Path = "configs/paths.yaml",
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        include_tier3: bool = True,
    ) -> None:
        if split not in SPLIT_DIR_MAP:
            raise ValueError(
                f"split must be one of {list(SPLIT_DIR_MAP)}, got {split!r}"
            )

        self.split = split
        self.is_train = split == "train"

        xbd_root = self._resolve_xbd_root(Path(paths_cfg))

        split_dirs: list[Path] = [xbd_root / SPLIT_DIR_MAP[split]]
        if self.is_train and include_tier3:
            tier3 = xbd_root / "tier3"
            if tier3.exists():
                split_dirs.append(tier3)

        self._samples: list[tuple[Path, Path, Path]] = []
        for d in split_dirs:
            found = self._discover_pairs(d)
            self._samples.extend(found)

        logger.info(
            "XBDDataset | split=%s | samples=%d", split, len(self._samples)
        )

        if transforms is not None:
            self.transforms = transforms
        elif self.is_train:
            self.transforms = _build_train_transforms()
        else:
            self.transforms = _build_eval_transforms()

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pre_tif, post_tif, label_json = self._samples[idx]

        pre_img = self._read_geotiff(pre_tif)    # (H, W, 3) uint8
        post_img = self._read_geotiff(post_tif)  # (H, W, 3) uint8

        # Concatenate along channel axis so spatial transforms are applied
        # identically to both images.
        combined = np.concatenate([pre_img, post_img], axis=2)  # (H, W, 6)
        combined = self.transforms(image=combined)["image"]     # (H, W, 6)

        # HWC → CHW, normalise to [0, 1]
        tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float() / 255.0

        label = self._load_scene_label(label_json)
        return tensor, label

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_xbd_root(paths_cfg: Path) -> Path:
        with paths_cfg.open() as fh:
            cfg = yaml.safe_load(fh)
        return Path(cfg["data"]["raw"]) / "xbd"

    @staticmethod
    def _discover_pairs(split_dir: Path) -> list[tuple[Path, Path, Path]]:
        """Return (pre_tif, post_tif, label_json) triplets under *split_dir*."""
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            logger.warning("images/ directory not found: %s", images_dir)
            return []

        pairs: list[tuple[Path, Path, Path]] = []
        for pre_tif in sorted(images_dir.glob("*_pre_disaster.tif")):
            stem = pre_tif.stem.replace("_pre_disaster", "")
            post_tif = images_dir / f"{stem}_post_disaster.tif"
            label_json = labels_dir / f"{stem}_post_disaster.json"

            if not post_tif.exists():
                logger.debug("Missing post image for %s — skipping", stem)
                continue
            if not label_json.exists():
                logger.debug("Missing label JSON for %s — skipping", stem)
                continue

            pairs.append((pre_tif, post_tif, label_json))

        return pairs

    @staticmethod
    def _read_geotiff(path: Path) -> np.ndarray:
        """Read up to 3 bands of a GeoTIFF and return a (H, W, 3) uint8 array.

        Handles single-band grayscale (padded to 3 channels) and uint16 images
        (linearly rescaled to [0, 255]).
        """
        with rasterio.open(path) as src:
            n_bands = min(src.count, 3)
            bands = src.read(list(range(1, n_bands + 1)))  # (C, H, W)

        if n_bands < 3:
            pad = np.zeros(
                (3 - n_bands, bands.shape[1], bands.shape[2]), dtype=bands.dtype
            )
            bands = np.concatenate([bands, pad], axis=0)

        if bands.dtype != np.uint8:
            bands = _scale_to_uint8(bands)

        return bands.transpose(1, 2, 0)  # CHW → HWC

    @staticmethod
    def _load_scene_label(label_json: Path) -> int:
        """Parse an xBD post-disaster JSON and return the scene-level label.

        Scene-level label = most severe damage class present.
        Un-classified buildings are skipped.
        Returns 0 (no-damage) when no valid annotations exist.
        """
        with label_json.open() as fh:
            data = json.load(fh)

        labels: list[int] = []
        for feature in data.get("features", {}).get("xy", []):
            subtype = feature.get("properties", {}).get("subtype", "")
            if subtype in DAMAGE_LABEL_MAP:
                labels.append(DAMAGE_LABEL_MAP[subtype])

        return max(labels) if labels else 0


# ── utilities ─────────────────────────────────────────────────────────────────

def _scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale *arr* to the uint8 range [0, 255]."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.clip(0.0, 255.0).astype(np.uint8)
