"""Unit tests for src/data/xbd_seg_dataset.py — no real xBD data required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data.xbd_seg_dataset import XBDSegDataset


# ── fixtures ──────────────────────────────────────────────────────────────────

def _write_png(path: Path, h: int = 1024, w: int = 1024, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:  # grayscale mask
        arr = np.random.randint(0, 5, (h, w), dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


def _write_paths_cfg(cfg_path: Path, raw_root: Path) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as fh:
        yaml.safe_dump({"data": {"raw": str(raw_root), "processed": "", "interim": ""}}, fh)


@pytest.fixture()
def seg_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "xbd"
    for split, n in [("train", 3), ("hold", 2)]:
        images_dir  = root / split / "images"
        targets_dir = root / split / "targets"
        for i in range(n):
            stem = f"hurricane_test_{i:08d}"
            _write_png(images_dir / f"{stem}_pre_disaster.png")
            _write_png(images_dir / f"{stem}_post_disaster.png")
            _write_png(targets_dir / f"{stem}_post_disaster_target.png", mode="L")
    return root


@pytest.fixture()
def paths_cfg(tmp_path: Path, seg_root: Path) -> Path:
    cfg_path = tmp_path / "configs" / "paths.yaml"
    _write_paths_cfg(cfg_path, seg_root.parent)
    return cfg_path


# ── length ────────────────────────────────────────────────────────────────────

def test_train_length(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    assert len(ds) == 3


def test_val_length(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
    assert len(ds) == 2


def test_missing_targets_dir_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "xbd"
    images_dir = root / "train" / "images"
    _write_png(images_dir / "ev_00000001_pre_disaster.png")
    _write_png(images_dir / "ev_00000001_post_disaster.png")
    # no targets/ dir
    cfg_path = tmp_path / "configs" / "paths.yaml"
    _write_paths_cfg(cfg_path, root.parent)
    ds = XBDSegDataset(paths_cfg=cfg_path, split="train", include_tier3=False)
    assert len(ds) == 0


# ── output shapes & dtypes ────────────────────────────────────────────────────

def test_train_image_shape(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    img, mask = ds[0]
    assert img.shape == (6, 512, 512)


def test_train_mask_shape(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    _, mask = ds[0]
    assert mask.shape == (512, 512)


def test_val_image_shape(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
    img, mask = ds[0]
    assert img.shape == (6, 1024, 1024)
    assert mask.shape == (1024, 1024)


def test_image_dtype(paths_cfg: Path) -> None:
    import torch
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    img, _ = ds[0]
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_mask_dtype(paths_cfg: Path) -> None:
    import torch
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    _, mask = ds[0]
    assert mask.dtype == torch.int64


def test_mask_values_in_range(paths_cfg: Path) -> None:
    ds = XBDSegDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
    for i in range(len(ds)):
        _, mask = ds[i]
        assert int(mask.min()) >= 0
        assert int(mask.max()) <= 4


# ── invalid split ─────────────────────────────────────────────────────────────

def test_invalid_split_raises(paths_cfg: Path) -> None:
    with pytest.raises(ValueError, match="split must be"):
        XBDSegDataset(paths_cfg=paths_cfg, split="invalid")
