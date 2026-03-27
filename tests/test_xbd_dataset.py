"""
Unit tests for src/data/xbd_dataset.py.

All tests use synthetic GeoTIFFs and JSON label files written to pytest's
tmp_path fixture — no real xBD data required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import albumentations as A
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.data.xbd_dataset import (
    DAMAGE_LABEL_MAP,
    XBDDataset,
    _scale_to_uint8,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _write_geotiff(path: Path, height: int = 600, width: int = 600, bands: int = 3) -> None:
    """Write a synthetic uint8 RGB GeoTIFF to *path*."""
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    transform = from_bounds(0.0, 0.0, 1.0, 1.0, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_label_json(path: Path, subtypes: list[str]) -> None:
    """Write a minimal xBD-format post-disaster label JSON to *path*."""
    features = [
        {
            "properties": {"subtype": s, "uid": f"uid_{i}"},
            "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        }
        for i, s in enumerate(subtypes)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump({"features": {"xy": features}}, fh)


def _write_paths_cfg(cfg_path: Path, raw_root: Path) -> None:
    """Write a minimal paths.yaml pointing raw data to *raw_root*."""
    cfg = {
        "data": {
            "raw": str(raw_root),
            "processed": str(raw_root / "processed"),
            "interim": str(raw_root / "interim"),
        }
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as fh:
        yaml.safe_dump(cfg, fh)


@pytest.fixture()
def xbd_root(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Build a minimal xBD directory tree under tmp_path with 3 train samples
    and 2 val (hold) samples, then return the xbd/ root.

    Structure:
        tmp_path/
          raw/
            xbd/
              train/
                images/   (3 pre + 3 post GeoTIFFs)
                labels/   (3 post label JSONs)
              hold/
                images/   (2 pre + 2 post GeoTIFFs)
                labels/   (2 post label JSONs)
    """
    root = tmp_path / "raw" / "xbd"

    # --- train split (3 samples, varying damage classes) ---------------------
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)

    train_samples = [
        ("hurricane_michael_00000001", ["no-damage", "no-damage"]),
        ("hurricane_michael_00000002", ["minor-damage", "major-damage"]),
        ("hurricane_michael_00000003", ["destroyed", "un-classified"]),
    ]
    for stem, subtypes in train_samples:
        _write_geotiff(train_images / f"{stem}_pre_disaster.tif")
        _write_geotiff(train_images / f"{stem}_post_disaster.tif")
        _write_label_json(train_labels / f"{stem}_post_disaster.json", subtypes)

    # --- hold (val) split (2 samples) ----------------------------------------
    hold_images = root / "hold" / "images"
    hold_labels = root / "hold" / "labels"
    hold_images.mkdir(parents=True)
    hold_labels.mkdir(parents=True)

    hold_samples = [
        ("hurricane_harvey_00000001", ["no-damage"]),
        ("hurricane_harvey_00000002", ["major-damage", "destroyed"]),
    ]
    for stem, subtypes in hold_samples:
        _write_geotiff(hold_images / f"{stem}_pre_disaster.tif")
        _write_geotiff(hold_images / f"{stem}_post_disaster.tif")
        _write_label_json(hold_labels / f"{stem}_post_disaster.json", subtypes)

    yield root


@pytest.fixture()
def paths_cfg(tmp_path: Path, xbd_root: Path) -> Path:
    """Write configs/paths.yaml and return its path."""
    cfg_path = tmp_path / "configs" / "paths.yaml"
    _write_paths_cfg(cfg_path, xbd_root.parent)  # raw_root = xbd_root.parent = raw/
    return cfg_path


# ── dataset length ────────────────────────────────────────────────────────────

class TestDatasetLength:
    def test_train_length(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        assert len(ds) == 3

    def test_val_length(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        assert len(ds) == 2

    def test_empty_split_returns_zero(self, paths_cfg: Path) -> None:
        # test/ directory was not created — dataset should be empty, not crash.
        ds = XBDDataset(paths_cfg=paths_cfg, split="test", include_tier3=False)
        assert len(ds) == 0


# ── output shapes ─────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_image_shape_train(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        image, _ = ds[0]
        assert image.shape == (6, 512, 512), f"Expected (6,512,512), got {image.shape}"

    def test_image_shape_val(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        image, _ = ds[0]
        assert image.shape == (6, 512, 512), f"Expected (6,512,512), got {image.shape}"

    def test_image_dtype_float32(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        image, _ = ds[0]
        assert image.dtype == __import__("torch").float32

    def test_image_values_normalised(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        image, _ = ds[0]
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0


# ── label correctness ─────────────────────────────────────────────────────────

class TestLabels:
    @pytest.mark.parametrize(
        ("idx", "expected_label"),
        [
            (0, 0),  # ["no-damage", "no-damage"]     → max = 0
            (1, 2),  # ["minor-damage", "major-damage"] → max = 2
            (2, 3),  # ["destroyed", "un-classified"]  → max = 3 (un-classified skipped)
        ],
    )
    def test_train_labels(
        self, paths_cfg: Path, idx: int, expected_label: int
    ) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        _, label = ds[idx]
        assert label == expected_label

    def test_all_labels_in_valid_range(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label <= 3, f"Label {label} out of [0, 3] at index {i}"

    def test_val_labels_in_valid_range(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label <= 3

    def test_no_annotations_falls_back_to_zero(self, paths_cfg: Path, xbd_root: Path) -> None:
        """A label JSON with no valid features should return label 0."""
        images_dir = xbd_root / "hold" / "images"
        labels_dir = xbd_root / "hold" / "labels"
        stem = "hurricane_harvey_00000099"
        _write_geotiff(images_dir / f"{stem}_pre_disaster.tif")
        _write_geotiff(images_dir / f"{stem}_post_disaster.tif")
        _write_label_json(labels_dir / f"{stem}_post_disaster.json", [])

        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        # Find the empty-label sample (it's the last one alphabetically)
        empty_idx = next(
            i for i, s in enumerate(ds._samples)
            if "00000099" in str(s[0])
        )
        _, label = ds[empty_idx]
        assert label == 0

    def test_unclassified_only_falls_back_to_zero(
        self, paths_cfg: Path, xbd_root: Path
    ) -> None:
        images_dir = xbd_root / "hold" / "images"
        labels_dir = xbd_root / "hold" / "labels"
        stem = "hurricane_harvey_00000098"
        _write_geotiff(images_dir / f"{stem}_pre_disaster.tif")
        _write_geotiff(images_dir / f"{stem}_post_disaster.tif")
        _write_label_json(
            labels_dir / f"{stem}_post_disaster.json",
            ["un-classified", "un-classified"],
        )

        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        unclassified_idx = next(
            i for i, s in enumerate(ds._samples) if "00000098" in str(s[0])
        )
        _, label = ds[unclassified_idx]
        assert label == 0


# ── augmentation behaviour ────────────────────────────────────────────────────

class TestAugmentation:
    def test_train_uses_augmentation_pipeline(self, paths_cfg: Path) -> None:
        """Train dataset should include stochastic transforms beyond Resize."""
        ds = XBDDataset(paths_cfg=paths_cfg, split="train", include_tier3=False)
        transform_types = {type(t).__name__ for t in ds.transforms.transforms}
        # At least one non-deterministic transform expected
        stochastic = {"HorizontalFlip", "VerticalFlip", "RandomRotate90",
                      "RandomBrightnessContrast", "GaussNoise", "CoarseDropout"}
        assert stochastic & transform_types, (
            f"No stochastic transforms found in train pipeline: {transform_types}"
        )

    def test_val_uses_only_resize(self, paths_cfg: Path) -> None:
        """Val dataset should only resize — no stochastic transforms."""
        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        transform_types = {type(t).__name__ for t in ds.transforms.transforms}
        assert transform_types == {"Resize"}, (
            f"Val pipeline should only contain Resize, got: {transform_types}"
        )

    def test_test_uses_only_resize(self, paths_cfg: Path) -> None:
        ds = XBDDataset(paths_cfg=paths_cfg, split="test", include_tier3=False)
        transform_types = {type(t).__name__ for t in ds.transforms.transforms}
        assert transform_types == {"Resize"}

    def test_custom_transform_is_respected(self, paths_cfg: Path) -> None:
        """Passing a custom transform overrides the default for any split."""
        identity = A.Compose([A.Resize(512, 512)], is_check_shapes=False)
        ds = XBDDataset(
            paths_cfg=paths_cfg,
            split="train",
            transforms=identity,
            include_tier3=False,
        )
        assert ds.transforms is identity

    def test_val_output_is_deterministic(self, paths_cfg: Path) -> None:
        """Running the same val sample twice must return identical tensors."""
        ds = XBDDataset(paths_cfg=paths_cfg, split="val", include_tier3=False)
        img1, label1 = ds[0]
        img2, label2 = ds[0]
        assert __import__("torch").equal(img1, img2)
        assert label1 == label2


# ── PNG extension support ─────────────────────────────────────────────────────

class TestPngSupport:
    def test_png_images_are_discovered(self, tmp_path: Path) -> None:
        """Dataset must find pre/post pairs when images are .png (real xBD format)."""
        root = tmp_path / "raw" / "xbd"
        images_dir = root / "train" / "images"
        labels_dir = root / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        stems = ["hurricane_harvey_00000001", "hurricane_harvey_00000002"]
        for stem in stems:
            _write_geotiff(images_dir / f"{stem}_pre_disaster.png")
            _write_geotiff(images_dir / f"{stem}_post_disaster.png")
            _write_label_json(labels_dir / f"{stem}_post_disaster.json", ["major-damage"])

        cfg_path = tmp_path / "configs" / "paths.yaml"
        _write_paths_cfg(cfg_path, root.parent)

        ds = XBDDataset(paths_cfg=cfg_path, split="train", include_tier3=False)
        assert len(ds) == 2, f"Expected 2 PNG samples, got {len(ds)}"

    def test_png_output_shape_and_dtype(self, tmp_path: Path) -> None:
        """PNG samples must return the same (6, 512, 512) float32 tensor as TIF."""
        root = tmp_path / "raw" / "xbd"
        images_dir = root / "train" / "images"
        labels_dir = root / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        stem = "socal_fire_00000001"
        _write_geotiff(images_dir / f"{stem}_pre_disaster.png")
        _write_geotiff(images_dir / f"{stem}_post_disaster.png")
        _write_label_json(labels_dir / f"{stem}_post_disaster.json", ["destroyed"])

        cfg_path = tmp_path / "configs" / "paths.yaml"
        _write_paths_cfg(cfg_path, root.parent)

        ds = XBDDataset(paths_cfg=cfg_path, split="train", include_tier3=False)
        img, label = ds[0]
        assert img.shape == (6, 512, 512)
        assert img.dtype == __import__("torch").float32
        assert label == 3  # destroyed

    def test_mixed_tif_and_png_both_discovered(self, tmp_path: Path) -> None:
        """A split with both .tif and .png files must discover all pairs."""
        root = tmp_path / "raw" / "xbd"
        images_dir = root / "train" / "images"
        labels_dir = root / "train" / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for stem, ext in [("event_a_00000001", ".tif"), ("event_b_00000001", ".png")]:
            _write_geotiff(images_dir / f"{stem}_pre_disaster{ext}")
            _write_geotiff(images_dir / f"{stem}_post_disaster{ext}")
            _write_label_json(labels_dir / f"{stem}_post_disaster.json", ["no-damage"])

        cfg_path = tmp_path / "configs" / "paths.yaml"
        _write_paths_cfg(cfg_path, root.parent)

        ds = XBDDataset(paths_cfg=cfg_path, split="train", include_tier3=False)
        assert len(ds) == 2, f"Expected 2 samples (1 tif + 1 png), got {len(ds)}"


# ── utility function ──────────────────────────────────────────────────────────

class TestScaleToUint8:
    def test_uint16_is_rescaled(self) -> None:
        arr = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        result = _scale_to_uint8(arr)
        assert result.dtype == np.uint8
        assert int(result.min()) == 0
        assert int(result.max()) == 255

    def test_constant_array_does_not_raise(self) -> None:
        arr = np.full((3, 4, 4), 1000, dtype=np.uint16)
        result = _scale_to_uint8(arr)
        assert result.dtype == np.uint8


# ── invalid input handling ────────────────────────────────────────────────────

class TestValidation:
    def test_invalid_split_raises(self, paths_cfg: Path) -> None:
        with pytest.raises(ValueError, match="split must be one of"):
            XBDDataset(paths_cfg=paths_cfg, split="invalid")

    def test_damage_label_map_completeness(self) -> None:
        expected_keys = {"no-damage", "minor-damage", "major-damage", "destroyed"}
        assert set(DAMAGE_LABEL_MAP.keys()) == expected_keys

    def test_damage_label_values_are_0_to_3(self) -> None:
        assert sorted(DAMAGE_LABEL_MAP.values()) == [0, 1, 2, 3]
