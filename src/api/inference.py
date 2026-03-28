"""
inference.py — Model loading and GeoTIFF prediction for the FastAPI server.

Supports two model types loaded from Lightning checkpoints:
  - segmentor       : DamageSegmentor (6-channel concat input)
  - change_detector : SiameseChangeDetector (separate pre/post 3-channel inputs)

predict_geotiff() is the main entry point:
  Returns (mask_array, class_stats, geojson_dict | None)
"""
from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
import yaml

from src.utils.tiling import sliding_window_inference
from src.utils.metrics import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, torch.nn.Module] = {}   # keyed by checkpoint path


# ── model loading ──────────────────────────────────────────────────────────────

def _load_model(checkpoint: Path, model_type: str, cfg: dict) -> torch.nn.Module:
    key = str(checkpoint)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if model_type == "segmentor":
        from src.models.segmentor import DamageSegmentor
        model = DamageSegmentor.load_from_checkpoint(str(checkpoint), cfg=cfg)
    elif model_type == "change_detector":
        from src.models.change_detector import SiameseChangeDetector
        model = SiameseChangeDetector.load_from_checkpoint(str(checkpoint), cfg=cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.eval()
    _MODEL_CACHE[key] = model
    logger.info("Loaded %s from %s", model_type, checkpoint)
    return model


# ── GeoTIFF reading ────────────────────────────────────────────────────────────

def _read_geotiff(path: Path) -> tuple[np.ndarray, Any, str]:
    """Return (C, H, W) float32 array, transform, crs."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W), raw DN
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
    # Normalize to [0, 1] assuming uint8 imagery
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr, transform, crs


# ── polygonisation ─────────────────────────────────────────────────────────────

def _mask_to_geojson(mask: np.ndarray, transform: Affine, crs: str) -> dict[str, Any]:
    """Convert (H, W) uint8 class mask → GeoJSON FeatureCollection."""
    features = []
    for class_id in range(1, NUM_CLASSES):   # skip background
        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        for geom, value in shapes(binary, mask=binary, transform=transform):
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                },
            })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }


# ── main predict function ──────────────────────────────────────────────────────

def predict_geotiff(
    pre_path: Path,
    post_path: Path,
    model_type: str,
    checkpoint: Path,
    cfg: dict,
    tile_size: int = 512,
    overlap: int = 128,
    return_geojson: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run inference on a pre/post pair and return structured results.

    Returns:
        {
          "job_id":      str,
          "mask":        np.ndarray (H, W) uint8 class indices,
          "class_stats": list[dict],
          "geojson":     dict | None,
          "crs":         str,
          "transform":   list[float],
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint, model_type, cfg).to(device)

    pre_arr, transform, crs = _read_geotiff(pre_path)
    post_arr, _, _ = _read_geotiff(post_path)

    if model_type == "segmentor":
        # Concatenate pre+post along channel dim → (1, 6, H, W)
        combined = np.concatenate([pre_arr, post_arr], axis=0)   # (6, H, W)
        x = torch.from_numpy(combined).unsqueeze(0).to(device)   # (1, 6, H, W)
        with torch.no_grad():
            logits = sliding_window_inference(
                model, x,
                tile_size=tile_size,
                overlap=overlap,
                num_classes=cfg["model"]["num_classes"],
            )   # (num_classes, H, W)
    else:
        # change_detector: separate pre/post inputs
        pre_t  = torch.from_numpy(pre_arr).unsqueeze(0).to(device)   # (1,3,H,W)
        post_t = torch.from_numpy(post_arr).unsqueeze(0).to(device)  # (1,3,H,W)

        class _WrapForTiling(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self._pre_t = pre_t
                self._tile = tile_size

            def forward(self, post_tile):
                B, C, h, w = post_tile.shape
                # Find tile position (all tiles come in sequentially; just use model directly)
                return self.m(self._pre_t[:, :, :h, :w], post_tile)

        # For change_detector use the model directly on full image if fits in VRAM
        # Else fall back to CPU tiling
        with torch.no_grad():
            if pre_arr.shape[1] <= 1024 and pre_arr.shape[2] <= 1024:
                logits = model(pre_t, post_t).squeeze(0)  # (num_classes, H, W)
            else:
                # Sliding window: tile only post, reuse full pre (simple approach)
                wrapped = _WrapForTiling(model)
                logits = sliding_window_inference(
                    wrapped, post_t,
                    tile_size=tile_size,
                    overlap=overlap,
                    num_classes=cfg["model"]["num_classes"],
                )

    mask = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)   # (H, W)
    total_px = mask.size

    class_stats = []
    for i, name in enumerate(CLASS_NAMES):
        count = int((mask == i).sum())
        class_stats.append({
            "class_id": i,
            "class_name": name,
            "pixel_count": count,
            "pixel_pct": round(100.0 * count / total_px, 2),
        })

    geojson = None
    if return_geojson:
        geojson = _mask_to_geojson(mask, transform, crs)

    transform_list = list(transform)[:6]

    return {
        "job_id": str(uuid.uuid4()),
        "mask": mask,
        "class_stats": class_stats,
        "geojson": geojson,
        "crs": crs,
        "transform": transform_list,
    }
