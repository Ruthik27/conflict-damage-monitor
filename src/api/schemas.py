"""
schemas.py — Pydantic request/response models for the inference API.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Body for /predict when supplying file paths instead of uploads."""
    pre_path: str = Field(..., description="Absolute path to pre-disaster GeoTIFF")
    post_path: str = Field(..., description="Absolute path to post-disaster GeoTIFF")
    model_type: str = Field("segmentor", description="'segmentor' or 'change_detector'")
    checkpoint: str | None = Field(None, description="Override checkpoint path")
    tile_size: int = Field(512, ge=128, le=1024)
    overlap: int = Field(128, ge=0, le=256)
    return_geojson: bool = Field(True, description="Polygonise mask and return GeoJSON")


class ClassStats(BaseModel):
    class_id: int
    class_name: str
    pixel_count: int
    pixel_pct: float


class PredictResponse(BaseModel):
    job_id: str
    model_type: str
    geojson: dict[str, Any] | None = None   # FeatureCollection or None
    class_stats: list[ClassStats]
    macro_f1: float | None = None
    macro_iou: float | None = None
    crs: str | None = None
    transform: list[float] | None = None    # 6-element GDAL affine


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
