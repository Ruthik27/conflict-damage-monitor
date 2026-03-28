"""
test_api.py — FastAPI TestClient smoke tests.

These tests mock the heavy inference call so no GPU or real data is needed.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a TestClient with inference and DB mocked out."""
    # Mock predict_geotiff to avoid loading a real model
    fake_result = {
        "job_id":      str(uuid.uuid4()),
        "mask":        np.zeros((64, 64), dtype=np.uint8),
        "class_stats": [
            {"class_id": 0, "class_name": "background",   "pixel_count": 4096, "pixel_pct": 100.0},
            {"class_id": 1, "class_name": "no-damage",     "pixel_count": 0,    "pixel_pct": 0.0},
            {"class_id": 2, "class_name": "minor-damage",  "pixel_count": 0,    "pixel_pct": 0.0},
            {"class_id": 3, "class_name": "major-damage",  "pixel_count": 0,    "pixel_pct": 0.0},
            {"class_id": 4, "class_name": "destroyed",     "pixel_count": 0,    "pixel_pct": 0.0},
        ],
        "geojson":     {"type": "FeatureCollection", "features": []},
        "crs":         "EPSG:4326",
        "transform":   [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    }

    # Patch at import location used by main.py
    with patch("src.api.main.predict_geotiff", return_value=fake_result), \
         patch("src.api.main.save_result",     return_value=False), \
         patch("src.api.main._resolve_checkpoint", return_value=Path("/fake/ckpt.ckpt")), \
         patch("src.api.main._load_cfg", return_value={
             "model": {"num_classes": 5, "encoder": "efficientnet-b4",
                       "encoder_weights": "imagenet", "in_channels": 6}
         }):
        from src.api.main import app
        yield TestClient(app)


# ── tests ──────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "device" in data


def test_predict_missing_files(client):
    """Requesting non-existent paths should return 400."""
    resp = client.post("/predict", json={
        "pre_path":  "/nonexistent/pre.tif",
        "post_path": "/nonexistent/post.tif",
        "model_type": "segmentor",
    })
    assert resp.status_code == 400


def test_predict_path_ok(client, tmp_path):
    """With real files on disk the endpoint should return 200 with class_stats."""
    pre  = tmp_path / "pre.tif";  pre.write_bytes(b"\x00" * 10)
    post = tmp_path / "post.tif"; post.write_bytes(b"\x00" * 10)

    resp = client.post("/predict", json={
        "pre_path":  str(pre),
        "post_path": str(post),
        "model_type": "segmentor",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"]
    assert len(data["class_stats"]) == 5
    assert data["geojson"]["type"] == "FeatureCollection"


def test_results_not_found(client):
    resp = client.get("/results/does-not-exist-xyz")
    assert resp.status_code == 404


def test_openapi_schema(client):
    """Ensure OpenAPI schema is accessible."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
