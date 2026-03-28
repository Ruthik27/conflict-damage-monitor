"""
main.py — FastAPI inference server for conflict damage monitoring.

Endpoints:
  GET  /health                 — liveness check
  POST /predict                — run inference on a pre/post GeoTIFF pair (path-based)
  POST /predict/upload         — same but accepts multipart file uploads
  GET  /results/{job_id}       — retrieve a stored result from PostGIS

Environment variables (via configs/.env or shell):
  CHECKPOINT_SEG       path to segmentor .ckpt (default: latest in checkpoints/)
  CHECKPOINT_CHANGE    path to change_detector .ckpt
  CONFIG_SEG           path to train_seg.yaml    (default: configs/train_seg.yaml)
  CONFIG_CHANGE        path to train_change.yaml (default: configs/train_change.yaml)
  PATHS_CFG            path to paths.yaml        (default: configs/paths.yaml)
  DATABASE_URL         optional PostGIS DSN
  DEVICE               "cuda" | "cpu" (auto-detected if unset)

Run locally:
  PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.db import get_result, save_result
from src.api.inference import predict_geotiff
from src.api.schemas import (
    ClassStats,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# ── setup ──────────────────────────────────────────────────────────────────────
load_dotenv(Path("configs/.env"), override=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conflict Damage Monitor API",
    description="Satellite building damage classification — UNet++ / Siamese change detector",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(model_type: str, override: str | None) -> Path:
    if override:
        ck = Path(override)
        if not ck.exists():
            raise HTTPException(400, f"Checkpoint not found: {override}")
        return ck

    env_key = "CHECKPOINT_SEG" if model_type == "segmentor" else "CHECKPOINT_CHANGE"
    env_val = os.getenv(env_key)
    if env_val:
        ck = Path(env_val)
        if ck.exists():
            return ck

    # Last-resort: pick latest .ckpt in the checkpoints directory
    with open("configs/paths.yaml") as fh:
        paths = yaml.safe_load(fh)
    ckpt_dir = Path(paths["training"]["checkpoints"])
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise HTTPException(503, "No checkpoint found. Set CHECKPOINT_SEG or CHECKPOINT_CHANGE.")
    return candidates[0]


def _load_cfg(model_type: str) -> dict:
    env_key = "CONFIG_SEG" if model_type == "segmentor" else "CONFIG_CHANGE"
    cfg_path = Path(os.getenv(env_key, f"configs/train_{model_type[:3]}.yaml"))
    # Fallback names
    fallbacks = {
        "segmentor":       Path("configs/train_seg.yaml"),
        "change_detector": Path("configs/train_change.yaml"),
    }
    if not cfg_path.exists():
        cfg_path = fallbacks.get(model_type, Path("configs/train_seg.yaml"))
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def _device() -> torch.device:
    spec = os.getenv("DEVICE", "")
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_response(result: dict, model_type: str) -> PredictResponse:
    return PredictResponse(
        job_id=result["job_id"],
        model_type=model_type,
        geojson=result.get("geojson"),
        class_stats=[ClassStats(**s) for s in result["class_stats"]],
        crs=result.get("crs"),
        transform=result.get("transform"),
    )


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    device = _device()
    ckpt_env = os.getenv("CHECKPOINT_SEG") or os.getenv("CHECKPOINT_CHANGE")
    model_loaded = bool(ckpt_env and Path(ckpt_env).exists())
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=str(device),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Run inference given absolute file paths on the server filesystem."""
    pre_path  = Path(req.pre_path)
    post_path = Path(req.post_path)
    for p in (pre_path, post_path):
        if not p.exists():
            raise HTTPException(400, f"File not found: {p}")

    checkpoint = _resolve_checkpoint(req.model_type, req.checkpoint)
    cfg        = _load_cfg(req.model_type)
    device     = _device()

    result = predict_geotiff(
        pre_path=pre_path,
        post_path=post_path,
        model_type=req.model_type,
        checkpoint=checkpoint,
        cfg=cfg,
        tile_size=req.tile_size,
        overlap=req.overlap,
        return_geojson=req.return_geojson,
        device=device,
    )

    save_result(
        job_id=result["job_id"],
        model_type=req.model_type,
        pre_path=str(pre_path),
        post_path=str(post_path),
        class_stats=result["class_stats"],
        geojson=result.get("geojson"),
    )

    return _build_response(result, req.model_type)


@app.post("/predict/upload", response_model=PredictResponse, tags=["inference"])
async def predict_upload(
    pre_file:   UploadFile = File(..., description="Pre-disaster GeoTIFF"),
    post_file:  UploadFile = File(..., description="Post-disaster GeoTIFF"),
    model_type: str        = Form("segmentor"),
    checkpoint: str | None = Form(None),
    tile_size:  int        = Form(512),
    overlap:    int        = Form(128),
) -> PredictResponse:
    """Run inference on uploaded GeoTIFF files."""
    tmpdir = tempfile.mkdtemp(prefix="cdm_upload_")
    try:
        pre_path  = Path(tmpdir) / "pre.tif"
        post_path = Path(tmpdir) / "post.tif"

        for upload, dest in [(pre_file, pre_path), (post_file, post_path)]:
            with open(dest, "wb") as fh:
                shutil.copyfileobj(upload.file, fh)

        ck  = _resolve_checkpoint(model_type, checkpoint)
        cfg = _load_cfg(model_type)
        dev = _device()

        result = predict_geotiff(
            pre_path=pre_path,
            post_path=post_path,
            model_type=model_type,
            checkpoint=ck,
            cfg=cfg,
            tile_size=tile_size,
            overlap=overlap,
            return_geojson=True,
            device=dev,
        )
        save_result(
            job_id=result["job_id"],
            model_type=model_type,
            pre_path=pre_file.filename or "uploaded_pre.tif",
            post_path=post_file.filename or "uploaded_post.tif",
            class_stats=result["class_stats"],
            geojson=result.get("geojson"),
        )
        return _build_response(result, model_type)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/results/{job_id}", tags=["results"])
def get_stored_result(job_id: str) -> JSONResponse:
    """Retrieve a previously stored inference result from PostGIS."""
    row = get_result(job_id)
    if row is None:
        raise HTTPException(404, f"Result {job_id!r} not found")
    # Convert non-serialisable types
    row = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}
    return JSONResponse(content=row)
