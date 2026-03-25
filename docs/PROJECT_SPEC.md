# Conflict Damage Monitor – Project Specification

## 1. High-level goal and scope

**Goal**

Build a **Conflict Damage Monitor**: an end-to-end system that estimates and visualizes **building damage over time** from satellite imagery for conflict-affected regions (e.g., Iran, Israel, Lebanon, others), using only **open datasets** and **free satellite imagery**.

**Inputs**

- Pre- and post-event satellite imagery:
  - High-resolution RGB from xBD / BRIGHT (for training).
  - Sentinel-2 optical imagery for inference in target regions.
  - (Optional for v2) Sentinel-1 SAR imagery and BRIGHT SAR for multimodal modeling.
- Contextual conflict data (e.g., ACLED) for analytics and correlations.

**Outputs**

- Per-pixel or per-building **damage class**:
  - Intact
  - Minor damage
  - Major damage
  - Destroyed
- Spatial and temporal aggregates: damage indices per city/region and over time.
- A **web dashboard** with interactive map (Leaflet), time slider, and analytics.

**Constraints**

- Non-commercial, research/portfolio use.
- Humanitarian framing: situational awareness, macro-scale impact — NOT targeting.
- Only open imagery and datasets (xBD, BRIGHT, Sentinel, ACLED).

---

## 2. Datasets

### 2.1 Training (labeled damage)

#### xBD
- 22,068 images (1024x1024 px), 850,736 building polygons, 45,361 km²
- 19 natural disasters, damage labels 0–3 (no/minor/major/destroyed)
- Role: Train RGB U-Net model

#### BRIGHT
- ~4,500 paired optical+SAR images, >350,000 labeled buildings, 0.3–1m resolution
- 5 natural + 2 man-made disaster types (incl. armed conflicts)
- Damage classes: Intact / Damaged / Destroyed
- Role: Train multimodal model (v2)

### 2.2 Inference imagery

#### Sentinel-2 (optical)
- 10–20m resolution, 5-day revisit, free via Copernicus
- Role: Pre/post imagery for Iran, Israel, Lebanon, Gaza

#### Sentinel-1 (SAR)
- ~10m, all-weather day/night, free via Copernicus
- Role: v2 multimodal inference

### 2.3 Context data

#### ACLED
- Conflict events with coordinates, dates, fatalities
- Role: Correlate predicted damage with recorded events on dashboard

---

## 3. v1 Scope

1. **Regions**: Gaza Strip + Southern Lebanon (2023–2024)
2. **Modalities**: RGB-only (xBD) + Sentinel-2 inference
3. **Output**: Per-pixel damage map (4 classes)
4. **Priority order**:
   1. xBD preprocessing + RGB U-Net training
   2. Sentinel-2 download + inference pipeline
   3. Minimal Leaflet dashboard
   4. BRIGHT multimodal (v2)

---

## 4. Environment

- **Compute**: HiperGator GPU cluster (UF Research Computing)
- **Python**: 3.10+
- **ML**: PyTorch + PyTorch Lightning
- **Geo**: rasterio, rioxarray, GDAL, GeoPandas, shapely
- **Web**: FastAPI or Flask (backend), React + Leaflet (frontend)
- **Utils**: numpy, pandas, scikit-learn, albumentations

---

## 5. Repository structure

```
conflict-damage-monitor/
├── data/
│   ├── prepare_xbd.py
│   ├── prepare_bright.py
│   ├── download_sentinel.py
│   └── preprocess_sentinel.py
├── models/
│   ├── unet_rgb.py
│   └── unet_multimodal.py
├── training/
│   ├── train_xbd_rgb.py
│   └── train_bright_multimodal.py
├── inference/
│   └── run_inference.py
├── dashboard/
│   ├── frontend/   # React + Leaflet
│   └── backend/    # FastAPI
├── docs/
│   ├── PROJECT_SPEC.md  (this file)
│   ├── methodology.md
│   └── ethics.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Ordered task list for Claude

1. Repo & environment setup (requirements.txt, folder skeleton)
2. xBD preprocessing (tiling, mask generation, train/val/test splits)
3. RGB U-Net model + training script on xBD
4. Sentinel-2 download + preprocessing pipeline
5. Inference pipeline (load model, run over Sentinel-2 AOIs, output GeoTIFF/GeoJSON)
6. Minimal Leaflet dashboard (map + damage overlay + basic stats)
7. BRIGHT multimodal preprocessing + model (v2)
8. Documentation + evaluation notebooks

---

## 7. Preferences

- Python-first for all ML/geo work
- Leaflet for mapping (consistent with GeoInsight project)
- FastAPI for lightweight backend if needed
- Only open data — no commercial imagery
- Document all limitations (resolution, missing labels) clearly
