# Conflict Damage Monitor

> End-to-end ML pipeline for estimating and visualizing **building damage over time** in conflict-affected regions using open satellite imagery.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

This project trains a deep learning model (U-Net) on labeled building damage datasets (xBD, BRIGHT), then applies it to free Sentinel-1/2 satellite imagery over conflict zones to produce:

- **Per-pixel damage maps** (Intact / Minor / Major / Destroyed)
- **Time-series damage analytics** per city and region
- **Interactive web dashboard** with before/after imagery, damage overlays, and trend charts

## Target regions (v1)

- Gaza Strip (Oct 2023 – present)
- Southern Lebanon (Sep 2024 – present)

v2 will add: Iran, Israel, West Bank, and others.

---

## Repository structure

```
conflict-damage-monitor/
├── data/           # Download + preprocessing scripts (xBD, BRIGHT, Sentinel)
├── models/         # U-Net model definitions (RGB and multimodal)
├── training/       # Training scripts for xBD (v1) and BRIGHT (v2)
├── inference/      # Run trained model over Sentinel AOIs → GeoTIFF/GeoJSON
├── dashboard/      # React + Leaflet frontend + FastAPI backend
├── docs/           # PROJECT_SPEC.md, methodology, ethics
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Datasets

| Dataset | Type | Size | Role |
|---------|------|------|------|
| [xBD](https://xview2.org/dataset) | RGB imagery + damage labels | 850k buildings, 45k km² | Train RGB U-Net (v1) |
| [BRIGHT](https://zenodo.org/record/BRIGHT) | Optical + SAR + labels | 350k+ buildings | Train multimodal model (v2) |
| Sentinel-2 | Free optical imagery, 10m | Global, daily | Inference over target regions |
| Sentinel-1 | Free SAR imagery, 10m | Global, daily | v2 inference |
| [ACLED](https://acleddata.com) | Conflict events + locations | Global | Dashboard analytics |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Ruthik27/conflict-damage-monitor
cd conflict-damage-monitor
pip install -r requirements.txt

# 2. Prepare xBD data
python data/prepare_xbd.py --xbd_dir /path/to/xbd --out_dir data/tiles/xbd

# 3. Train RGB U-Net on xBD
python training/train_xbd_rgb.py --data_dir data/tiles/xbd --epochs 50 --gpus 1

# 4. Download Sentinel-2 for Gaza
python data/download_sentinel.py --region gaza --start 2023-10-01 --end 2024-06-01

# 5. Run inference
python inference/run_inference.py --region gaza --checkpoint checkpoints/best.ckpt

# 6. Launch dashboard
cd dashboard && uvicorn backend.main:app --reload
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| ML | PyTorch, PyTorch Lightning, albumentations |
| Geo | rasterio, rioxarray, GeoPandas, GDAL |
| Sentinel | sentinelsat, eodag, Copernicus Data Space |
| Dashboard | React 18, React-Leaflet 4, Recharts, FastAPI |
| Compute | UF HiperGator GPU cluster |

---

## Ethics and limitations

This project is for **research and humanitarian situational awareness only**. It is not intended for targeting, operational military use, or any decision support that could endanger human life.

Limitations:
- Sentinel imagery is 10m resolution vs sub-meter training data — models are applied zero-shot with potential accuracy loss
- No ground truth labels in target regions; results are model estimates
- SAR / optical mismatch between training data and inference imagery (addressed in v2)

See [docs/ethics.md](docs/ethics.md) for full discussion.

---

## Project spec

Full project specification, dataset details, and ordered task list for AI coding assistants: [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)

---

## Author

**Ruthik Kale** — [ruthik27.github.io](https://ruthik27.github.io) | [GitHub](https://github.com/Ruthik27)
