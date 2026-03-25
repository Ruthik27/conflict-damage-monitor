Below is a `PROJECT_SPEC.md` you can drop into your root folder and also paste into Claude as context.

***

# Conflict Damage Monitor – Project Specification

## 1. High-level goal and scope

**Goal**

Build a **Conflict Damage Monitor**: an end-to-end system that estimates and visualizes **building damage over time** from satellite imagery for conflict-affected regions (e.g., Iran, Israel, Lebanon, others), using only **open datasets** and **free satellite imagery**.

**Inputs**

- Pre- and post-event satellite imagery:
  - High-resolution RGB from xBD / BRIGHT (for training). [arxiv](https://arxiv.org/abs/1911.09296)
  - Sentinel‑2 optical imagery for inference in target regions. [eoportal](https://www.eoportal.org/satellite-missions/copernicus-sentinel-2)
  - (Optional for v2) Sentinel‑1 SAR imagery and BRIGHT SAR for multimodal modeling. [zenodo](https://zenodo.org/records/14619798)
- Contextual conflict data (e.g., ACLED) for analytics and correlations. [github](https://github.com/Ruthik27/GeoInsight-Global-Conflict-and-Crisis-Mapping)

**Outputs**

- Per-pixel or per-building **damage class**:
  - Intact  
  - Minor damage  
  - Major damage  
  - Destroyed  
- Spatial and temporal aggregates:
  - Damage indices per city/region and over time.
  - Maps and charts showing how damage accumulates during conflict periods.
- A **web dashboard** with:
  - Interactive map (Leaflet or similar) showing damage overlays.
  - Time slider to step through pre/post dates.
  - Simple analytics (counts/percentages of damaged buildings, time series).

**Constraints & framing**

- Non-commercial, research/portfolio use.
- Humanitarian framing: situational awareness, macro-scale impact analysis, **not** targeting or operational decision support.
- Use only **open** imagery and datasets (xBD, BRIGHT, Sentinel, ACLED, etc.). [arxiv](https://arxiv.org/abs/2501.06019)

***

## 2. Datasets and how we’ll use them

### 2.1 Training datasets (labeled damage)

#### xBD – Building Damage from Satellite Imagery

- Open, large-scale dataset for **building damage assessment**. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1911.09296)
- Contents:
  - 22,068 high-resolution satellite images, each 1024×1024 pixels. [hyper](https://hyper.ai/en/datasets/13272)
  - 850,736 building polygons (footprints) with damage labels across 45,361.79 km² of imagery. [reddit](https://www.reddit.com/r/MachineLearning/comments/dpre53/n_xview2_updated_xbd_building_damage_dataset_850k/)
  - 19 natural disasters (earthquakes, floods, wildfires, storms, volcanic eruptions) in multiple countries. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1911.09296)
  - Each building labeled with **ordinal damage level** (no, minor, major, destroyed; sometimes represented as 0–3/4). [sei.cmu](https://www.sei.cmu.edu/projects/xview-2-challenge/)
  - Additional labels for environmental factors (fire, water, smoke). [arxiv](https://arxiv.org/abs/1911.09296)
- Role in project:
  - Primary dataset to train an **RGB-only** U-Net style model for building damage segmentation and classification.  
  - Provides diverse pre/post imagery and labels to learn generic damage patterns. [arxiv](https://arxiv.org/abs/1911.09296)

#### BRIGHT – Multimodal Building Damage Dataset

- Open-access, globally distributed **multimodal** dataset specifically for AI-based disaster response. [zenodo](https://zenodo.org/records/14619798)
- Contents:
  - ~4,500 paired optical and SAR images, each covering an AOI. [zenodo](https://zenodo.org/records/14619798)
  - >350,000 labeled building instances (footprints). [zenodo](https://zenodo.org/records/14619797)
  - Spatial resolution between **0.3–1 m per pixel** (very high resolution). [arxiv](https://arxiv.org/abs/2501.06019)
  - Covers **five natural disaster types** and **two man-made disasters** (including **armed conflicts**) across 12–14 regions worldwide. [arxiv](https://arxiv.org/html/2501.06019v4)
  - Uses a standardized **three-tier damage classification**:
    - Intact (1), Damaged (2), Destroyed (3). [arxiv](https://arxiv.org/html/2501.06019v4)
- Role in project:
  - Train and evaluate **multimodal models** (optical + SAR) for more robust all-weather damage assessment. [themoonlight](https://www.themoonlight.io/en/review/bright-a-globally-distributed-multimodal-building-damage-assessment-dataset-with-very-high-resolution-for-all-weather-disaster-response)
  - Provide conflict-like examples to better match war damage scenarios. [arxiv](https://arxiv.org/abs/2501.06019)

### 2.2 Imagery for inference (target regions)

#### Sentinel‑2 – Optical Multispectral Imagery

- Part of Copernicus Sentinel missions; fully free under the open data policy. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)
- Key characteristics:
  - 13 spectral bands:
    - 4 bands at 10 m resolution.
    - 6 bands at 20 m.
    - 3 bands at 60 m. [docs.planet](https://docs.planet.com/data/public-data/copernicus/sentinel-2/)
  - 290 km swath width; 5-day revisit at the equator (2–3 days at mid-latitudes). [eoportal](https://www.eoportal.org/satellite-missions/copernicus-sentinel-2)
  - Supports land monitoring, disaster control, emergency management, risk mapping, and humanitarian relief. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)
- Access:
  - Copernicus Data Space / Copernicus Browser. [browser.dataspace.copernicus](https://browser.dataspace.copernicus.eu)
  - Other portals like Planet’s Sentinel mirror (still free for Sentinel). [docs.planet](https://docs.planet.com/data/public-data/copernicus/sentinel-2/)
- Role in project:
  - Provide **pre** and **post** optical imagery for Iran, Israel, Lebanon, and other regions.  
  - Use resampled or adapted models trained on xBD/BRIGHT to produce **damage maps at 10 m resolution**. [eoportal](https://www.eoportal.org/satellite-missions/copernicus-sentinel-2)

#### Sentinel‑1 – SAR Imagery (optional in v1, important for v2)

- C-band SAR constellation; operates day/night and in all weather. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions)
- Key characteristics:
  - ~10 m resolution for IW mode products.
  - Polar-orbiting satellites providing frequent global coverage. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions)
- Role in project:
  - For a later version, leverage BRIGHT’s SAR + optical structure to build **SAR-based or multimodal models**, then apply them to Sentinel‑1 over conflict zones. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions)

### 2.3 Contextual conflict/event data

#### ACLED (Armed Conflict Location & Event Data)

- Rich conflict event dataset with:
  - Exact locations, dates, event types, and fatalities for political violence and protests. [github](https://github.com/topics/acled-data)
- Already used in previous project **GeoInsight – Global Conflict and Crisis Mapping**. [github](https://github.com/Ruthik27/GeoInsight-Global-Conflict-and-Crisis-Mapping)
- Role in project:
  - Not directly used for pixel-level training, but for **analytics**:
    - Correlate model-predicted damage with conflict events over time.
    - Visualize events and damage together on the dashboard. [github](https://github.com/topics/acled-data)

***

## 3. v1 scope decisions

To keep this achievable and focused, **v1** is defined as:

1. **Regions**
   - Primary demonstration regions:
     - Gaza Strip (2023–2024 conflict period).
     - Southern Lebanon (recent escalation period).
   - Other countries (e.g., Iran) can be added later as **v2**, once pipeline is stable.

2. **Modalities**
   - For **v1**:
     - Train on **RGB-only** using xBD (pre/post high-res imagery). [hyper](https://hyper.ai/en/datasets/13272)
     - Optionally fine-tune or validate on BRIGHT optical data. [zenodo](https://zenodo.org/records/14619798)
     - Inference on **Sentinel‑2** only. [dataspace.copernicus](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)
   - For **v2**:
     - Incorporate **multimodal** (optical + SAR) modeling from BRIGHT and apply to Sentinel‑1/2. [arxiv](https://arxiv.org/abs/2501.06019)

3. **Output granularity**
   - v1: **Per-pixel damage map** with four classes (intact, minor, major, destroyed).  
   - Optional: aggregate to per-building using external building footprints (OSM/Overture) where feasible.

4. **Priority ordering**
   - Highest priority:
     - Working data pipeline (xBD → patches; Sentinel‑2 → patches). [arxiv](https://arxiv.org/abs/1911.09296)
     - Trained RGB U‑Net model with reasonable metrics on xBD. [nrrdc.ruc.edu](http://nrrdc.ruc.edu.cn/docs/2021-04/f50532f57638470f8a30560580f2652c.pdf)
     - Inference + map overlays for Gaza & Southern Lebanon.
   - Second priority:
     - BRIGHT integration and multimodal experiments. [zenodo](https://zenodo.org/records/14619798)
     - Time-series analysis for multiple dates.
   - Third priority:
     - Fancy dashboard features, advanced analytics, and additional regions.

***

## 4. Environment and repository setup

### 4.1 Compute environment

- Hardware: HiperGator GPU cluster (NVIDIA GPUs; exact models may be A100/V100/T4 – assume CUDA support).
- OS: Linux (typical HPC environment).
- Preferred stack:
  - Python 3.10+.
  - PyTorch + (optionally) PyTorch Lightning.
  - Geo-related libs: GDAL, rasterio, rioxarray, GeoPandas, shapely.
  - Data tools: numpy, pandas, xarray.
  - ML utilities: scikit-learn, albumentations/torchvision transforms.
  - Web/backend: FastAPI or Flask (for API), or Node/Next.js if using JS.
  - Frontend mapping: Leaflet (to stay consistent with previous GeoInsight project). [github](https://github.com/Ruthik27/GeoInsight-Global-Conflict-and-Crisis-Mapping)

### 4.2 Repository layout

Target repo structure:

- `data/`  
  - Download and preprocessing scripts:
    - `prepare_xbd.py`
    - `prepare_bright.py`
    - `download_sentinel.py`
    - `preprocess_sentinel.py`
- `models/`  
  - `unet_rgb.py` – U-Net-based model for RGB damage segmentation.  
  - (Later) `unet_multimodal.py` – optical + SAR.
- `training/`  
  - `train_xbd_rgb.py` – training loop for xBD.  
  - (Later) `train_bright_multimodal.py`.
- `inference/`  
  - `run_inference.py` – run model on Sentinel‑2 for AOIs, output GeoTIFF + GeoJSON.
- `dashboard/`  
  - Frontend (Leaflet or React/Leaflet).  
  - Backend/API (FastAPI or Flask) if necessary.
- `docs/`  
  - `PROJECT_SPEC.md` (this file).  
  - Methodology, ethics, notes.

***

## 5. Preferences and constraints for Claude

### 5.1 Languages and libraries

- **ML & data**: Python-first, PyTorch preferred.
- **Mapping**:
  - Prefer **Leaflet** (JS or React-Leaflet) because previous work (GeoInsight) already uses Leaflet and I’m familiar with it. [github](https://github.com/Ruthik27/GeoInsight-Global-Conflict-and-Crisis-Mapping)
- **Web backend**:
  - Light Python backend (FastAPI/Flask) if needed for serving tiles/GeoJSON, otherwise static hosting + pre-generated data is fine.

### 5.2 Time budget & priorities

- Approximate overall timeline: **10–14 weeks**, part-time alongside other work.
- Priorities:
  1. Data pipelines + xBD RGB model + basic Sentinel‑2 inference.
  2. Minimal usable dashboard (map + basic stats).
  3. Multimodal (BRIGHT/SAR), more regions, and polish.

### 5.3 Data policy

- Only use **open** datasets listed above:
  - xBD, BRIGHT, Sentinel‑1/2, ACLED, and other clearly open sources. [github](https://github.com/topics/acled-data)
- Do **not** depend on paid/commercial imagery (e.g., Maxar/Planet proprietary), except where they are part of an open dataset like BRIGHT. [arxiv](https://arxiv.org/abs/2501.06019)
- Document limitations of resolution and labels clearly (especially when applying models to new regions).

***

## 6. Ordered task list for Claude

When starting, please follow this stepwise plan instead of trying to do everything at once.

1. **Repo & environment setup**
   - Create base repo structure as above.
   - Add environment configuration (`requirements.txt` or `pyproject.toml`) with all needed Python dependencies.

2. **xBD preprocessing**
   - Implement `data/prepare_xbd.py` to:
     - Read xBD metadata and geojson labels. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1911.09296)
     - Generate pre/post paired image tiles (e.g., 256×256 or 512×512).
     - Rasterize building polygons into per-pixel masks with damage classes.
     - Save train/val/test splits.

3. **RGB U-Net model and training**
   - Implement `models/unet_rgb.py` (configurable depth, channels).
   - Implement `training/train_xbd_rgb.py`:
     - Dataset/dataloader using prepared tiles.
     - Training loop, evaluation metrics (mIoU, F1, accuracy). [mcv.uab](https://mcv.uab.cat/abstracts/737-2023-04-17%2015:26:14-xview2.pdf)
   - Target: reasonable performance on xBD test split.

4. **Sentinel‑2 data pipeline**
   - Implement `data/download_sentinel.py` to download Sentinel‑2 L1C/L2A imagery for specified AOIs/time ranges via Copernicus Data Space. [gisgeography](https://gisgeography.com/how-to-download-sentinel-satellite-data/)
   - Implement `data/preprocess_sentinel.py` to:
     - Cloud-filter and select best images per date. [gisgeography](https://gisgeography.com/how-to-download-sentinel-satellite-data/)
     - Resample/reproject to a consistent grid.
     - Generate pre/post stacks aligned with model input.

5. **Inference over Sentinel‑2**
   - Implement `inference/run_inference.py`:
     - Load trained xBD RGB model.
     - Run over Sentinel‑2 AOIs (Gaza & Southern Lebanon) for selected dates.
     - Output:
       - GeoTIFF damage probability maps.
       - Simplified GeoJSON/tiles for web display.

6. **Minimal web dashboard**
   - In `dashboard/`:
     - Build a simple Leaflet-based web map.
     - Load basemap (e.g., standard tiles or Sentinel‑2 cloudless). [s2maps](https://s2maps.eu)
     - Overlay damage layer (GeoJSON/tiles).
     - Add UI to switch between dates and show basic stats.

7. **(Stretch) BRIGHT & multimodal**
   - Implement `data/prepare_bright.py`:
     - Handle optical + SAR pairs, building polygons, and three-tier damage labels. [arxiv](https://arxiv.org/html/2501.06019v4)
   - Implement `models/unet_multimodal.py` and corresponding training script.

8. **Documentation & evaluation**
   - Add doc pages explaining:
     - Data sources, model design, evaluation metrics. [arxiv](https://arxiv.org/abs/1911.09296)
     - Limitations (Sentinel resolution, missing labels in some regions).
   - Provide example notebooks with qualitative before/after visualizations.

***

## 7. Key references (for Claude, not to scrape directly)

- xBD dataset description and stats. [hyper](https://hyper.ai/en/datasets/13272)
- BRIGHT dataset description and stats. [themoonlight](https://www.themoonlight.io/en/review/bright-a-globally-distributed-multimodal-building-damage-assessment-dataset-with-very-high-resolution-for-all-weather-disaster-response)
- Sentinel‑2 mission description and basic facts. [docs.planet](https://docs.planet.com/data/public-data/copernicus/sentinel-2/)

***
