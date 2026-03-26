# Conflict Damage Monitor — Claude Instructions

## My Environment
- HPC: UF HiperGator (FGCU allocation: smin.fgcu)
- User: rkale.fgcu
- Code path: ~/local_project/conflict-damage-monitor/
- Data path: /blue/smin.fgcu/rkale.fgcu/cdm/data/
- Checkpoints: /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints/
- Logs: /blue/smin.fgcu/rkale.fgcu/cdm/logs/
- Outputs: /blue/smin.fgcu/rkale.fgcu/cdm/outputs/
- Conda env: cdm (Python 3.10, PyTorch 2.x, CUDA 11.8)
- GitHub: github.com/Ruthik27/conflict-damage-monitor

## Compute Constraints
- Scheduler: SLURM
- Max walltime: 48h
- Partition: gpu, gres: gpu:a100:2
- Never run heavy compute on login node — write .sbatch scripts
- Always activate: conda activate cdm

## Tech Stack
- ML: PyTorch, EfficientNet-B4, UNet++, segmentation-models-pytorch
- Geo: rasterio, GDAL, geopandas
- Data: xBD (850k buildings), BRIGHT (350k buildings), Sentinel-1/2
- Backend: FastAPI, PostGIS, Docker
- Frontend: React, Leaflet.js, Chart.js

## Project Phases
1. Data ingestion + preprocessing (tiling 512x512, augmentation)
2. Baseline EfficientNet-B4 classifier
3. UNet++ segmentation model
4. Change detection (pre/post diff, F1 per class)
5. FastAPI inference server + PostGIS
6. React dashboard

## Code Rules
- argparse for all CLI args
- Log to wandb (project: conflict-damage-monitor)
- Checkpoints every 5 epochs to /blue/.../checkpoints/
- No hardcoded paths — use configs/train.yaml
- No large data files in git
- Update environment.yml when adding packages
