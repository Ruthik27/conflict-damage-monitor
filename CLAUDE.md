# Conflict Damage Monitor

See .claude/CLAUDE.md for full Claude instructions.

Quick context:
- ML pipeline: satellite building damage classification (4 classes)
- HiperGator HPC (SLURM), conda env: cdm, user: rkale.fgcu
- Data: /blue/smin.fgcu/rkale.fgcu/cdm/data/
- Stack: PyTorch + EfficientNet-B4 + UNet++ + FastAPI + React
- Datasets: xBD (850k buildings) + BRIGHT (350k) + Sentinel-1/2
