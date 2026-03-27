---
name: hpc-slurm
description: >
  HiperGator HPC workflow skill for the conflict-damage-monitor project. Use this skill
  whenever the user needs to run training, preprocessing, inference, or any compute-heavy
  task on the cluster — including writing .sbatch scripts, submitting jobs, checking status,
  debugging SLURM errors, or thinking through resource allocation. Also trigger for questions
  about conda environments on HPC, reading logs from /blue/, or anything involving squeue,
  sbatch, scancel, or sacct. If the user says "run this on the cluster", "submit a job",
  "write a batch script", or "train on HiperGator", use this skill.
---

# HPC-SLURM Skill — HiperGator / conflict-damage-monitor

## The Golden Rule: /home is for code, /blue is for everything else

Code lives in `~/local_project/conflict-damage-monitor/`. That's it. Every data file,
checkpoint, log, and output **must** go to `/blue/smin.fgcu/rkale.fgcu/cdm/`. Writing
large files to `/home` will fill the quota and break other users on the allocation.

```
/blue/smin.fgcu/rkale.fgcu/cdm/
├── data/          ← raw + processed datasets (xBD, BRIGHT, Sentinel)
├── checkpoints/   ← model weights saved every 5 epochs
├── logs/          ← SLURM stdout/stderr + wandb offline cache
└── outputs/       ← inference results, tiled imagery, exports
```

---

## Standard .sbatch Template

Every job script starts from this template. Fill in `<job-name>` and the python command;
the resource block stays the same unless the user has a specific reason to change it.

```bash
#!/bin/bash
#SBATCH --job-name=cdm_<job-name>
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.err
#SBATCH --account=smin.fgcu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

# Environment setup — always in this order
module purge
module load conda
conda activate cdm

# Always cd to the project root so relative imports and configs work
cd ~/local_project/conflict-damage-monitor

# Your command here — use configs/train.yaml, never hardcode paths
python src/<module>/<script>.py --config configs/train.yaml
```

Save scripts to `slurm/` inside the project root. Submit with:
```bash
sbatch slurm/<script>.sbatch
```

---

## Common Job Patterns

### Training run
```bash
python src/training/train.py \
    --config configs/train.yaml \
    --checkpoint-dir /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints/ \
    --log-dir /blue/smin.fgcu/rkale.fgcu/cdm/logs/
```
Checkpoints save every 5 epochs automatically (configured in train.yaml).
Set `WANDB_DIR` so wandb cache stays on /blue too:
```bash
export WANDB_DIR=/blue/smin.fgcu/rkale.fgcu/cdm/logs/wandb
```
wandb project name: `conflict-damage-monitor`

### Preprocessing / tiling (CPU-bound)
For GDAL/rasterio tiling jobs, drop the GPU request:
```bash
#SBATCH --partition=hpg-default
# remove --gres line entirely
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
```

### Array jobs (multiple seeds / configs)
```bash
#SBATCH --array=0-4
python src/training/train.py --config configs/train.yaml --seed $SLURM_ARRAY_TASK_ID
```

---

## Monitoring & Debugging

```bash
# Check your running jobs
squeue -u rkale.fgcu

# Detailed status of a specific job
scontrol show job <job-id>

# Resource usage after job ends
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# Tail live output
tail -f /blue/smin.fgcu/rkale.fgcu/cdm/logs/<job-id>.out

# Cancel a job
scancel <job-id>
```

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `DUE TO TIME LIMIT` in .err | Walltime exceeded | Increase `--time` or checkpoint more often |
| `conda: command not found` | Missing `module load conda` | Add `module purge && module load conda` |
| `CUDA out of memory` | Batch size too large for 2×A100 | Reduce `batch_size` in configs/train.yaml |
| `No space left on device` | Writing to /home | Redirect outputs to /blue |
| `ImportError` | Wrong env active | Verify `conda activate cdm` ran |
| Job stays in PD state | Resource contention | Check `squeue --start -j <id>` for estimated start |

---

## Environment Notes

- Conda env: `cdm` — Python 3.10, PyTorch 2.x, CUDA 11.8
- To install new packages: `srun --pty --mem=8gb --time=1:00:00 bash` then install inside that session
- To update the env after editing environment.yml: `conda env update -f environment.yml --prune`
- Never run `python`, `pip install`, or heavy shell commands directly on the login node

---

## Quick Reference

| Setting | Value |
|---|---|
| Account | `smin.fgcu` |
| Partition | `gpu` |
| GPU | `gpu:a100:2` |
| CPUs | `8` |
| Memory | `64gb` |
| Walltime | `48:00:00` |
| Code path | `~/local_project/conflict-damage-monitor/` |
| Data | `/blue/smin.fgcu/rkale.fgcu/cdm/data/` |
| Checkpoints | `/blue/smin.fgcu/rkale.fgcu/cdm/checkpoints/` |
| Logs | `/blue/smin.fgcu/rkale.fgcu/cdm/logs/` |
| Outputs | `/blue/smin.fgcu/rkale.fgcu/cdm/outputs/` |
| wandb project | `conflict-damage-monitor` |
