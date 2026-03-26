# Training

## Test run (login node OK, dry-run only)
conda activate cdm
python src/train/train.py --config configs/train.yaml --dry-run

## Full training (SLURM)
sbatch slurm/train_job.sbatch
squeue -u rkale.fgcu
tail -f /blue/smin.fgcu/rkale.fgcu/cdm/logs/<job_id>.out
