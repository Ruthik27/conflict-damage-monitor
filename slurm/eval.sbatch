#!/bin/bash
# Evaluate a checkpoint on the test split (CPU node — no GPU needed for small test set).
# Usage: sbatch slurm/eval.sbatch --checkpoint /blue/.../last.ckpt --model-type segmentor
#SBATCH --job-name=cdm-eval
#SBATCH --account=smin.fgcu
#SBATCH --qos=smin.fgcu
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/eval_%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

echo "Job $SLURM_JOB_ID | $(hostname) | $(date)"
module purge && module load conda
conda activate cdm
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
[[ -f configs/.env ]] && { set -a; source configs/.env; set +a; }

python src/train/eval.py \
    --paths-cfg configs/paths.yaml \
    "$@"

echo "Exit $? | $(date)"
