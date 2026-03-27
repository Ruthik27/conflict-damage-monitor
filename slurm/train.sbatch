#!/bin/bash
# =============================================================================
# train.sbatch — Train EfficientNet-B4 damage classifier on HiperGator L4 GPU.
#
# Allocation limits: 1 GPU, 8 CPU, 62 GB RAM (smin.fgcu)
# Partition: hpg-turin (L4 24 GB VRAM — shorter queue than b200/h100)
#
# Usage:
#   sbatch slurm/train.sbatch
#   sbatch slurm/train.sbatch --resume /blue/.../checkpoints/last.ckpt
# =============================================================================

#SBATCH --job-name=cdm-train
#SBATCH --account=smin.fgcu
#SBATCH --qos=smin.fgcu
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=24:00:00
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/train_%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

echo "========================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start    : $(date)"
echo "========================================================"

module purge
module load conda

conda activate cdm

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Load secrets (WANDB_API_KEY etc.) from configs/.env
if [[ -f "$SLURM_SUBMIT_DIR/configs/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$SLURM_SUBMIT_DIR/configs/.env"
    set +a
fi

mkdir -p /blue/smin.fgcu/rkale.fgcu/cdm/logs \
         /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints \
         /blue/smin.fgcu/rkale.fgcu/cdm/outputs

# Forward any extra args (e.g. --resume path/to/last.ckpt)
python src/train/train.py \
    --config    configs/train.yaml \
    --paths-cfg configs/paths.yaml \
    "$@"

EXIT_CODE=$?
echo "========================================================"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================"
exit $EXIT_CODE
