#!/bin/bash
# Serve the FastAPI inference server on a HiperGator compute node.
# Binds to port 8000; use SSH tunnel to access from laptop:
#   ssh -L 8000:$SLURMD_NODENAME:8000 rkale.fgcu@hpg.rc.ufl.edu
#
# Usage:
#   sbatch slurm/serve.sbatch
#   sbatch slurm/serve.sbatch --gres=gpu:l4:1   # GPU inference
#
#SBATCH --job-name=cdm-serve
#SBATCH --account=smin.fgcu
#SBATCH --qos=smin.fgcu
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/serve_%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/serve_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

echo "Job $SLURM_JOB_ID | node: $(hostname) | $(date)"
echo "Access via:  ssh -L 8000:$(hostname):8000 rkale.fgcu@hpg.rc.ufl.edu"

module purge && module load conda
conda activate cdm

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

[[ -f configs/.env ]] && { set -a; source configs/.env; set +a; }

# Pick best segmentor checkpoint if CHECKPOINT_SEG not set
if [[ -z "$CHECKPOINT_SEG" ]]; then
    CHECKPOINT_SEG=$(ls -t /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints/unetpp-*.ckpt 2>/dev/null | head -1)
    export CHECKPOINT_SEG
    echo "Auto-selected checkpoint: $CHECKPOINT_SEG"
fi

uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info

echo "Server exited $? | $(date)"
