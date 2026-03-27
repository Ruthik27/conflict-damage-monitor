#!/bin/bash
# =============================================================================
# download_data.sbatch — Download xBD dataset on a HiperGator compute node.
#
# Usage:
#   sbatch slurm/download_data.sbatch --token <YOUR_TOKEN>
#
# All positional/named arguments after the script name are forwarded to
# src/data/download_xbd.py, so you can also pass --splits or --keep-archives:
#   sbatch slurm/download_data.sbatch --token <TOKEN> --splits train tier3
#
# Logs written to: /blue/smin.fgcu/rkale.fgcu/cdm/logs/download_%j.out
# =============================================================================

#SBATCH --job-name=xbd-download
#SBATCH --account=smin.fgcu
#SBATCH --qos=smin.fgcu
#SBATCH --partition=hpg-default      # CPU-only node — no GPU needed for I/O
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00              # 12 h ceiling — xBD is ~50 GB
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/download_%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

# ── environment ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "Working dir   : $SLURM_SUBMIT_DIR"
echo "========================================================"

module purge
module load conda

conda activate cdm

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# ── ensure log directory exists on /blue ─────────────────────────────────────
LOG_DIR=/blue/smin.fgcu/rkale.fgcu/cdm/logs
mkdir -p "$LOG_DIR"

# ── run download script ───────────────────────────────────────────────────────
# All arguments passed to sbatch after the script name are forwarded here.
# Example: sbatch slurm/download_data.sbatch --token abc123
python src/data/download_xbd.py \
    --paths-cfg configs/paths.yaml \
    "$@"

EXIT_CODE=$?

echo "========================================================"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE
