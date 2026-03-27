#!/bin/bash
#SBATCH --job-name=cdm-change
#SBATCH --account=smin.fgcu
#SBATCH --qos=smin.fgcu
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=24:00:00
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/train_change_%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/train_change_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

echo "Job $SLURM_JOB_ID | $(hostname) | $(date)"
module purge && module load conda
conda activate cdm
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
[[ -f configs/.env ]] && { set -a; source configs/.env; set +a; }

mkdir -p /blue/smin.fgcu/rkale.fgcu/cdm/logs /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints

python src/train/train_change.py \
    --config    configs/train_change.yaml \
    --paths-cfg configs/paths.yaml \
    "$@"

echo "Exit $? | $(date)"
