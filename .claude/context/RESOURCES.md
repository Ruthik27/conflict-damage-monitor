# HiperGator Resources

## Paths
- Code:        ~/local_project/conflict-damage-monitor/
- Data:        /blue/smin.fgcu/rkale.fgcu/cdm/data/
- Checkpoints: /blue/smin.fgcu/rkale.fgcu/cdm/checkpoints/
- Logs:        /blue/smin.fgcu/rkale.fgcu/cdm/logs/
- Outputs:     /blue/smin.fgcu/rkale.fgcu/cdm/outputs/

## SLURM Template
#!/bin/bash
#SBATCH --job-name=cdm_train
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00

module load conda
conda activate cdm
cd ~/local_project/conflict-damage-monitor
python src/train/train.py --config configs/train.yaml
