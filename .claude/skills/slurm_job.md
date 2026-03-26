
# Skill: Writing SLURM Jobs for HiperGator

Always use this template when creating .sbatch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=cdm_<task>
#SBATCH --output=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.out
#SBATCH --error=/blue/smin.fgcu/rkale.fgcu/cdm/logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rkale.fgcu@ufl.edu

module purge
module load conda
conda activate cdm

cd ~/local_project/conflict-damage-monitor
python src/<module>/<script>.py --config configs/g>.yaml

Rules:

Always write logs to /blue/.../logs/ not home dir

Always run module purge before module load

Never hardcode paths — reference configs/

Submit with: sbatch slurm/<script>.sbatch

Monitor with: squeue -u rkale.fgcu

