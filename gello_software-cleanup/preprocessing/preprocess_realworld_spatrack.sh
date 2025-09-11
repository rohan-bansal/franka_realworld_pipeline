#!/bin/bash
#SBATCH --job-name=preprocess_realworld
#SBATCH --output=/srv/rl2-lab/flash8/rbansal66/ATM/scripts/preprocess_realworld.out
#SBATCH --error=/srv/rl2-lab/flash8/rbansal66/ATM/scripts/preprocess_realworld.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy,xaea-12,chappie"
#SBATCH --mem-per-gpu=64

export PYTHONUNBUFFERED=TRUE

source ~/.bashrc
conda deactivate
conda activate atm

nvidia-smi

cd /srv/rl2-lab/flash8/rbansal66/ATM

srun -u python -m scripts.preprocess_realworld_spatrack
