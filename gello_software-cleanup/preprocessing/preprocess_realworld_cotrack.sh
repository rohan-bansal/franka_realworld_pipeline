#!/bin/bash
#SBATCH --job-name=preprocess_realworld_cotrack
#SBATCH --output=/srv/rl2-lab/flash8/rbansal66/ATM/scripts/preprocess_realworld_cotrack.out
#SBATCH --error=/srv/rl2-lab/flash8/rbansal66/ATM/scripts/preprocess_realworld_cotrack.err
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun -u python -m scripts.preprocess_realworld_cotrack
