#!/bin/bash
#SBATCH --job-name=mimicgen_add_all_actions
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/processing/mimicgen_add_all_actions.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/processing/mimicgen_add_all_actions.err
#SBATCH --partition=overcap
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --exclude="voltron"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

dataset_dir=/nethome/nkra3/flash7/phd_project/mimicgen/datasets/core

srun -u python -u dev/dataset_processing/add_all_actions.py --dataset_dir=$dataset_dir