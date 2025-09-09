#!/bin/bash
#SBATCH --job-name=can_awe_actions
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/processing/can_awe_actions.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/processing/can_awe_actions.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/awe

srun -u python -u utils/robomimic_convert_action.py --dataset=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/can/ph/all_obs_v141.hdf5