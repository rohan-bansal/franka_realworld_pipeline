#!/bin/bash
#SBATCH --job-name=joint_control_all_obs_no_filter
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/real_robot/serve_apple/joint_control_all_obs_no_filter.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/real_robot/serve_apple/joint_control_all_obs_no_filter.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

config_folder=/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/serve_apple/
config="absolute_joint_positions.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path