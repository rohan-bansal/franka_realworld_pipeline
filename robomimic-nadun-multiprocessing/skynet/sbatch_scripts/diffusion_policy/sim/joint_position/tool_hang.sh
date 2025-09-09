#!/bin/bash
#SBATCH --job-name=robomimic_tool_hang_joint_actions_rerun
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/sim/joint_position/robomimic_tool_hang_joint_actions_rerun.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/sim/joint_position/robomimic_tool_hang_joint_actions_rerun.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=128G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

config_folder=/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/configs/diffusion-policy/sim/joint_position/
config="tool_hang.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path
