#!/bin/bash
#SBATCH --job-name=ee_control_pick_cube_wrist_cam
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/real_robot/pick_cube/ee_control_pick_cube_wrist_cam.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/real_robot/pick_cube/ee_control_pick_cube_wrist_cam.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="rtx_6000:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

config_folder=/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/pick_cube/
config="pick_cube.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path