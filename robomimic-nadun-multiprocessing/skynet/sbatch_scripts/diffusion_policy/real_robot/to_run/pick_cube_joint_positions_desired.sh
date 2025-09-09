#!/bin/bash
#SBATCH --job-name=joint_desired_pos_franka
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/pick_cube_11_06/joint_desired_position.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/pick_cube_11_06/joint_desired_position.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /srv/rl2-lab/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv


cd /srv/rl2-lab/flash7/zhenyang/robomimic-nadun/robomimic

config_folder=/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/pick_cube_franka/
config="joint_desired_pos_franka.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path