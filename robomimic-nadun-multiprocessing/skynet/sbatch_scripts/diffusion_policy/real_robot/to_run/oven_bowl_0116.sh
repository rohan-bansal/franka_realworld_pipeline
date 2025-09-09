#!/bin/bash
#SBATCH --job-name=oven_bowl_0118_reached_awe
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/oven_bowl/0123_reached_pose_awe_joint_pos.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/oven_bowl/0123_reached_pose_awe_joint_pos.err

#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --partition=overcap
#SBATCH --requeue

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /srv/rl2-lab/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv

cd /srv/rl2-lab/flash7/zhenyang/robomimic-nadun/robomimic

config_folder=/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/oven_bowl/
config="joint_reached_pos_0120.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path