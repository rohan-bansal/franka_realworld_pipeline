#!/bin/bash
#SBATCH --job-name=wiping_board_0119_desired
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/wiping_board/0122_reached.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/wiping_board/0122_reached.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"

#SBATCH --partition=overcap

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /srv/rl2-lab/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv

cd /srv/rl2-lab/flash7/zhenyang/robomimic-nadun/robomimic

config_folder=/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/wiping_board_franka/
config="osc_reached_pos_franka.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path