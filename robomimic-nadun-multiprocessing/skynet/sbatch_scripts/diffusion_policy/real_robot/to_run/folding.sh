#!/bin/bash
#SBATCH --job-name=folding_cloths_CFG_2
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/folding/folding_cloths_50_0130_osc_reached_pos_CFG_2.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/logs/real_robot/folding/folding_cloths_50_0130_osc_reached_pos_CFG_2.err

#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --partition=overcap
#SBATCH --mem-per-gpu="15G"
#SBATCH --exclude="clippy"
#SBATCH --requeue

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /srv/rl2-lab/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv

cd /srv/rl2-lab/flash7/zhenyang/robomimic-nadun/robomimic

config_folder=/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/folding/
config="osc_reached_pos_franka_CFG_0130.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path