#!/bin/bash
#SBATCH --job-name=square_skill_image_3k    
#SBATCH --output=/coc/flash7/zhenyang/logs/sim/square/square_skill_image_3k_0602.out
#SBATCH --error=/coc/flash7/zhenyang/logs/sim/square/square_skill_image_3k_0602.err
#SBATCH --time=24:00:00
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /coc/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv

cd /coc/flash7/zhenyang/robomimic-nadun/robomimic

config_folder=/coc/flash7/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/sim/Zhenyang/
config="square_skill_osc_image.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path
