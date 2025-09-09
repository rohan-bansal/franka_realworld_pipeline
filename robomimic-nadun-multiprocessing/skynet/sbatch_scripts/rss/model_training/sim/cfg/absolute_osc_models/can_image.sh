#!/bin/bash
#SBATCH --job-name=can_image_mask_on
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/rss_training/sim/cfg/absolute_osc/can_image_mask_on.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/rss_training/sim//cfg/absolute_osc/can_image_mask_on.err
#SBATCH --partition=overcap
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu=64

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

config_folder=/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/configs/rss/sim/cfg/absolute_osc_models/
config="can_image.json"
config_path="$config_folder$config"

srun -u python -u scripts/train.py --config=$config_path
