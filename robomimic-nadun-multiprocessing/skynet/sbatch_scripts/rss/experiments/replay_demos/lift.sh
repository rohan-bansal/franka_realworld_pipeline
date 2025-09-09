#!/bin/bash
#SBATCH --job-name=lift_replay_over_kp_40_hz
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/rss_experiments/replay_demos/lift_replay_over_kp_40_hz.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/rss_experiments/replay_demos/lift_replay_over_kp_40_hz.err
#SBATCH --partition=overcap
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



srun -u python -u dev/replay_demos.py 