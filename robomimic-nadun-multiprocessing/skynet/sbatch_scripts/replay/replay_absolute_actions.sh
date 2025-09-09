#!/bin/bash
#SBATCH --job-name=tool_hang_replay_absolute
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/replay/tool_hang_replay_absolute.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/replay/tool_hang_replay_absolute.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

srun -u python -u dev/speed_up_demo.py 