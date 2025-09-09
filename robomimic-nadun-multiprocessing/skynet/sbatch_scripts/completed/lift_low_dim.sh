#!/bin/bash
#SBATCH --job-name=robomimic_lift_low_dim
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/robomimic_lift_low_dim.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/robomimic_lift_low_dim.err
#SBATCH --partition=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --qos=long
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic

srun -u python -u scripts/train.py --config=/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/configs/lift_bc_rnn_low_dim.json --dataset=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/lift/ph/all_obs_v141.hdf5

