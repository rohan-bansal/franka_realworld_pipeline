#!/bin/bash
#SBATCH --job-name=can_Ta_eval_56
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/eval/action_horizon_experiments/absolute_osc/can_Ta_eval_56.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/eval/action_horizon_experiments/absolute_osc/can_Ta_eval_56.err
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="titan_x:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic


srun -u python -u dev/sim/run_trained_agent.py --agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_56/20241101205408/models/model_epoch_300.pth