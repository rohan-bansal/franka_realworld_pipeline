#!/bin/bash
#SBATCH --job-name=square_image_eval_adaptive_speed
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/eval/adaptive_speed/square_image_eval_adaptive_speed.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/eval/adaptive_speed/square_image_eval_adaptive_speed.err
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


agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/diffusion_policy/sim/speed_adaptive_models/absolute_osc/square_all_obs_horizon_16_speed_adaptive/20250108125452/models/model_epoch_1000.pth
agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/diffusion_policy/sim/speed_adaptive_models/absolute_osc/square_image_action_horizon_16_speed_adaptive/20250108125500/models/model_epoch_1000.pth

init_states_path=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/square/ph/square_init_states.pkl

srun -u python -u dev/sim/run_trained_agent_receding_horizon.py --agent=$agent --init_states_path=$init_states_path