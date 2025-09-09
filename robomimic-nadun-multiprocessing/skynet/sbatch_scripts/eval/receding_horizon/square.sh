#!/bin/bash
#SBATCH --job-name=square_eval_fast_inpaint
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/eval/receding_horizon/square_eval_fast_inpaint.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/eval/receding_horizon/square_eval_fast_inpaint.err
#SBATCH --partition=overcap
#SBATCH --time 6:00:00
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

agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/square_image_action_horizon_16/20241211173329/models/model_epoch_1000.pth
init_states_path=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/square/ph/square_init_states.pkl

srun -u python -u dev/sim/run_trained_agent_receding_horizon.py --agent=$agent --init_states_path=$init_states_path