#!/bin/bash
#SBATCH --job-name=lift_convex_4
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/rss_dev/blended_inpainting/lift_convex_4.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/rss_dev/blended_inpainting/lift_convex_4.err
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


agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/rss/sim/speed_adaptive_models/lift_image_action_horizon_16_speed_adaptive/20250114170107/models/model_epoch_1000.pth

init_states_path=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/lift/ph/lift_init_states.pkl
config=/nethome/nkra3/flash7/phd_project/robomimic-nadun/robomimic/dev/guiding_configs/base_config_lift.json


srun -u python -u dev/sim/run_trained_agent_receding_horizon.py --agent=$agent --init_states_path=$init_states_path --config=$config


