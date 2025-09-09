#!/bin/bash
#SBATCH --job-name=stack_three_baseline_uncapped
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/rss_experiments/baseline_dp/stack_three_baseline_uncapped.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/rss_experiments/baseline_dp/stack_three_baseline_uncapped.err
#SBATCH --partition=overcap
#SBATCH --time=10:00:00
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


agent=/nethome/nkra3/flash7/phd_project/robomimic-nadun/bc_trained_models/rss/sim/absolute_osc/stack_three_image_action_horizon_16_speed_baseline_absolute_osc/20250122162954/models/model_epoch_1000.pth

init_states_path=/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/stack_three/human/stack_three_init_states.pkl

srun -u python -u dev/sim/run_trained_agent.py --agent=$agent --init_states_path=$init_states_path


