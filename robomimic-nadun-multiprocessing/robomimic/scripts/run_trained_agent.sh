#!/bin/bash
# export PYTHONPATH=$PYTHONPATH:/home/mbronars/zhenyang/robomimic-nadun/robosuite

source /srv/rl2-lab/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic_venv

# Config values not changed often
CAMERA_NAMES="agentview"
SEED=1
MAX_HORIZON=100
NUM_ROLLOUT=6

cd /coc/flash7/zhenyang/robomimic-nadun/robomimic/scripts

# Set path values
# AGENT_PATH="/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/sim/pick_cube_osc_desired_sim/20241113050225/models/model_epoch_1000.pth"
# AGENT_PATH="/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/sim/pick_cube_osc_desired_future_action_obs_sim/20241119215918/models/model_epoch_1000.pth"
# DATASET_SAVE_PATH="/home/mbronars/zhenyang/demos/pick_cube_osc_sim_1127_test.hdf5"
# VIDEO_SAVE_PATH="/home/mbronars/zhenyang/demos/pick_cube_sim_infer_1121_action_obs.mp4"
# VIDEO_SAVE_PATH="/home/mbronars/zhenyang/demos/pick_cube_osc_sim_1127_test.mp4"
# AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_osc_low_dim_0601/20250601000126/models/model_epoch_1000.pth"


# BASE_DIR="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_osc_low_dim/20250601233004/models"
VIDEO_DIR="/coc/flash7/zhenyang/data/robomimic-sim"


# AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_osc_image/20250602153357/models/latest.pth"
# AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_skill_osc_image/20250602153355/models/latest.pth"
AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_skill_osc_low_dim_0602/20250602211659/models/latest.pth" #"/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_osc_low_dim_0602/20250602211704/models/latest.pth"
AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/square_osc_image_3k/20250602232508/models/latest.pth"
AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/lift_osc_low_dim_0602/20250602233109/models/latest.pth"
AGENT_PATH="/coc/flash7/zhenyang/robomimic-nadun/dp_trained_models/sim/can_osc_low_dim_0602/20250602233208/models/latest.pth"
VIDEO_SAVE_PATH="${VIDEO_DIR}/square_videos/can_low_dim_infer.mp4"

# python /coc/flash7/zhenyang/robomimic/robomimic/scripts/run_trained_agent.py \
#     --agent ${AGENT_PATH} \
#     --camera_names ${CAMERA_NAMES} \
#     --seed ${SEED} \
#     --horizon ${MAX_HORIZON} \
#     --n_rollout ${NUM_ROLLOUT} \
#     --video_path ${VIDEO_SAVE_PATH}

python ./run_trained_agent.py \
    --agent ${AGENT_PATH} \
    --camera_names ${CAMERA_NAMES} \
    --seed ${SEED} \
    --horizon ${MAX_HORIZON} \
    --n_rollout ${NUM_ROLLOUT} \
    --video_path ${VIDEO_SAVE_PATH}
    # --sample_initial_state

# # Loop through epochs 100-900
# for epoch in $(seq 100 100 900); do
#     AGENT_PATH="${BASE_DIR}/model_epoch_${epoch}.pth"
#     VIDEO_SAVE_PATH="${VIDEO_DIR}/square_videos/square_osc_low_dim_infer_epoch${epoch}.mp4"
    
#     # Execute script with configs
#     python ./run_trained_agent.py \
#         --agent ${AGENT_PATH} \
#         --camera_names ${CAMERA_NAMES} \
#         --seed ${SEED} \
#         --horizon ${MAX_HORIZON} \
#         --n_rollout ${NUM_ROLLOUT} \
#         --video_path ${VIDEO_SAVE_PATH}
# done