#!/bin/bash

# Python Script path - fix the path format
SCRIPT="/storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/robomimic/dev/sim/run_trained_agent_rh_hparam_noprecision.py"

# Common parameters for all runs
COMMON_ARGS="--n_rollouts 50 --horizon 400 --N_eval 3 --speed_up_factor 5"

# Task-dependent parameters
INIT_STATES="--init_states_path /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/datasets/can/ph/can_init_states.pkl"
VIDEO_DIR="--video_dir /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/compare_torque/videos/can/"
LOG_DIR="--log_dir /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/compare_torque/logs/compare_5x/can/"

# Models
UNCOND_MODEL="--agent /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/bc_trained_models/rss/sim/absolute_osc_reached/can_image_action_horizon_16_speed_baseline_absolute_osc_held_out/20250122231542/models/can_baseline_epoch_1000.pth"
COND_MODEL="--agent /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/bc_trained_models/rss/sim/cfg/absolute_osc/can_image_action_horizon_16_cfg_binary_mask_off/20250122230035/models/can_cfg_bm_off_epoch_1000.pth"

# Guide configs
NO_GUIDE="--config /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/robomimic/dev/hparam/config/guide_template/base_no_guide.json"
INPAINT="--config /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/robomimic/dev/hparam/config/guide_template/base_inpaint_nres_3.json"
CFG="--config /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/robomimic/dev/hparam/config/guide_template/base_cfg_weight_0.json"

# Baseline model without inpainting
python $SCRIPT $UNCOND_MODEL $NO_GUIDE $INIT_STATES $VIDEO_DIR $LOG_DIR $COMMON_ARGS

# Baseline model with inpainting
python $SCRIPT $UNCOND_MODEL $INPAINT $INIT_STATES $VIDEO_DIR $LOG_DIR $COMMON_ARGS

# CFG model without inpainting
python $SCRIPT $COND_MODEL $NO_GUIDE $INIT_STATES $VIDEO_DIR $LOG_DIR $COMMON_ARGS

# CFG model with inpainting
python $SCRIPT $COND_MODEL $INPAINT $INIT_STATES $VIDEO_DIR $LOG_DIR $COMMON_ARGS

# CFG model with weight 0
python $SCRIPT $COND_MODEL $CFG $INIT_STATES $VIDEO_DIR $LOG_DIR $COMMON_ARGS


