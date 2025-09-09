#!/bin/bash

# Python Script path - fix the path format
SCRIPT="/storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/robomimic/dev/sim/run_trained_agent_rh_hparam_noprecision.py"

# Common parameters for all runs
COMMON_ARGS="--n_rollouts 50 --horizon 400 --N_eval 3 --speed_up_factor 3"

# Task-dependent parameters
INIT_STATES="--init_states_path /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/datasets/square/ph/square_init_states.pkl"
VIDEO_DIR="--video_dir /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/compare/videos/compare_3x/square/"
LOG_DIR="--log_dir /storage/home/hcoda1/3/nkra3/p-dediger3-0/fast-imitation/robomimic-nadun/compare/logs/compare_3x/square/"

# Models
UNCOND_MODEL="--agent /home/wjung85/Repo/projects/FastIL/diffusion_trained_models/no_held_out/model_epoch_1000_square.pth"
COND_MODEL="--agent /home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/square/square_cfg_bm_false_model_epoch_1000.pth"

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


