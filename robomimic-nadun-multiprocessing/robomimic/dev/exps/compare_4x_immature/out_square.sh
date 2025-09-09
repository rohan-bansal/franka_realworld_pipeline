#!/bin/bash

# Python Script path - fix the path format
SCRIPT="/home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_hparam_noprecision.py"

# Common parameters for all runs
COMMON_ARGS="--n_rollouts 50 --horizon 400 --N_eval 3 --speed_up_factor 4"

# Task-dependent parameters
INIT_STATES="--init_states_path /home/wjung85/Repo/projects/FastIL/eval_scenarios/square_init_states.pkl"
VIDEO_DIR="--video_dir /home/wjung85/Repo/projects/FastIL/videos/compare_4x_immature/square/"
LOG_DIR="--log_dir /home/wjung85/Repo/projects/FastIL/logs/compare_4x_immature/square/"

# Models
UNCOND_MODEL="--agent /home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/square/square_baseline_model_epoch_100.pth"
COND_MODEL="--agent /home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/square/square_cfg_bm_false_model_epoch_100.pth"

# Guide configs
NO_GUIDE="--config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/guide_template/base_no_guide.json"
INPAINT="--config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/guide_template/base_inpaint_nres_3.json"
CFG="--config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/guide_template/base_cfg_weight_0.json"

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


