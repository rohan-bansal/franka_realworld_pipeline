#!/bin/bash

DEST_PATH="/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot"

CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/oven_bowl_osc_reached_shift_pos_awe_franka_0122_2/20250122003118/models/model_epoch_400.pth" # /srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/wiping_osc_reached_pos_franka_rightwrist_0119/20250119020530/models/model_epoch_300.pth"
DEST_NAME="0122_oven_bowl_reached_ep400.pth"
scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/oven_bowl_osc_reached_shift_pos_awe_franka_0120/20250120012202/models/model_epoch_400.pth" #/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/wiping_osc_desired_pos_franka_rightwrist_0119/20250119020551/models/model_epoch_200.pth"
# DEST_NAME="0120_oven_bowl_reached_shift_awe_400.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_reached_shift_pos_awe_52demos_0122_obs_horizon_4_3/20250122010136/models/model_epoch_700.pth"
# DEST_NAME="0122_stacking_cup_reached_shift_awe_obs_horizon4_700.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_reached_shift_pos_awe_52demos_0122_obs_horizon_4_3/20250122010136/models/model_epoch_600.pth"
# DEST_NAME="0122_stacking_cup_reached_shift_awe_obs_horizon4_600.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_reached_shift_pos_awe_52demos_0122_obs_horizon_4_3/20250122010136/models/model_epoch_400.pth"
# DEST_NAME="0122_stacking_cup_reached_shift_awe_obs_horizon4_400.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_desired_pos_awe_52demos_0120/20250120131256/models/model_epoch_500.pth" #/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_desired_pos_awe_24demos_0118/20250118180442/models/model_epoch_400.pth"
# DEST_NAME="0120_stacking_cup_desired_awe_500.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"

# CKPT="/srv/rl2-lab/flash7/zhenyang/robomimic-nadun/dp_trained_models/real_robot/stacking_cup_osc_desired_pos_52demos_0118/20250118213502/models/model_epoch_300.pth"
# DEST_NAME="0118_stacking_desired_52demos.pth"
# scp zchen927@sky1.cc.gatech.edu:"$CKPT" "$DEST_PATH"/"$DEST_NAME"