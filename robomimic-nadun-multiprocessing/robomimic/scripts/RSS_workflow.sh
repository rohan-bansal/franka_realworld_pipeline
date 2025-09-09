#!/bin/bash

# DEMO_NAME="/home/mbronars/zhenyang/demos/speed_stacking_agentview_0116/with_agentview_demo.hdf5"
# DEMO_NAME="/home/mbronars/zhenyang/demos/oven_bowl_50_demos_0117/with_agentview_demo.hdf5"
DEMO_NAME="/home/mbronars/zhenyang/demos/wiping_board_52_0118/with_agentview_demo.hdf5"
TARGET_NAME="wiping_board_52_0118.hdf5"

### Get waypoints from AWE
source /home/mbronars/anaconda3/etc/profile.d/conda.sh
conda activate awe_venv
python /home/mbronars/zhenyang/robomimic-nadun/robomimic/dev/awe/save_waypoints_realrobot_dataset_concurrent.py --dataset "$DEMO_NAME"
python /home/mbronars/zhenyang/robomimic-nadun/robomimic/dev/awe/label_awe_fastIL.py --demo "$DEMO_NAME"
python /home/mbronars/zhenyang/robomimic-nadun/robomimic/dev/awe/label_awe_fastIL.py --demo "$DEMO_NAME" --reached_act

scp "$DEMO_NAME" zchen927@sky1.cc.gatech.edu:/srv/rl2-lab/flash7/zhenyang/data/"$TARGET_NAME"

# source /home/mbronars/anaconda3/etc/profile.d/conda.sh
# conda activate robomimic-nadun
# cd /home/mbronars/zhenyang/robomimic-nadun/robomimic
# python scripts/train.py --config /home/mbronars/zhenyang/robomimic-nadun/skynet/configs/diffusion-policy/real_robot/stacking_cup/osc_desired_pos_franka_all_0116.json

