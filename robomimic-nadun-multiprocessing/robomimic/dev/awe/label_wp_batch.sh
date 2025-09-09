#bin/bash
source /home/mbronars/anaconda3/etc/profile.d/conda.sh
conda activate awe_venv

DEMO_PATH="/home/mbronars/zhenyang/demos/speed_stacking_52_0118/stacking_only.hdf5"
# DEMO_PATH="/home/mbronars/zhenyang/demos/oven_bowl_50_demos_0117/with_agentview_demo.hdf5"
NUM_WORKERS=6
ERR_THRESHOLD=0.01
THRESHOLDS=(0.02 0.03 0.04)

for THRESHOLD in "${THRESHOLDS[@]}"; do
    python /home/mbronars/zhenyang/robomimic-nadun/robomimic/dev/awe/save_waypoints_realrobot_dataset_concurrent.py --dataset $DEMO_PATH --num_workers $NUM_WORKERS --err_threshold $THRESHOLD
done
