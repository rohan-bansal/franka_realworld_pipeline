# Preprocessing pipeline

After merging all demos into one file using scripts/postprocess_demos.py, run rename_dataset_keys.py to prune to important training keys.

Then preprocess_realworld_cotrack.py or preprocess_realworld_spatrack.py.

After this is done, make_gripper_states_2d.py, then normalize_axis_angle_direction.py

Then you are ready to run split_realworld_dataset.py

Now data is in ATM-compatible format!