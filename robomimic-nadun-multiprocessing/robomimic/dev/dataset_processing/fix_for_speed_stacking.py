import h5py
import numpy as np
# from sklearn.cluster import DBSCAN
import argparse
import os

def detect_gripper_transition(abs_actions):
    gripper_data = abs_actions[:, 6]

    prev_state = -1
    transition_indices = [] # from close to open
    for i in range(len(gripper_data)):
        curr_state = gripper_data[i]
        if prev_state >= 0.9 and curr_state <=   -0.9:
            transition_indices.append(i)
        prev_state = curr_state

    print(transition_indices)
    return transition_indices[1]

def main(dataset):
    f = h5py.File(dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # Create a copy of the original dataset
    with h5py.File(f"{os.path.dirname(dataset)}/stacking_only.hdf5", "w") as new_f:
        f.copy("data", new_f)

    # Open the new dataset for trimming
    with h5py.File(f"{os.path.dirname(dataset)}/stacking_only.hdf5", "r+") as new_f:
        for i in range(len(demos)):
            ep = demos[i]
        
            absolute_actions = new_f[f"data/{ep}/absolute_actions"][()]
            absolute_commanded_actions_with_precision = new_f[f"data/{ep}/absolute_commanded_actions_with_precision"][()]
            absolute_reached_actions_eefpose_shift_precision = new_f[f"data/{ep}/absolute_reached_actions_eefpose_shift_precision"][()]

            transition_index = detect_gripper_transition(absolute_actions)
            transition_index += 10           
            
            # Trim the absolute actions
            absolute_actions_trimmed = absolute_actions[:transition_index]
            absolute_commanded_actions_with_precision_trimmed = absolute_commanded_actions_with_precision[:transition_index]
            absolute_reached_actions_eefpose_shift_precision_trimmed = absolute_reached_actions_eefpose_shift_precision[:transition_index]
            
            # Save the trimmed absolute actions
            new_f[f"data/{ep}/absolute_actions"][...] = absolute_actions_trimmed
            new_f[f"data/{ep}/absolute_commanded_actions_with_precision"][...] = absolute_commanded_actions_with_precision_trimmed
            new_f[f"data/{ep}/absolute_reached_actions_eefpose_shift_precision"][...] = absolute_reached_actions_eefpose_shift_precision_trimmed

            # Trim and save the observations
            for obs_key in new_f[f"data/{ep}/obs"].keys():
                new_f[f"data/{ep}/obs/{obs_key}"][...] = new_f[f"data/{ep}/obs/{obs_key}"][:transition_index]

def add_num_samples(dataset):
    f = h5py.File(dataset, "r+")
    for ep in f["data"].keys():
        f[f"data/{ep}"].attrs["num_samples"] = f[f"data/{ep}/absolute_actions"].shape[0]
    f.close()

def add_env_meta(dataset):
    import json
    f = h5py.File(dataset, "r+")
def fix_for_stacking_cup(dataset):
    f = h5py.File(dataset, "r+")

    ## fix attr['env_args']
    ## fix attr['num_samples]

    f2 = h5py.File("/home/mbronars/zhenyang/demos/speed_stacking_52_0118/with_agentview_demo.hdf5", "r")

    f['data'].attrs['env_args'] = f2['data'].attrs['env_args']

    demo_keys = list(f2['data'].keys())
    for demo_key in demo_keys:
        action_data = f['data'][demo_key]['absolute_actions']
        print(f"demo_key: {demo_key} num_samples: {action_data.shape[0]}")
        # print(f"demo key for f2 num_samples: {f2['data'][demo_key].attrs['num_samples']}")

        print(f"action shape {f['data'][demo_key]['abs_reached_actions_joint_pos_shift_precision'].shape}")
        print(f"obs shape {f['data'][demo_key]['obs/joint_positions'].shape} image shape {f['data'][demo_key]['obs/agentview_image'].shape}")
        f['data'][demo_key].attrs['num_samples'] = action_data.shape[0]

if __name__ == '__main__':

    env_args = {"env_name": "RL2RobotEnv", "type": 4, "lang": "33", "env_kwargs":
                 {"camera_config_dict": {"agentview": {"sn": "001039114912", "type": "Kinect", "resize": True, "resize_resolution": [512, 512]},
                                         "wrist": {"sn": 14620168, "fps": 60.0, "resize": True, "resize_resolution": [512, 512]}},
                                         "general_cfg_file": None, "control_freq": 20, "controller_type": "OSC_POSE", "controller_cfg_file": None,
                                         "controller_cfg_dict": None, "use_depth_obs": False, "state_freq": 100.0, "control_timeout": 1.0, "has_gripper":
                                         True, "use_visualizer": False, "gripper_type": "robotiq",
                                         "reset_joint_positions": [0.0007925123372213324, -0.16362003257456625, 0.14518421694487627, -2.5930469890684105, 0.02467854399933535, 2.4951139314723902, 0.21047918180756753]}}
    
    f.attrs["env_args"] = json.dumps(env_args)

    f.close()

if __name__ == '__main__':
    main("/home/mbronars/zhenyang/demos/speed_stacking_52_0118/with_agentview_demo.hdf5")
    # add_num_samples("/srv/rl2-lab/flash7/zhenyang/data/speed_stacking_52_stacking_only_0120.hdf5")
    # add_env_meta("/srv/rl2-lab/flash7/zhenyang/data/speed_stacking_52_stacking_only_0120.hdf5")
