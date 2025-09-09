import nexusformat.nexus as nx
import numpy as np
import h5py
import os
import traceback
import multiprocessing

import robomimic.utils.transform_utils as T
import robomimic.utils.file_utils as FileUtils
from robomimic.dev.dev_utils import complete_setup_for_replay, create_absolute_actions
from copy import deepcopy
import argparse


def delta_axis_angle_to_absolute(delta_angles, quats):
    ret = []
    for i in range(quats.shape[0]):
        quat = quats[i]
        delta_aa = delta_angles[i]

        start_rot = T.quat2mat(quat)
        axis, angle = T.vec2axisangle(delta_aa)
        delta_rot = T.axisangle2mat(axis, angle)
        abs_rot_mat = np.dot(delta_rot, start_rot)  # Final absolute rot mat

        abs_axis, abs_angle = T.mat2axisangle(abs_rot_mat)
        abs_aa = T.axisangle2vec(abs_axis, abs_angle)
        if any(np.isnan(abs_aa)):
            print()
        ret.append(abs_aa)

    return np.array(ret)

def batch_quat_to_axis_angles(quats):
    ret = []
    for i in range(quats.shape[0]):
        quat = quats[i]
        axis, angle = T.quat2axisangle(quat)
        axis_angle = T.axisangle2vec(axis, angle)
        ret.append(axis_angle)
    return np.array(ret)

def add_actions_to_dataset(demo_fn):

    print(f"Processing dataset: {demo_fn}")

    demo_f = h5py.File(demo_fn, "a")
    demo_file = demo_f['data']

    # Create env for absolute actions generation
    env, _ = complete_setup_for_replay(demo_fn)

    counter = 0

    for demo in demo_file:
        if (counter % 10) == 0:
            print(f"Processing demo: {counter}")
        counter += 1
        demo = demo_file[demo]

        ### Cleanup everything first
        if "absolute_actions" in demo:
            del demo["absolute_actions"]
        if "commanded_absolute_actions" in demo:
            del demo["commanded_absolute_actions"]
        if "joint_actions" in demo:
            del demo["joint_actions"]
        if "joint_position_actions" in demo:
            del demo["joint_position_actions"]
        if "delta_joint_actions" in demo:
            del demo["delta_joint_actions"]
        if "reached_pose_action" in demo:
            del demo["reached_pose_action"]


        delta_actions = demo['actions'][:]
        states = demo['states'][:]

        ### Create absolute eef actions
        reached_absolute_actions, commanded_absolute_actions = create_absolute_actions(states, actions=delta_actions, env=env)
        demo.create_dataset("absolute_actions", data=reached_absolute_actions)
        demo.create_dataset("commanded_absolute_actions", data=commanded_absolute_actions)

        ### Create reached position action
        target_pos = demo['obs/robot0_eef_pos'][1:]
        target_pos = np.vstack([target_pos, target_pos[-1, :]])
        target_axis_angles = reached_absolute_actions[:, 3:-1]
        gripper_act = delta_actions[:, -1, np.newaxis]

        # TODO, we are using the commanded axis angles for this, use the actual reached axis angles
        reached_pose_action = np.concatenate((target_pos, target_axis_angles, gripper_act), axis=1)
        demo.create_dataset("reached_pose_action", data=reached_pose_action)

        ### Create joint position action
        joint_actions = demo['obs/robot0_joint_pos'][1:] # We offset by 1 since the next joint position is the action
        last_joint_action = np.copy(demo['obs/robot0_joint_pos'][-1:, :])
        joint_actions = np.concatenate([joint_actions, last_joint_action], axis=0)

        # Create new joint action array with the gripper action included:
        gripper_act = delta_actions[:, -1, np.newaxis]
        joint_actions = np.concatenate((joint_actions, gripper_act), axis=1)
        demo.create_dataset("joint_position_actions", data=joint_actions)

        # Creating delta joint actions
        delta_joint_actions = deepcopy(demo['joint_position_actions'][:])

        joint_obs = demo['obs/robot0_joint_pos'][:]
        delta_joint_actions[:, :-1] = delta_joint_actions[:, :-1] - joint_obs
        demo.create_dataset("delta_joint_actions", data=delta_joint_actions)

    print("Done!")
    demo_f.close()

def process_datasets_in_parallel(dataset_paths, num_workers=6):
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(add_actions_to_dataset, dataset_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to dataset
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="/media/nadun/Data/phd_project/robomimic/datasets/stack_three/all_obs_v141.hdf5",
        # default="/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141_old.hdf5",
        help="path to dataset",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        # required=True,
        default=None,
        help="path to dataset",
    )

    args = parser.parse_args()

    if args.dataset_dir is not None:
        dataset_paths = [os.path.join(args.dataset_dir, filename) for filename in os.listdir(args.dataset_dir) 
                         if os.path.isfile(os.path.join(args.dataset_dir, filename))]
        
        print(f"These are the dataset paths: {dataset_paths}")

        process_datasets_in_parallel(dataset_paths)
    else:
        add_actions_to_dataset(args.dataset)




