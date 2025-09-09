import h5py
import numpy as np
from copy import deepcopy, error
import pickle
import argparse
import json

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.dev.dev_utils import complete_setup_for_replay

from scipy.spatial.transform import Rotation

import seaborn as sns
import matplotlib.pyplot as plt




def evaluate_rollout_error(env,
                           states, actions,
                           robot0_eef_pos,
                           robot0_eef_quat,
                           metric_skip_steps=1):
    '''
    copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
    Args:
        env:
        states:
        actions:
        robot0_eef_pos:
        robot0_eef_quat:
        metric_skip_steps:

    Returns:

    '''
    # first step have high error for some reason, not representative

    # evaluate abs actions
    rollout_next_states = list()
    rollout_next_eef_pos = list()
    rollout_next_eef_quat = list()
    obs = env.reset_to({'states': states[0]})
    for i in range(len(states)):
        # print(f"Evaling state: {i}")
        # obs = env.reset_to({'states': states[i]})
        obs, reward, done, info = env.step(actions[i])
        obs = env.get_observation()
        rollout_next_states.append(env.get_state()['states'])
        rollout_next_eef_pos.append(obs['robot0_eef_pos'])
        rollout_next_eef_quat.append(obs['robot0_eef_quat'])

    rollout_next_states = np.array(rollout_next_states)
    rollout_next_eef_pos = np.array(rollout_next_eef_pos)
    rollout_next_eef_quat = np.array(rollout_next_eef_quat)

    next_state_diff = states[1:] - rollout_next_states[:-1]
    max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

    next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
    next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
    max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

    next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
                        * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
    next_eef_rot_dist = next_eef_rot_diff.magnitude()
    max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

    info = {
        'state': next_state_diff,
        'pos': next_eef_pos_dist,
        'rot': next_eef_rot_dist,
        'max_state': max_next_state_diff,
        'max_pos': max_next_eef_pos_dist,
        'max_rot': max_next_eef_rot_dist
    }
    return info

def evaluate_rollout_error_joint_position_actions(env,
                           states, actions,
                           robot0_eef_pos,
                           robot0_eef_quat,
                           metric_skip_steps=1):
    '''
    copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
    Args:
        env:
        states:
        actions:
        robot0_eef_pos:
        robot0_eef_quat:
        metric_skip_steps:

    Returns:

    '''
    # first step have high error for some reason, not representative

    # evaluate abs actions
    rollout_next_states = list()
    rollout_next_eef_pos = list()
    rollout_next_eef_quat = list()
    obs = env.reset_to({'states': states[0]})
    for i in range(len(states)):
        act = np.copy(actions[i])
        next_obs = env.get_observation()
        joint_pos = next_obs["robot0_joint_pos"]
        act[:-1] = act[:-1] - joint_pos
        obs, reward, done, info = env.step(act)
        obs = env.get_observation()
        rollout_next_states.append(env.get_state()['states'])
        rollout_next_eef_pos.append(obs['robot0_eef_pos'])
        rollout_next_eef_quat.append(obs['robot0_eef_quat'])

    rollout_next_states = np.array(rollout_next_states)
    rollout_next_eef_pos = np.array(rollout_next_eef_pos)
    rollout_next_eef_quat = np.array(rollout_next_eef_quat)

    next_state_diff = states[1:] - rollout_next_states[:-1]
    max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

    next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
    next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
    max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

    next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
                        * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
    next_eef_rot_dist = next_eef_rot_diff.magnitude()
    max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

    info = {
        'state': next_state_diff,
        'pos': next_eef_pos_dist,
        'rot': next_eef_rot_dist,
        'max_state': max_next_state_diff,
        'max_pos': max_next_eef_pos_dist,
        'max_rot': max_next_eef_rot_dist
    }
    return info

def get_rollout_error_for_control_freq(demo_fn, control_freq):

    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    abs_env_meta = deepcopy(env_meta)
    abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    abs_env_meta["env_kwargs"]["control_freq"] = control_freq
    abs_env_meta['env_kwargs']["use_camera_obs"] = False
    abs_env_meta["env_kwargs"]["camera_names"] = []
    abs_env_meta["env_kwargs"]['has_offscreen_renderer'] = False
    env, demo_file = complete_setup_for_replay(demo_fn, env_meta=abs_env_meta)

    pos_error = None
    rot_error = None
    num_processed = 0

    max_pos = []
    max_rot = []

    for ep in demo_file['data']:
        demo = demo_file[f'data/{ep}']

        if (num_processed % 10) == 0:
            print(f"Processed demo : {num_processed} for control freq: {control_freq}")

        # if num_processed > 10:
        #     break

        num_processed += 1

        states = demo['states'][:]
        abs_actions = demo['absolute_actions'][:]

        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        error_info = evaluate_rollout_error(env, states, abs_actions, robot0_eef_pos, robot0_eef_quat)

        if pos_error is None:
            pos_error = error_info['pos'].tolist()
            rot_error = error_info['rot'].tolist()
        else:
            pos_error += error_info['pos'].tolist()
            rot_error += error_info['rot'].tolist()

        max_pos.append(error_info['max_pos'])
        max_rot.append(error_info['max_rot'])

    stats = {
        "pos_errors": pos_error,
        "rot_errors": rot_error,
        "max_pos_errors": max_pos,
        "max_rot_errors": max_rot
    }

    return stats


def get_tracking_error_in_dataset(demo_fn, absolute_action_key="commanded_absolute_actions"):
    demo_file = h5py.File(demo_fn)

    means = []
    num_processed = 0
    for ep in demo_file['data']:
        demo = demo_file[f'data/{ep}']

        if (num_processed % 10) == 0:
            print(f"Processed demo : {num_processed} ")


        num_processed += 1


        actions = demo[absolute_action_key][:, :3]

        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]

        diff = actions - robot0_eef_pos
        dist = np.linalg.norm(diff, axis=1)
        means.append(np.mean(dist))

    return means




def get_rollout_error_for_control_freq_joint_control(demo_fn, control_freq):

    env, demo_file = complete_setup_for_replay(demo_fn)

    ### Init env
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    from robomimic.robosuite_configs.paths import joint_position_nadun as jp_path
    joint_controller_fp = jp_path()
    controller_configs = json.load(open(joint_controller_fp))
    env_meta["env_kwargs"]["controller_configs"] = controller_configs
    env_meta["env_kwargs"]["control_freq"] = control_freq

    env = EnvUtils.create_env_from_metadata(env_meta,
                                            render=True,
                                            use_image_obs=True)

    pos_error = None
    rot_error = None
    num_processed = 0

    max_pos = []
    max_rot = []

    for ep in demo_file['data']:
        demo = demo_file[f'data/{ep}']

        if (num_processed % 10) == 0:
            print(f"Processed demo : {num_processed} for control freq: {control_freq}")

        if num_processed > 40:
            break

        num_processed += 1

        states = demo['states'][:]
        actions = demo["joint_position_actions"][:]

        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        error_info = evaluate_rollout_error_joint_position_actions(env, states, actions,
                                                                   robot0_eef_pos, robot0_eef_quat)

        if pos_error is None:
            pos_error = error_info['pos'].tolist()
            rot_error = error_info['rot'].tolist()
        else:
            pos_error += error_info['pos'].tolist()
            rot_error += error_info['rot'].tolist()

        max_pos.append(error_info['max_pos'])
        max_rot.append(error_info['max_rot'])

    stats = {
        "pos_errors": pos_error,
        "rot_errors": rot_error,
        "max_pos_errors": max_pos,
        "max_rot_errors": max_rot
    }

    return stats

def evaluate_tracking_error_over_control_freqs(demo_fn, stat_save_path, start_freq=10, end_freq=400, step=10):

    freq_to_errors = {}
    for freq in range(start_freq, end_freq, step):
        # TODO: make this a parameter
        # errors = get_rollout_error_for_control_freq(demo_fn, control_freq=freq)
        errors = get_rollout_error_for_control_freq_joint_control(demo_fn, control_freq=freq)
        freq_to_errors[freq] = errors

    with open(stat_save_path, "wb") as f:
        pickle.dump(freq_to_errors, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to dataset to evaluate
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5",
        help="path to dataset",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        # required=True,
        # default="/media/nadun/Data/phd_project/mimicgen/datasets/core/stack_d0.hdf5",
        # default="/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5",
        help="where to save the rollout stats"
    )

    args = parser.parse_args()

    # evaluate_tracking_error_over_control_freqs(demo_fn=args.dataset, stat_save_path=args.save_path)

    mimicgen_dataset = "/media/nadun/Data/phd_project/mimicgen/datasets/core/stack_d0.hdf5"
    error = get_tracking_error_in_dataset(mimicgen_dataset)
    # Create a density plot
    sns.kdeplot(error, shade=True, label="MimicGen_Stack")

    coffee = "/media/nadun/Data/phd_project/mimicgen/datasets/core/coffee_d0.hdf5"
    error = get_tracking_error_in_dataset(coffee)
    sns.kdeplot(error, shade=True, label="MimicGen_Coffee")

    mug_cleanup = "/media/nadun/Data/phd_project/mimicgen/datasets/core/mug_cleanup_d0.hdf5"
    error = get_tracking_error_in_dataset(mug_cleanup)
    sns.kdeplot(error, shade=True, label="MimicGen_MugCleanup")

    can =   "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"
    error = get_tracking_error_in_dataset(can)
    sns.kdeplot(error, shade=True, label="RoboMimic_Can")

    square = "/media/nadun/Data/phd_project/robomimic/datasets/square/ph/all_obs_v141.hdf5"
    error = get_tracking_error_in_dataset(square)
    sns.kdeplot(error, shade=True, label="RoboMimic_Square")

    plt.xlabel("Mean tracking error")
    plt.title("Tracking error for different MimicGen and Robomimic tasks (demos)")

    # Display the plot
    plt.legend()
    plt.show()