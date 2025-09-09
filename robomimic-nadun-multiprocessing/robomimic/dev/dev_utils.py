import numpy as np
from copy import deepcopy


import time
import h5py
import cv2
from scipy.spatial.transform import Rotation
import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


### Setup some constants
# TODO: find best way to handle these constants
DELTA_ACTION_MAGNITUDE_LIMIT = 1.0
DELTA_EPSILON = np.array([1e-7, 1e-7, 1e-7])
DELTA_ACTION_DIRECTION_THRESHOLD = 0.25

SCALE_ACTION_LIMIT = [0.05 * DELTA_ACTION_MAGNITUDE_LIMIT for i in range(3)] + [0.5 * DELTA_ACTION_MAGNITUDE_LIMIT for i in range(3)]

# For checking gripper states for slowdown heuristics
GRIPPER_VELOCITY_THRESHOLD = 0.5
GRIPPER_COMMAND_CHANGE_THRESHOLD = 0.2
GRIPPER_VEL_MIN_DEQUE_SIZE = 10

REPEAT_LAST_ACTION_TIMES = 10

# For rendering
SIM_RENDER_INTERVAL = 0.05




def postprocess_obs(obs):
    for key in obs:
        if "image" in key or "rgb" in key and "time" not in key:
            obs[key] = ObsUtils.process_obs(obs[key], obs_modality="rgb")
        if "depth" in key and "time" not in key:
            obs[key] = ObsUtils.process_obs(obs[key], obs_modality="depth")
    return obs







def complete_setup_for_replay(demo_fn, env_meta = None):
    demo_file = h5py.File(demo_fn)

    ### Init env
    if env_meta is None:
        env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)

        ### Resetting controller max control input and output here
        env_meta['env_kwargs']['controller_configs']['input_min'] = -DELTA_ACTION_MAGNITUDE_LIMIT
        env_meta['env_kwargs']['controller_configs']['input_max'] = DELTA_ACTION_MAGNITUDE_LIMIT

        env_meta['env_kwargs']['controller_configs']['output_min'] = [-x for x in SCALE_ACTION_LIMIT]
        env_meta['env_kwargs']['controller_configs']['output_max'] = SCALE_ACTION_LIMIT


    env = EnvUtils.create_env_from_metadata(env_meta,
                                            render=True,
                                            use_image_obs=True
                                            )

    # print("CREATED ENVIRONMENT =================================================")
    # print(env)

    ### Initializing OBS specs
    demo = demo_file['data/demo_0']

    obs = demo['obs']
    obs_modality_spec = {"obs": {
        "low_dim": [],
        "rgb": []
    }
    }
    for obs_key in obs:
        if "image" in obs_key:
            obs_modality_spec["obs"]["rgb"].append(obs_key)
        else:
            obs_modality_spec["obs"]["low_dim"].append(obs_key)

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_spec, verbose=False)
    return env, demo_file



def demo_obs_to_obs_dict(demo_obs, ind, observation_horizon=2):
    obs_dict = {}
    if ind == 0: #special case for first obs
        for o in demo_obs:
            obs_array = np.array(demo_obs[o][ind])
            obs_array = repeat_array(obs_array, observation_horizon)
            if obs_array.shape[0] < 2:
                obs_array = np.squeeze(obs_array, axis=0)
            obs_dict[o] = obs_array
    else:
        for o in demo_obs:
            obs_array = np.array(demo_obs[o][ind - observation_horizon + 1:ind+1])
            if obs_array.shape[0] < 2:
                obs_array = np.squeeze(obs_array, axis=0)
            obs_dict[o] = obs_array
    return obs_dict




def repeat_array(arr, C):
    """
    created through ChatGPT
    Repeat an array C times along a new axis at the start.

    Parameters:
    arr (numpy.ndarray): The input array of shape (N, D1, D2, ...)
    C (int): The number of times to repeat the array along the new axis.

    Returns:
    numpy.ndarray: The repeated array with shape (C, N, D1, D2, ...)
    """
    # Add a new dimension at axis 0
    arr_expanded = np.expand_dims(arr, axis=0)

    # Create the tiling pattern, with C repetitions in the new dimension and 1 for the others
    tile_pattern = (C,) + (1,) * arr.ndim

    # Repeat the array C times along the new dimension
    arr_repeated = np.tile(arr_expanded, tile_pattern)

    return arr_repeated

def in_same_direction(act1, act2, threshold=DELTA_ACTION_DIRECTION_THRESHOLD):
    """
    Check if two actions are in the same direction
    """
    act1_delta_pos = act1[0:3] + DELTA_EPSILON
    act1_norm = np.linalg.norm(act1_delta_pos)
    act1_delta_pos /= act1_norm

    act2_delta_pos = act2[0:3] + DELTA_EPSILON
    act2_norm = np.linalg.norm(act2_delta_pos)
    act2_delta_pos /= act2_norm

    d_prod_1 = np.dot(act1_delta_pos, act2_delta_pos)

    act1_delta_angle = act1[3:6] + DELTA_EPSILON
    act1_norm = np.linalg.norm(act1_delta_angle)
    act1_delta_angle /= act1_norm

    act2_delta_angle = act2[3:6] + DELTA_EPSILON
    act2_norm = np.linalg.norm(act2_delta_angle)
    act2_delta_angle /= act2_norm

    d_prod_2 = np.dot(act1_delta_angle, act2_delta_angle)

    if d_prod_1 > threshold and d_prod_2 > threshold:
        return True
    else:
        return False

def aggregate_delta_actions(actions, **kwargs):
    """
    This function will take a list of delta OSC/pose actions, represented as [position, axis_angles, gripper] and
    aggregated them for scaled execution
    Args:
        actions: list/numpy array of [position, axis_angles, gripper]
        obs:
        **kwargs:
            delta_action_magnitude_limit: how large each aggregated delta action can be,
            in terms of position displacement. This should be the same units as the actions themselves
            delta_action_direction_threshold: how aligned the actions should be, in terms of dot product

    Returns: Aggregated delta actions

    """
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    # This is how large a single action can be
    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)

    # This is how aligned actions must be for aggregation
    delta_action_direction_threshold = kwargs.get('delta_action_direction_threshold', DELTA_ACTION_DIRECTION_THRESHOLD)

    for i in range(1, actions.shape[0]):

        # if the magnitude of the current aggregated action is larger than the limit, stop aggregating and
        # add it to the list
        if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        same_dir = in_same_direction(np.copy(actions[i]), np.copy(curr_action), delta_action_direction_threshold)

        if same_dir:
            # If the current aggregated action and next action are in the same direction, keep aggregating
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
        else:
            # curr action and next action are not in the same direction
            # append the current action and start aggregating new
            agg_actions.append(curr_action)
            curr_action = actions[i]

    agg_actions.append(curr_action)
    return agg_actions

def aggregate_delta_actions_naive(actions, obs=None, **kwargs):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]

    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)


    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
            agg_actions.append(curr_action)
            curr_action = actions[i]
        else:
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
    agg_actions.append(curr_action)
    return agg_actions


def aggregate_delta_actions_with_gripper_check(actions, gripper_obs):
    actions = np.array(actions)
    agg_actions = []
    curr_action = actions[0]


    for i in range(1, actions.shape[0]):
        if sum(np.abs(curr_action[0:3])) > DELTA_ACTION_MAGNITUDE_LIMIT:
            agg_actions.append(curr_action)
            curr_action = actions[i]
            continue

        curr_gripper_obs = gripper_obs[i]
        prev_gripper_obs = gripper_obs[i-1]

        gripper_same = True
        if np.sum(np.abs(curr_gripper_obs - prev_gripper_obs)) > GRIPPER_CHANGE_THRESHOLD:
            gripper_same = False

        if in_same_direction(actions[i], curr_action) and gripper_same:
            # If actions are in the same direction and the gripper action does not change, aggregate
            curr_action[0:6] += actions[i][0:6]
            curr_action[-1] = actions[i][-1]
        else:
            # Either not in same direction or gripper action changes
            agg_actions.append(curr_action)
            curr_action = actions[i]

    return agg_actions

def gripper_command_changed(gripper_traj, last_gripper_act):
    """
    Args:
        gripper_traj: nd.Array of gripper actions (1, N)
        last_gripper_act: single gripper action - float

    Made with help from ChatGPT
    """
    # Compute the pairwise absolute differences between gripper commands in the trajectory
    diff_matrix = np.abs(gripper_traj[:, np.newaxis] - gripper_traj)

    # Extract the upper triangle of the difference matrix (excluding the diagonal)
    upper_triangle_diff = np.triu(diff_matrix, k=1)

    # Check if any difference exceeds the threshold
    changed = np.any(upper_triangle_diff > GRIPPER_COMMAND_CHANGE_THRESHOLD)

    if changed or np.abs((gripper_traj[0]) - last_gripper_act) > GRIPPER_COMMAND_CHANGE_THRESHOLD:
        return True
    else:
        return False


def create_aggregated_delta_actions_with_gripper_check(actions, obs, **kwargs):

    agg_actions = []
    gripper_vel = obs['robot0_gripper_qvel'][:]*100

    # Define maximum magnitude of delta action
    delta_action_magnitude_limit = kwargs.get('delta_action_magnitude_limit', DELTA_ACTION_MAGNITUDE_LIMIT)

    for i in range(0, actions.shape[0]):
        curr_action = deepcopy(actions[i])
        for j in range(i + 1, actions.shape[0]):
            if sum(np.abs(curr_action[0:3])) > delta_action_magnitude_limit:
                # Magnitude is too large, stop aggregating
                break

            gripper_moving = False
            if max(np.abs(gripper_vel[j])) > GRIPPER_VELOCITY_THRESHOLD:
                gripper_moving = True

            if in_same_direction(actions[j], curr_action) and not gripper_moving:
                # If actions are in the same direction and the gripper is not moving
                curr_action[0:6] += deepcopy(actions[j][0:6])
                curr_action[-1] = deepcopy(actions[j][-1])
            else:
                # Either not in same direction or gripper is moving, stop aggregating
                break

        agg_actions.append(curr_action)
    return np.array(agg_actions)


def rollout_open_loop_bc_rnn_joint_actions(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, **kwargs):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    total_inference_time = 0

    start_rollout = time.time()

    rollout_length = policy.policy.algo_config.rnn.horizon
    num_actual_actions = 0

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            start = time.time()
            actions = []
            if kwargs["return_action_sequence"]:
                for i in range(rollout_length):
                    act = policy(ob=obs)
                    actions.append(act)
            else:
                act = policy(ob=obs)
                actions.append(act)
            total_inference_time += time.time() - start

            # play regular actions
            for act in actions:
                num_actual_actions += 1
                next_obs, r, done, _ = env.step(act)
                if done:
                    break

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()


    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    total_time_taken = time.time() - start_rollout

    stats = dict(Return=total_reward, Horizon=num_actual_actions, Success_Rate=float(success), Time_Taken_in_rollout= total_time_taken)

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj

def create_absolute_actions(
                    states: np.ndarray,
                    actions: np.ndarray,
                    env) -> np.ndarray:
    """
    copied/adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py
    """
    """
    Given state and delta action sequence
    generate equivalent goal position and orientation for each step
    keep the original gripper action intact.
    """
    # in case of multi robot
    # reshape (N,14) to (N,2,7)
    # or (N,7) to (N,1,7)
    stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)

    # generate abs actions, both reached and commanded
    reached_action_goal_pos = np.zeros(
        stacked_actions.shape[:-1] + (3,),
        dtype=stacked_actions.dtype)
    reached_action_goal_ori = np.zeros(
        stacked_actions.shape[:-1] + (3,),
        dtype=stacked_actions.dtype)

    commanded_action_goal_pos = np.zeros(
        stacked_actions.shape[:-1] + (3,),
        dtype=stacked_actions.dtype)
    commanded_action_goal_ori = np.zeros(
        stacked_actions.shape[:-1] + (3,),
        dtype=stacked_actions.dtype)

    action_gripper = stacked_actions[..., [-1]]
    for i in range(len(states)):
        _ = env.reset_to({'states': states[i]})

        # taken from robot_env.py L#454
        for idx, robot in enumerate(env.env.robots):
            # run controller goal generator
            robot.control(stacked_actions[i, idx], policy_step=True)

            # read pos and ori from robots
            controller = robot.controller

            # TODO: reached pose
            reached_action_goal_pos[i, idx] = controller.ee_pos
            reached_action_goal_ori[i, idx] = Rotation.from_matrix(
                controller.ee_ori_mat).as_rotvec()

            # TODO: commanded pose
            commanded_action_goal_pos[i, idx] = controller.goal_pos
            commanded_action_goal_ori[i, idx] = Rotation.from_matrix(
                controller.goal_ori).as_rotvec()


    stacked_reached_abs_actions = np.concatenate([
        reached_action_goal_pos,
        reached_action_goal_ori,
        action_gripper
    ], axis=-1)

    stacked_commanded_abs_actions = np.concatenate([
        commanded_action_goal_pos,
        commanded_action_goal_ori,
        action_gripper
    ], axis=-1)

    reached_abs_actions = stacked_reached_abs_actions.reshape(actions.shape)
    commanded_abs_actions = stacked_commanded_abs_actions.reshape(actions.shape)

    return reached_abs_actions, commanded_abs_actions

def write_video(video_writer, video_count, video_skip, camera_names, env):
    if video_writer is not None:
        if video_count % video_skip == 0:
            video_img = []
            for cam_name in camera_names:
                video_img.append(
                    env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
            video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
            video_writer.append_data(video_img)
        video_count += 1
    return video_count

def save_obs(traj, obs, **kwargs):
    save_high_dim_obs = kwargs.get('save_high_dim_obs', False)
    if not save_high_dim_obs:
        obs_to_save = deepcopy(obs)
        for o in obs:
            if "image" in o:
                del obs_to_save[o]
        traj["obs"].append(ObsUtils.unprocess_obs_dict(obs_to_save))
    else:
        traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))

    return traj

def write_video_constant_sim_time(video_writer, video_count, camera_names, env, render_interval=SIM_RENDER_INTERVAL, text=None):
    control_frequency = env.env.env.control_freq
    sim_steps, sim_time_elapsed = env.getSimTimeInfo()

    # setup font for annotating images:
    font = cv2.FONT_HERSHEY_SIMPLEX

    if (sim_time_elapsed % render_interval < (1 / control_frequency)):
        video_img = []
        for cam_name in camera_names:
            video_img.append(
                env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
        video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
        if text is not None:
            cv2.putText(video_img, text, (10,30), font, 0.7, (255, 255, 255), 2)
        video_writer.append_data(video_img)
        video_count += 1
    return video_count


def get_slowdown_mode_from_model(prev_act, current_act, future_act, slowdown_window_size):
    """
    This function will check if we need to slowdown based on model predictions
    Args:
        prev_act: list of previously executed actions
        current_act: np array of shape (act_dim)
        future_act: np array of shape (Ta, act_dim)

    Returns:

    """


    # Special case for one
    if slowdown_window_size == 1:
        return current_act[-1] > 0.5
    else:
        segment_size = slowdown_window_size // 2
        left_window_size = min(len(prev_act), segment_size)
        right_window_size = min(future_act.shape[0], segment_size)
        left_window = np.array(prev_act[-left_window_size:])
        if left_window.shape[0] > 0:
            left_window = left_window[..., -1]
        else:
            left_window = np.zeros(1)
        slowdown_window = np.concatenate([left_window,
                                          current_act[np.newaxis, -1] , future_act[:right_window_size, -1]])

    return np.any(slowdown_window > 0.5)


def get_slowdown_mode_from_gripper(gripper_act, gripper_vel, vel_threshold=None):
    """
    This function will check if we need to slowdown based on the gripper action and obs.
    Logic: if the gripper action (over some n timesteps) changes,
            or if the gripper vel is over a threshold,
            we need to slowdown.
    Args:
        gripper_act: trajectory of gripper commands ndarray of shape (n)
        gripper_vel: deque of numpy arrays of shape (2,)

    Returns: bool, slowdown or not

    """

    if vel_threshold is None:
        vel_threshold = GRIPPER_VELOCITY_THRESHOLD

    # At the start of the demo/rollout we do not slowdown
    if len(gripper_vel) < GRIPPER_VEL_MIN_DEQUE_SIZE:
        return False

    # gripper vel is in m/s, convert to cm for readability
    gripper_vel = [x * 100 for x in gripper_vel]

    # check if any velocity exceeds threshold, slowdown if yes
    for vel in gripper_vel:
        if max(np.abs(vel)) > vel_threshold:
            return True

    # Compute the pairwise absolute differences between gripper commands in the trajectory
    diff_matrix = np.abs(gripper_act[:, np.newaxis] - gripper_act)

    # Extract the upper triangle of the difference matrix (excluding the diagonal)
    upper_triangle_diff = np.triu(diff_matrix, k=1)

    # Check if any difference exceeds the threshold
    command_changed = np.any(upper_triangle_diff > GRIPPER_COMMAND_CHANGE_THRESHOLD)

    return command_changed


def prepare_action(act, env, **kwargs):
    """
    Prepares an action for OSC vs joint control
    Args:
        act:
        kwargs:

    Returns:

    """
    if kwargs['osc_control']:
        return act
    elif kwargs['joint_position_control']:
        # Change absolute joint position prediction to delta
        next_obs = env.get_observation()
        joint_pos = next_obs["robot0_joint_pos"]
        act[:-1] = act[:-1] - joint_pos
        return act



