"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import time
import pickle
import math

import h5py
import imageio
import numpy as np
from copy import deepcopy
import signal
import sys

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.dev.dev_utils import demo_obs_to_obs_dict
from robomimic.utils.ros_utils import float_time_to_ros, ros_time_to_float

import matplotlib.pyplot as plt

import h5py
from scipy.interpolate import UnivariateSpline
from scipy import spatial
# import nexusformat.nexus as nx

# demo_fn = "/mnt/Data/atrp_ac_learning/demos/merged_demos/demo_foam_cube_generalizable.hdf5"
# # file = nx.nxload(demo_fn)
# demo = h5py.File(demo_fn)
# demo = demo['data']['demo_1']
# actions = demo['delta_actions']
# rec_obs = demo['obs']

def spline_actions(act):
    """
    Fit a spline to action and smooth them
    """
    x = range(act.shape[0])
    for d in range(act.shape[1]):
        y = act[:, d]
        spline = UnivariateSpline(x, y, s=0.50)
        new_y = spline(x)
        act[:, d] = new_y
    return act

# TODO: compplete
def temporal_ensemble(env, new_act, joint_obs_time, ):
    """
    Smooth the transition between one set of actions and the next one by using a temporal ensemble

    """
    prev_act_weights = [1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.1, 0.1, 0, 0, 0, 0, 0]
    prev_weight_index = 0
    # time_since_last_sent_actions = time.time() - last_time_sent_actions

    last_act_start_time = ros_time_to_float(env.robot_interface.prev_joint_traj_start_time)
    curr_time = ros_time_to_float(env.robot_interface.node.get_clock().now().to_msg())
    time_since_last_sent_actions = curr_time - last_act_start_time
    # this is the pointer to where we are in the previous action traj, and where the actual robot is
    prev_action_index = math.ceil(time_since_last_sent_actions / env.robot_interface.j_point_time)
    prev_actions_completed.append(prev_action_index)
    blended_act = deepcopy(act)

    # this is the pointer to where we are in the current prediction's traj
    new_action_start_time = joint_obs_time
    new_action_index = math.ceil((curr_time - new_action_start_time) / env.robot_interface.j_point_time)
    drop_actions = deepcopy(new_action_index)
    kwargs['drop_actions'] = drop_actions
    drop_action_list.append(drop_actions)
    blended_act = blended_act[drop_actions:]
    new_action_index = 0

    ### Implementing with fixed blending window
    blending_window_size = 8
    blended_actions = 0
    last_blended_action = prev_action[-1]
    while blended_actions < blending_window_size:
        prev_act_weight = prev_act_weights[prev_weight_index]
        new_act_weight = 1 - prev_act_weight
        prev_weight_index += 1

        if prev_action_index < prev_action.shape[0]:
            p_act = prev_action[prev_action_index]
            prev_action_index += 1
        else:
            p_act = last_blended_action

        n_act = blended_act[new_action_index]
        b_act = p_act * prev_act_weight + n_act * new_act_weight

        blended_act[new_action_index] = b_act

        new_action_index += 1
        last_blended_action = b_act
        blended_actions += 1

    act = blended_act

def rollout_with_action_sequence(policy, env, horizon, render=False, video_writer=None, video_skip=5,
        return_obs=True, camera_names=None, real=True):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        real (bool): if real robot rollout
        rollout_demo: to use actions from demo instead of actual actions
        rollout_demo_obs: to use obs from the demo and rollout instead of actual obs
        demo: demo to use for rolling out actions or obs
        demo_act_key: which actions to rollout from demo if rolling out demo actions

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)  or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    policy.start_episode()
    obs = env.reset()

    if real:
        input("ready for next eval? hit enter to continue")
        time.sleep(3)
    state_dict = dict()


    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], preds=[], initial_state_dict=state_dict)
    diff = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
    step_i = 0

    # TODO: MOVE THIS
    kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "control_mode": "Joint_Position_Trajectory", "delta_model": False,
              "temporal_ensemble": True, "spline": True,
              "diffusion_sample_n": 1,  "run_actions" : 10}

    env.robot_interface.switch_to_joint_traj_controller()
    action_sequence_length = policy.policy.algo_config.horizon.action_horizon

    run_actions = kwargs["run_actions"]
    prev_action = None
    next_obs = deepcopy(obs)
    last_time_sent_actions = time.time()


    ### DATA COLLECTION STUFF

    env.robot_interface.reset_joint_position_messages()
    traj_publish_times = []
    traj_start_times = []
    published_traj_msgs = []
    inf_durations = []
    obs_acquisition_times = []
    inf_start_times = []
    obs_joint_pos = []
    obs_ee_pose = []
    prev_actions_completed = []
    drop_action_list = []
    all_preds_list = []

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        while step_i < horizon:

            obs = env.get_observation()

            if obs["joint_positions"].ndim > 1:
                obs_joint_pos.append(obs["joint_positions"][-1])
                obs_ee_pose.append(obs['ee_pose'][-1])
                # add the obs times to the traj
                obs_times = {}
                for mod in obs:
                    if "time" in mod:
                        obs_times[mod] = obs[mod][-1]
                obs_acquisition_times.append(obs_times)
            else:
                obs_joint_pos.append(obs["joint_positions"])
                obs_ee_pose.append(obs['ee_pose'])
                obs_times = {}
                for mod in obs:
                    if "time" in mod:
                        obs_times[mod] = obs[mod]
                obs_acquisition_times.append(obs_times)

            start_inf = time.time()
            kwargs["inf_start_time"] = env.robot_interface.node.get_clock().now().to_msg()

            # The traj is supposed to start from when the [revious joint position message was acuqiired
            joint_obs_time = obs_times['joint_state_time']
            kwargs["traj_start_time"] = float_time_to_ros(joint_obs_time)

            inf_start_times.append(kwargs["inf_start_time"])

            act = policy(ob=obs, **kwargs)

            traj["preds"].append(act)
            inf_time = time.time() - start_inf
            inf_durations.append(inf_time)

            if run_actions >= act.shape[0]:
                kwargs["traj_start_time"] = env.robot_interface.node.get_clock().now().to_msg()
                drop_action_list.append(0)

            # For the first timestep, since the robot is stationary, we start the trajectory when from the first action
            if step_i == 0: # TODO Find a better way of handling this
                kwargs["traj_start_time"] = env.robot_interface.node.get_clock().now().to_msg()
                drop_action_list.append(0)
            else:
                if kwargs['temporal_ensemble']:

                    # kwargs["traj_start_time"] = float_time_to_ros(curr_time)

            if kwargs["spline"]: # apply spline to smooth
                act = spline_actions(act)

            prev_action = act
            np.set_printoptions(precision=8)

            # send the whole action sequence using the joint trajectory
            last_time_sent_actions = time.time()
            next_obs, r, done, _ = env.step(act, need_obs=False, **kwargs)

            # Sleep for some actions to be executed
            sleep_time = run_actions * env.robot_interface.j_point_time - (time.time() - last_time_sent_actions)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            traj_publish_times.append(env.robot_interface.get_previous_traj_publish_time())
            traj_start_times.append(env.robot_interface.get_previous_traj_start_time())
            published_traj_msgs.append(env.robot_interface.get_previous_traj_msg())


            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            if real:
                traj["states"].append(obs["joint_positions"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(obs)

            # break if done or if success
            if done or success:
                break

            # Update step_i
            step_i += run_actions


    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    joint_position_messages = env.robot_interface.get_joint_positions_messages()
    traj["inf_start_times"] = inf_start_times
    traj["traj_publish_times"] = traj_publish_times
    traj['traj_start_times'] = traj_start_times
    traj['published_traj_msgs'] = published_traj_msgs
    traj["all_joint_msg"] = joint_position_messages
    traj['traj_point_time'] = env.robot_interface.j_point_time
    traj['run_actions'] = run_actions
    traj["inf_durations"] = inf_durations
    traj["skip_joint_actions"] = env.robot_interface.skip_joint_actions
    traj['obs_joint_pos'] = obs_joint_pos
    traj['obs_ee_pose'] = obs_ee_pose
    traj['obs_acquisition_times'] = obs_acquisition_times
    traj['prev_actions_completed'] = prev_actions_completed
    traj['drop_action_list'] = drop_action_list
    traj['all_preds_list'] = all_preds_list
    traj["kwargs"] = kwargs


    return stats, traj


def run_trained_agent(args):

    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
    ckpt_dict=ckpt_dict,
    env_name=args.env,
    render=args.render,
    render_offscreen=(args.video_path is not None),
    verbose=True,
    )

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)

    #Set obs shapes for policy: #TODO delete this later
    env.set_obs_shapes(policy.policy.obs_shapes)

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        rollout_data = {}

    rollout_stats = []
    rollout_horizon = 250 #TODO remove this


    for i in range(rollout_num_episodes):
        try:

            # TODO ROLLING OUT AS A SEQUENCE
            stats, traj = rollout_with_action_sequence(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                # return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C catched, stop execution")
                print("env rate measure")
                print(env.rate_measure)
                continue
            else:
                sys.exit(0)

        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            rollout_data["demo_{}".format(i)] = traj


    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        with open(args.dataset_path, "wb") as f:
            pickle.dump(rollout_data, f)
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        # required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=2,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["sideview_left_camera_rgb"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()

    args.agent = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_30/absolute_joint_control_no_framestack_all_obs_seq_32/20241030202541/models/model_epoch_1000.pth"

    args.agent = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32/20241108134533/models/model_epoch_800.pth"


    args.dataset_path = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32_run_10_2x_wrong.pkl"

    run_trained_agent(args)
    #
    # run_trained_agent_with_demo(args)
