"""
This script will use DP to predict a sequence of actions, but will only execute the first one and then re-predict

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
import h5py
import imageio
import numpy as np
from copy import deepcopy
import time
import os
import datetime
import pickle
from tqdm import tqdm

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.dev.dev_utils import save_obs
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from collections import defaultdict
import pandas as pd

from robomimic.dev.dev_utils import save_obs, write_video_constant_sim_time



def rollout(policy, env, horizon, max_sim_time, render=False, video_writer=None, video_skip=5, return_obs=False,
            camera_names=None, init_state_dict=None, **kwargs):
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

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))


    policy.start_episode()

    if init_state_dict is not None:  # For consistent eval
        obs = env.reset_to(init_state_dict)
        state_dict = env.get_state()
    else:  # random eval
        obs = env.reset()
        state_dict = env.get_state()

        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)


    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict, obs=[])
    total_inference_time = 0
    num_actual_actions = 0
    start_rollout = time.time()

    control_freq = kwargs.get("control_freq", 20)


    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        # for step_i in range(horizon):
        while env.getSimTimeInfo()[1] < max_sim_time:

            # get action from policy
            start = time.time()
            act = policy(ob=obs, **kwargs)

            if kwargs['osc_control']:
                act = act[:7] # making sure control is consistent with osc
                next_obs, r, done, _ = env.step(act, control_freq=control_freq)
            elif kwargs['joint_position_control']:
                # Change absolute joint position prediction to delta
                next_obs = env.get_observation()
                joint_pos = next_obs["robot0_joint_pos"]
                act[:-1] = act[:-1] - joint_pos
                next_obs, r, done, _ = env.step(act, control_freq=control_freq)

            total_inference_time += time.time() - start
            num_actual_actions += 1

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                video_annotation = f"rollout: {kwargs['rollout_number']}. control_freq: {control_freq}"
                video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env, text=video_annotation)

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                save_obs(traj, obs, **kwargs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=num_actual_actions , Success_Rate=float(success))

    action_sequence_length = policy.policy.algo_config.horizon.action_horizon
    traj['action_sequence_length'] = action_sequence_length
    traj['num_executed_actions'] = num_actual_actions
    traj['return'] = total_reward
    traj['rollout_time'] = time.time() - start_rollout
    traj['success'] = float(success)
    # print(f"Success is :{success}")
    sim_steps, sim_time_elapsed = env.getSimTimeInfo()
    traj['total_sim_time_elapsed'] = sim_time_elapsed

    return stats, traj


def run_trained_agent(args, **kwargs):
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
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

    # Setting the robosuite controller
    if kwargs["joint_position_control"]:
        from robomimic.robosuite_configs.paths import joint_position_nadun as jp_path
        joint_controller_fp = jp_path()
        controller_configs = json.load(open(joint_controller_fp))
        ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"] = controller_configs
    else:
        ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['control_delta'] = kwargs.get("control_delta", True)

    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['kp'] = kwargs['kp']
    ckpt_dict["env_metadata"]['env_kwargs']['torque_scale'] = kwargs.get("torque_scale", 1.0)

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

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe load initialization states for environment
    if args.init_states_path is not None:
        with open(args.init_states_path, "rb") as f:
            init_states = pickle.load(f)
    else:
        init_states = None

    # do we write the rollout trajs?
    write_dataset = (args.dataset_path is not None)

    rollout_stats = []
    rollout_trajs = {}
    start = time.time()

    for i in tqdm(range(rollout_num_episodes)):
        start_episode = time.time()
        kwargs['rollout_number'] = i

        if init_states is not None:
            init_state_dict = init_states[i]
        else:
            init_state_dict = None

        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon,
            max_sim_time=args.max_sim_time,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.camera_names,
            init_state_dict=init_state_dict,
            **kwargs
        )
        stats["time_taken_in_run_agent"] = time.time() - start_episode
        rollout_stats.append(stats)
        traj['kwargs'] = kwargs
        rollout_trajs[f'demo_{i}'] = traj

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)

    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats[f"Time for {rollout_num_episodes} demos"] = time.time() - start

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        rollout_results = {
            "stats": avg_rollout_stats,
            "rollouts": rollout_trajs,
        }
        with open(args.dataset_path, 'wb') as f:
            pickle.dump(rollout_results, f)
        print("Wrote dataset trajectories and stats to {}".format(args.dataset_path))


    return avg_rollout_stats, rollout_stats




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default="",
        required=False,
        help="path to saved checkpoint pth file",
    )


    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=100,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    parser.add_argument(
        "--max_sim_time",
        type=float,
        default=50.0,
        help="max amount of simulated time per rollout",
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

    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help = "(Optional) where to save videos of the different evals"
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
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an pkl file will be written with the rollout data
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

    parser.add_argument(
        "--control_freq",
        type=int,
        default=None,
        help="how fast to run robot",
    )
    parser.add_argument(
        "--evaluate_control_freqs",
        action='store_true',
        help="whether to evaluate model over different control frequencies"
    )

    parser.add_argument(
        '--rollout_stats_path',
        type=str,
        default=None,
        help="Where to save the rollout stats to, as a pandas dataframe"
    )

    parser.add_argument(
        '--init_states_path',
        type=str,
        default=None,
        help="Pickle file that has a list of initialization states for the environment, for consistent eval"
    )

    args = parser.parse_args()

    if args.video_dir is None:
        args.video_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'videos'))

    if args.video_path is None:
        args.video_path = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'videos', f"eval_normal_{datetime.datetime.now()}.mp4"))

    if args.dataset_path is None:
        args.dataset_path = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'logs',
                                                         f'eval_normal_{datetime.datetime.now()}.pkl'))


    kwargs = {
        # For controller
        "control_delta": False, "joint_position_control": False, "osc_control": True,
        "control_freq": 20,
        "kp": 150,
        "torque_scale": 1.0,
        # For saving data
              'return_obs': True, 'save_high_dim_obs': False}

    run_trained_agent(args, **kwargs)
