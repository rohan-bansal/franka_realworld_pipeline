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
import h5py
import imageio
import numpy as np
from copy import deepcopy
import time
import os
import datetime

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from collections import defaultdict
import pandas as pd

from dev_utils import rollout_open_loop_bc_rnn_joint_actions
from pathlib import Path


def rollout_diffusion_policy(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, **kwargs):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    total_inference_time = 0
    num_actual_actions = 0
    all_videos_imgs = []

    start_rollout = time.time()
    actions_remaining = horizon

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        while actions_remaining > 0:

            # get action from policy
            start = time.time()
            act = policy(ob=obs, **kwargs)
            total_inference_time += time.time() - start
            # print(f"Act shape is : {act.shape}")
            # play action
            if kwargs['return_action_sequence']:
                # Step through joint action sequence
                for i in range(act.shape[0]):
                    a = act[i]
                    next_obs = env.get_observation()
                    joint_pos = next_obs["robot0_joint_pos"]
                    a[:-1] = a[:-1] - joint_pos
                    next_obs, r, done, _ = env.step(a)
                    actions_remaining -= 1
                    num_actual_actions += 1
                    if video_writer is not None:
                        if video_count % video_skip == 0:
                            video_img = []
                            for cam_name in camera_names:
                                video_img.append(
                                    env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                            video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                            video_writer.append_data(video_img)
                            all_videos_imgs.append(video_img)
                        video_count += 1
                    if done:
                        break
            else:
                # Model returns a single action, play it here
                next_obs = env.get_observation()
                joint_pos = next_obs["robot0_joint_pos"]
                act[:-1] = act[:-1] - joint_pos
                # act[:-1] *= 2
                next_obs, r, done, _ = env.step(act)
                num_actual_actions += 1
                actions_remaining -= 1

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
                    all_videos_imgs.append(video_img)
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

    stats = dict(Return=total_reward, Horizon=num_actual_actions, Success_Rate=float(success), Time_Taken_in_rollout = time.time() - start_rollout)

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

    # Create only the successful videos
    if success:
        success_video_writer = kwargs.get("success_video_writer", None)
        if success_video_writer is not None:
            for img in all_videos_imgs:
                success_video_writer.append_data(img)

    return stats, traj

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, **kwargs):
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

    # Use separate function to rollout open loop bc rnn and diffusion policy TODO make this more robust

    if (policy.policy.global_config.ALGO_NAME == "bc"):
        if policy.policy.algo_config.rnn.enabled:
            if policy.policy.algo_config.rnn.open_loop:
                return rollout_open_loop_bc_rnn_joint_actions(policy, env, horizon, render, video_writer, video_skip, return_obs, camera_names, **kwargs)
    elif (policy.policy.global_config.ALGO_NAME == "diffusion_policy"):
        return rollout_diffusion_policy(policy, env, horizon, render, video_writer, video_skip, return_obs, camera_names, **kwargs)


    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    total_inference_time = 0


    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            start = time.time()
            act = policy(ob=obs, **kwargs)
            print(f"Act shape is : {act.shape}")
            total_inference_time += time.time() - start

            # play action
            # TODO for now, we step through an action sequence here, maybe move it to the env_wrapper later
            if kwargs['return_action_sequence']:
                for i in range(act.shape[0]):
                    single_act = act[i]
                    next_obs = env.get_observation()
                    joint_pos = next_obs["robot0_joint_pos"]
                    single_act[:-1] = single_act[:-1] - joint_pos
                    next_obs, r, done, _ = env.step(single_act)
                    if video_writer is not None:
                        if video_count % video_skip == 0:
                            video_img = []
                            for cam_name in camera_names:
                                video_img.append(
                                    env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                            video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                            video_writer.append_data(video_img)
                        video_count += 1
                    if done:
                        break
            else:
                next_obs = env.get_observation()
                joint_pos = next_obs["robot0_joint_pos"]
                act[:-1] = act[:-1] - joint_pos
                next_obs, r, done, _ = env.step(act)

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
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
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

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

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

    # TODO setting control freq here
    if args.control_freq is not None:
        ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"] = args.control_freq


    from robomimic.robosuite_configs.paths import joint_position_nadun as jp_path
    joint_controller_fp = jp_path()
    # joint_controller_fp = "/media/nadun/Data/phd_project/robosuite/robosuite/controllers/config/joint_velocity_nadun.json"
    controller_configs = json.load(open(joint_controller_fp))

    ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"] = controller_configs


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

    # create successful video writer
    success_video_path = args.video_path.replace(".mp4", "_success_only.mp4")
    if write_video:
        success_video_writer = imageio.get_writer(success_video_path, fps=20)
        kwargs['success_video_writer'] = success_video_writer

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    start = time.time()
    c_freq = ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"]
    print(f"Evaluating control frequency: {c_freq}")
    for i in range(rollout_num_episodes):
        start_episode = time.time()
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            **kwargs
        )
        stats["time_taken_in_run_agent"] = time.time() - start_episode
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)

    if args.rollout_stats_path is not None:
        df = pd.DataFrame(rollout_stats)
        df.to_excel(args.rollout_stats_path)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats[f"Time for {rollout_num_episodes} demos"] = time.time() - start

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    return avg_rollout_stats, rollout_stats


def evaluate_over_control_freqs(args, start_range=10, end_range=200, step=10, success_threshold = 0.05):

    eval_data = defaultdict(list)
    counter = 0

    today = datetime.date.today()
    control_freq_eval_save_path = os.path.abspath(
        os.path.join(args.agent, '..', '..', f'logs/control_freq_eval_{today}.pkl'))
    print(f"Saving stats to: {control_freq_eval_save_path}")


    kwargs = {"return_action_sequence": True}
    orig_horizon = args.horizon

    for freq in range(start_range, end_range, step):
        args.control_freq = freq
        args.video_skip = freq//10
        eval_data["control_freq"].append(freq)

        args.horizon = int(orig_horizon // (20 / freq))

        print(f"Args horizon for control freq {freq} is: {args.horizon}")

        args.video_path = f"{args.video_dir}/rollout_horizon_{orig_horizon}_freq_{freq}.mp4"
        avg_rollout_stats, rollout_stats = run_trained_agent(args, **kwargs)
        for stat in rollout_stats:
            eval_data[stat].append(rollout_stats[stat])


    df = pd.DataFrame(eval_data)
    df.to_pickle(control_freq_eval_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default="/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/joint_position/can_all_obs/20240921132324/models/model_epoch_600.pth",
        required=False,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=30,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
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
        default=5,
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

    args = parser.parse_args()

    if args.video_dir is None:
        args.video_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'videos'))

    if args.rollout_stats_path is None:
        args.rollout_stats_path = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'logs',
                                                               f'eval{datetime.datetime.now()}.xlsx'))

    kwargs = {"return_action_sequence": True}
    # run_trained_agent(args, **kwargs)

    evaluate_over_control_freqs(args)

