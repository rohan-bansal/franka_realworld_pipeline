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
import pickle
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
from robomimic.dev.dev_utils import aggregate_delta_actions, write_video_constant_sim_time, save_obs
from collections import defaultdict
import pandas as pd

### Setup some constants


def rollout_open_loop_bc_rnn(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, **kwargs):
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

            if kwargs["aggregate_actions"]:

                agg_actions = aggregate_delta_actions(actions, **kwargs)

                # play aggregate actions
                for act in agg_actions:
                    num_actual_actions += 1
                    next_obs, r, done, _ = env.step(act)
                    if done:
                        break
            else:
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

def rollout_diffusion_policy(policy, env, horizon, max_sim_time, render=False, video_writer=None, video_skip=5, return_obs=False, 
                             camera_names=None, init_state_dict=None, **kwargs):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()

    if init_state_dict is not None: # For consistent eval
        obs = env.reset_to(init_state_dict)
        state_dict = env.get_state()
    else: # random eval
        obs = env.reset()
        state_dict = env.get_state()

        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)



    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    # traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)

    traj = dict(executed_actions=[], preds=[],
                rewards=[], dones=[],
                states=[], obs=[],
                control_freqs=[], dt=[],
                initial_state_dict=state_dict)
    total_inference_time = 0
    num_actual_actions = 0
    all_videos_imgs = []


    start_rollout = time.time()

    actions_remaining = horizon

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        # while actions_remaining > 0:
        while env.getSimTimeInfo()[1] < max_sim_time:


            # get action from policy
            start = time.time()
            act = policy(ob=obs, **kwargs)
            total_inference_time += time.time() - start

            # play action

            # TODO for now, we aggregate the action sequence and step through it here
            if kwargs['return_action_sequence']:
                if kwargs['aggregate_actions']:
                    # Play aggregated actions
                    # if kwargs["check_gripper"]:
                    #     if slowdown_mode:
                    #         agg_actions = act
                    #         slowdown_mode = False
                    #     if gripper_command_changed(act[:, -1], last_gripper_act):
                    #         agg_actions = act
                    #         slowdown_mode = True
                    #     else:
                    #         agg_actions = aggregate_delta_actions(act, None, **kwargs)

                    agg_actions = aggregate_delta_actions(act, **kwargs)

                    for act in agg_actions:
                        next_obs, r, done, _ = env.step(act)
                        traj['executed_actions'].append(act)
                        num_actual_actions += 1
                        actions_remaining -= 1
                        if video_writer is not None:
                            video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env)
                            # if video_count % video_skip == 0:
                            #     video_img = []
                            #     for cam_name in camera_names:
                            #         video_img.append(
                            #             env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                            #     video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                            #     video_writer.append_data(video_img)
                            #     all_videos_imgs.append(video_img)
                            # video_count += 1
                        if done:
                            break
                        # state saving
                        if return_obs:
                            save_obs(traj, next_obs, **kwargs)
                elif kwargs['scale_delta_rollout']:
                    for i in range(act.shape[0]):
                        a = act[i]
                        a[:6] *= kwargs['scale_delta_multiplier']
                        next_obs, r, done, _ = env.step(a)
                        if return_obs:
                            save_obs(traj, next_obs, **kwargs)
                        num_actual_actions += 1
                        actions_remaining -= 1
                        if video_writer is not None:
                            video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env)
                        if done:
                            break


                else:
                    # Play normal actions
                    for i in range(act.shape[0]):
                        a = act[i]
                        next_obs, r, done, _ = env.step(a)
                        if return_obs:
                            save_obs(traj, next_obs, **kwargs)
                        num_actual_actions += 1
                        actions_remaining -= 1
                        if video_writer is not None:
                            video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env)
                        if done:
                            break

            else:
                # Model returns a single action, play it here
                next_obs, r, done, _ = env.step(act)
                if return_obs:
                    save_obs(traj, next_obs, **kwargs)
                num_actual_actions += 1
                actions_remaining -= 1

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env)
                # if video_count % video_skip == 0:
                #     video_img = []
                #     for cam_name in camera_names:
                #         video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                #     video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                #     video_writer.append_data(video_img)
                #     all_videos_imgs.append(video_img)
                # video_count += 1

            # collect transition
            # traj["executed_actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            # if return_obs:
            #     # Note: We need to "unprocess" the observations to prepare to write them to dataset.
            #     #       This includes operations like channel swapping and float to uint8 conversion
            #     #       for saving disk space.
            #     traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
            #     traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()



    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=num_actual_actions, Success_Rate=float(success), Time_Taken_in_rollout = time.time() - start_rollout)

    # if return_obs:
    #     # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    #     traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    #     traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    # for k in traj:
    #     if k == "initial_state_dict":
    #         continue
    #     if isinstance(traj[k], dict):
    #         for kp in traj[k]:
    #             traj[k][kp] = np.array(traj[k][kp])
    #     else:
    #         traj[k] = np.array(traj[k])


    # Create only the successful videos
    # if success:
    #     success_video_writer = kwargs.get("success_video_writer", None)
    #     if success_video_writer is not None:
    #         for img in all_videos_imgs:
    #             success_video_writer.append_data(img)


    action_sequence_length = policy.policy.algo_config.horizon.action_horizon
    traj['action_sequence_length'] = action_sequence_length
    traj['total_inference_time'] = total_inference_time
    # traj['num_inferences'] = num_inferences
    traj['num_executed_actions'] = num_actual_actions
    traj['return'] = total_reward
    traj['rollout_time'] = time.time() - start_rollout
    traj['success'] = float(success)
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

    # TODO setting control freq here
    if args.control_freq is not None:
        ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"] = args.control_freq

    # TODO setting some scaling things here
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['input_min'] = - kwargs["delta_action_magnitude_limit"]
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['input_max'] = kwargs["delta_action_magnitude_limit"]
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['output_min'] =  [kwargs["scale_action_limit"] * -1 for i in range(3)] + [kwargs["scale_action_limit"] * -10 for i in range(3)]
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['output_max'] = [kwargs["scale_action_limit"] * 1 for i in range(3)] + [kwargs["scale_action_limit"] * 10 for i in range(3)]
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['kp'] = kwargs["kp"]
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['control_delta'] = kwargs.get("control_delta", True)
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
        verbose=False,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Maybe do consistent eval
    if args.init_states_path is not None:
        with open(args.init_states_path, "rb") as f:
            init_states = pickle.load(f)
    else:
        init_states = None

    # maybe create video writer
    video_writer = None
    success_video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

        # create successful video writer
        success_video_path = args.video_path.replace(".mp4", "_success_only.mp4")
        success_video_writer = imageio.get_writer(success_video_path, fps=20)
        # kwargs['success_video_writer'] = success_video_writer
    # if write_video:


    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    # if write_dataset:
    #     data_writer = h5py.File(args.dataset_path, "w")
    #     data_grp = data_writer.create_group("data")
    #     total_samples = 0

    rollout_stats = []
    rollout_trajs = {}
    start = time.time()
    c_freq = ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"]
    print(f"Evaluating control frequency: {c_freq}")
    for i in range(rollout_num_episodes):
        print(f"Running rollout episode: {i}")
        start_episode = time.time()

        if init_states is not None:
            init_state_dict = init_states[i]
        else:
            init_state_dict = None

        stats, traj = rollout_diffusion_policy(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon,
            max_sim_time=args.max_sim_time,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            init_state_dict=init_state_dict,
            success_video_writer = success_video_writer,
            **kwargs
        )
        stats["time_taken_in_run_agent"] = time.time() - start_episode
        rollout_stats.append(stats)
        traj['kwargs'] = kwargs
        rollout_trajs[f'demo_{i}'] = traj



    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)

    # if args.rollout_stats_path is not None:
    #     df = pd.DataFrame(rollout_stats)
    #     df.to_excel(args.rollout_stats_path)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats[f"Time for {rollout_num_episodes} demos"] = time.time() - start

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    # if write_dataset:
    #     # global metadata
    #     data_grp.attrs["total"] = total_samples
    #     data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    #     data_writer.close()
    #     print("Wrote dataset trajectories to {}".format(args.dataset_path))

    if write_dataset:
        rollout_results = {
            "stats": avg_rollout_stats,
            "rollouts": rollout_trajs,
            # "env_args": json.dumps(env.serialize(), indent=4) # environment info
        }
        with open(args.dataset_path, 'wb') as f:
            pickle.dump(rollout_results, f)
        print("Wrote dataset trajectories and stats to {}".format(args.dataset_path))

    return avg_rollout_stats, rollout_stats

def evaluate_aggregated_actions(args):
    # TODO put these in config or pass in if necessary

    delta_action_magnitude_limits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    delta_epsilon = np.array([1e-7, 1e-7, 1e-7])
    delta_action_direction_threshold = 0.25
    scale_action_limits = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


    delta_action_magnitude_limits = [1.0, 2.0, 3.0]
    delta_epsilon = np.array([1e-7, 1e-7, 1e-7])
    delta_action_direction_threshold = 0.25
    scale_action_limits = [0.05, 0.10, 0.15]


    df = None

    kwargs = {"return_action_sequence": False, "aggregate_actions": False, "delta_action_direction_threshold": 0.25,
               "delta_epsilon": np.array([1e-7, 1e-7, 1e-7]), "scale_action_limit": 0.05,
               "delta_action_magnitude_limit": 1.0, "kp": 150}

    ### TODO: TESTING NORMAL ROLLOUTS
    args.video_path = f"{args.video_dir}/normal_rollout.mp4"
    avg_rollout_stats, rollout_stats = run_trained_agent(args, **kwargs)
    rollout_stats["action_magnitude_limit"] = [-1 for i in range(args.n_rollouts)]
    df = pd.DataFrame(rollout_stats)

    ### TODO: TESTING AGGREGATED ROLLOUTS

    check_gripper = True

    kwargs = {"return_action_sequence": True, "aggregate_actions": True, "delta_action_direction_threshold": 0.25,
               "delta_epsilon": np.array([1e-7, 1e-7, 1e-7]), "scale_action_limit": 0.05,
               "delta_action_magnitude_limit": 1.0, "kp": 150, "check_gripper": check_gripper}

    for idx, limit in enumerate(scale_action_limits):

        kwargs["scale_action_limit"] = limit
        kwargs["delta_action_magnitude_limit"] = delta_action_magnitude_limits[idx]

        if args.video_dir is not None:
            args.video_path = f"{args.video_dir}/action_magnitude_{delta_action_magnitude_limits[idx]}_gripper_check_{check_gripper}.mp4"

        avg_rollout_stats, rollout_stats = run_trained_agent(args, **kwargs)
        rollout_stats["action_magnitude_limit"] = [delta_action_magnitude_limits[idx] for i in range(args.n_rollouts)]
        # rollout_stats["kp"] = [kp for i in range(args.n_rollouts)]

        if df is None:
            df = pd.DataFrame(rollout_stats)
        else:
            new_df = pd.DataFrame(rollout_stats)
            df = pd.concat([df, new_df], ignore_index=True)


    if args.rollout_stats_path is not None:
        df.to_excel(args.rollout_stats_path)


def evaluate_over_control_freqs(args, start_range=10, end_range=200, step=10, success_threshold = 0.05):

    eval_data = defaultdict(list)
    counter = 0

    today = datetime.date.today()
    control_freq_eval_save_path = os.path.abspath(
        os.path.join(args.agent, '..', '..', f'logs/control_freq_eval_{today}.pkl'))
    print(f"Saving stats to: {control_freq_eval_save_path}")


    kwargs = {"return_action_sequence": True, "aggregate_actions": False, "delta_action_direction_threshold": 0.25,
              "delta_epsilon": np.array([1e-7, 1e-7, 1e-7]), "scale_action_limit": 0.05,
              "delta_action_magnitude_limit": 1.0, "kp": 150, "control_delta": False}

    orig_horizon = args.horizon
    for freq in range(start_range, end_range, step):
        args.control_freq = freq
        eval_data["control_freq"].append(freq)

        args.horizon = int(orig_horizon // (20 / freq))
        args.video_skip = freq//10
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
        default="/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/models/model_epoch_1000.pth",
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
        default=600,
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

    parser.add_argument(
    '--init_states_path',
    type=str,
    default=None,
    help="Pickle file that has a list of initialization states for the environment, for consistent eval"
    )

    args = parser.parse_args()

    if args.video_dir is None:
        args.video_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'videos/scaled_delta'))
        os.makedirs(args.video_dir, exist_ok=True)

    if args.dataset_path is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'logs/scaled_delta'))
        os.makedirs(dataset_dir, exist_ok=True)

        args.dataset_path = os.path.join(dataset_dir, f"scale_15_uncapped_{datetime.datetime.now()}.pkl")

    if args.video_path is None:

        args.video_path = os.path.abspath(
            os.path.join(args.video_dir, f"scale_15_uncapped_{datetime.datetime.now()}.mp4"))


    kwargs = {"return_action_sequence": True,
              # Type of rollout 
              "aggregate_actions": False,
              "scale_delta_rollout": True, 

              # Arguments for aggregated action rollout
              "delta_action_direction_threshold": 0.25,
               "delta_epsilon": np.array([1e-7, 1e-7, 1e-7]), "scale_action_limit": 0.05,
               "delta_action_magnitude_limit": 1.0, 
               
               # Arguments for scaled delta action rollout
               'scale_delta_multiplier' : 1.5,

                # Controller args
               "kp": 150,
               "torque_scale": 2.0
               
               }

    avg_rollout_stats, rollout_stats = run_trained_agent(args, **kwargs)

    # evaluate_aggregated_actions(args)
    # evaluate_over_control_freqs(args)

