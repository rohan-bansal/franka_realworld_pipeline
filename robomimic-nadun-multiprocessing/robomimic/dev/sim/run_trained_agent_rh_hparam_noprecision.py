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
from robomimic.config import config_factory
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.dev.dev_utils import write_video_constant_sim_time, prepare_action
from collections import defaultdict, deque
import pandas as pd
from robomimic.utils.deco_utils import time_profile



### Setup some constants

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

def add_noise_to_action(act, **kwargs):

    std_dev = 0.02

    if kwargs['osc_control']: # if osc control, we only add noise to the position part, for now
        noise = np.random.normal(loc=0.0, scale=std_dev, size=(3))
        act[:3] += noise
    elif kwargs['joint_position_control']: # add noise to all joints
        noise = np.random.normal(loc=0.0, scale=std_dev, size=act.shape)
        act += noise

    return act

def rollout_diffusion_policy(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False,
                             camera_names=None,  init_state_dict=None, guide_config=None, **kwargs):
    """
    Args
        kwargs:
            return_action_sequence (bool): whether to return the action sequence
            aggregate_actions (bool): whether to aggregate the actions
            delta_action_direction_threshold (float): threshold for delta action direction
            delta_epsilon (np.ndarray): epsilon for delta action
            guide_mode (str): [None, "inpaint", "consistency"]
            guide_n_actions (int): number of actions to guide/inpaint
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    Td = kwargs.get("inf_delay", 0)
    Tne = kwargs.get("execute_n_actions", 0)
    Ta = Td+Tne

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
    traj = dict(executed_actions=[], preds=[], rewards=[], dones=[], states=[], obs=[], predicted_actions=[], diffusion_logs=[], initial_state_dict=state_dict)
    total_inference_time = 0
    num_actual_actions = 0
    num_inferences = 0

    start_rollout = time.time()

    actions_remaining = horizon
    prev_act = None
    prev_action_index = None
    done = False

    # Stuff for inpainting
    noisy_eval = kwargs.get('noisy_eval', False) # whether to add noise to actions before executing

    # Stuff for slowing down and speeding up
    slowdown_mode = False
    gripper_vel_deque = deque(maxlen=11)

    # For guiding
    if guide_config is not None:
        kwargs["guide_config"] = guide_config
    
    planning_time_vec = []

    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        while actions_remaining > 0:
            # TODO: fix step_i and horizon for return action sequence

            # get action from policy
            start = time.time()
            act = policy(ob=obs, **kwargs) # act will be a sequence (N, act_dim)
            num_inferences += 1
            total_inference_time += time.time() - start
            traj['preds'].append(act)
            traj['diffusion_logs'].append(policy.policy.step_logs)

            # play action
            current_action_index = 0
            if prev_act is not None:
                # we execute the actions from the previous prediction to simulate inf delay
                for i in range(kwargs["inf_delay"]):

                    # get action for this timestep:
                    a = prev_act[prev_action_index]
                    traj['predicted_actions'].append(a)

                    # Check if we need to slowdown:
                    # slowdown_mode = get_slowdown_mode(prev_act[:, -1], gripper_vel_deque)
                    # slowdown_mode = (a[-1] > 0.5)  # if the last dim is 1 (this indicates precise motion)
                    # a = a[:-1]  # drop precise label from action

                    # prepare the action for osc vs joint control
                    a = prepare_action(a, env, **kwargs)

                    if noisy_eval: # add noise to the action
                        a = add_noise_to_action(deepcopy(a), **kwargs)

                    if slowdown_mode:  # if slowing down, change the control freq
                        control_freq = kwargs.get("slow_control_freq", 20)
                    else:
                        control_freq = kwargs.get("fast_control_freq", 20)

                    # Actually step the action
                    kwargs['control_freq'] = control_freq
                    next_obs, r, done, _ = env.step(a, **kwargs)

                    if next_obs['robot0_gripper_qvel'].ndim > 1:
                        gripper_vel = next_obs['robot0_gripper_qvel'][-1]
                    else:
                        gripper_vel = next_obs['robot0_gripper_qvel']
                    gripper_vel_deque.append(gripper_vel)

                    traj['executed_actions'].append(a)

                    if return_obs:
                        save_obs(traj, next_obs, **kwargs)

                    # For rendering
                    video_annotation = f"rollout: {kwargs['rollout_number']}. control_freq: {control_freq}, t: {num_actual_actions}"
                    video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env,
                                                                text=video_annotation)
                    num_actual_actions += 1
                    actions_remaining -= 1
                    prev_action_index += 1
                    current_action_index += 1
                    if done:
                        break

            # We switch over to executing actions from the new prediction
            for j in range(kwargs["execute_n_actions"]):
                if done:
                    break

                a = act[current_action_index]
                traj['predicted_actions'].append(a)

                # Check if we need to slowdown:
                # slowdown_mode = get_slowdown_mode(act[:, -1], gripper_vel_deque)
                # slowdown_mode = (a[-1] > 0.5)  # if the last dim is 1 (this indicates precise motion)
                # a = a[:-1]  # drop precise label from action

                # prepare the action for osc vs joint control
                a = prepare_action(a, env, **kwargs)

                # if slowing down, change the control freq
                if slowdown_mode:
                    control_freq = kwargs.get("slow_control_freq", 20)
                else:
                    control_freq = kwargs.get("fast_control_freq", 20)

                # add noise to the action
                if noisy_eval:
                    a = add_noise_to_action(deepcopy(a), **kwargs)

                # run the action
                kwargs['control_freq'] = control_freq
                next_obs, r, done, _ = env.step(a, **kwargs)

                # collect gripper velocity for slowdown heuristic
                if next_obs['robot0_gripper_qvel'].ndim > 1:
                    gripper_vel = next_obs['robot0_gripper_qvel'][-1]
                else:
                    gripper_vel = next_obs['robot0_gripper_qvel']
                gripper_vel_deque.append(gripper_vel)
                
                traj['executed_actions'].append(a)

                # state saving
                if return_obs:
                    save_obs(traj, next_obs, **kwargs)

                # For rendering
                video_annotation = f"rollout: {kwargs['rollout_number']}. control_freq: {control_freq}, t: {num_actual_actions}"
                video_count = write_video_constant_sim_time(video_writer, video_count, camera_names, env,
                                                            text=video_annotation)
                num_actual_actions += 1
                actions_remaining -= 1
                current_action_index += 1

            prev_action_index = current_action_index
            prev_act = act

            if done:
                break

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])

            # collect transition
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

            # update for inpainting (if needed) or consistency guiding
            # TODO: Refactor this to be more modular
            if guide_config is not None and guide_config.enabled:
                # prepare the actions to be inpainted/used for consistency guiding
                if guide_config.consistency_loss.enabled:
                    if guide_config.consistency_loss.loss_type == "sparc":
                        assert np.all(np.array(traj["predicted_actions"])[:, :-1] == np.array(traj["executed_actions"])), "Should be same"

                        ws = guide_config.consistency_loss.sparc.window_size # number of predictions
                        
                        if num_inferences < ws-1:
                            continue

                        recent_window_size = (ws-1) * Ta - Td
                        recent_actions     = np.array(traj["predicted_actions"][-recent_window_size:])
                        actions_to_execute = act[prev_action_index:prev_action_index + Td]
                        initial_actions    = np.concatenate([recent_actions, actions_to_execute])

                    elif guide_config.consistency_loss.loss_type in ["mse", "weighted_mse"]:
                        initial_actions = act[prev_action_index:prev_action_index + guide_config.n_actions_ref]
                    else:
                        raise ValueError(f"Unsupported consistency loss type: {guide_config.consistency_loss.loss_type}")
                    
                elif guide_config.inpainting.enabled:
                    # TODO: inpainting on more actions really does not make sense
                    initial_actions = act[prev_action_index:prev_action_index + Td]
                
                elif guide_config.cfg.enabled:
                    assert policy.policy.algo_config.future_action_condition.enabled, "CFG requires trained model based on future action condition"
                    # TODO: this could be expanded to support variable length of actions
                    Tf = policy.policy.algo_config.future_action_condition.horizon
                    initial_actions = act[prev_action_index:prev_action_index + Tf]
                else:
                    raise ValueError("No guiding mode selected")

                initial_actions = process_actions_for_guiding(initial_actions, policy)
                kwargs["guide_actions"] = initial_actions
                    

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=num_actual_actions, Success_Rate=float(success))

    action_sequence_length = policy.policy.algo_config.horizon.action_horizon
    traj['action_sequence_length'] = action_sequence_length

    traj['total_inference_time'] = total_inference_time
    traj['num_inferences'] = num_inferences
    traj['num_executed_actions'] = num_actual_actions
    traj['return'] = total_reward
    traj['rollout_time'] = time.time() - start_rollout
    traj['success'] = float(success)
    sim_steps, sim_time_elapsed = env.getSimTimeInfo()
    traj['total_sim_time_elapsed'] = sim_time_elapsed
    return stats, traj

def process_actions_for_guiding(actions, policy:RolloutPolicy):
    """
    Process actions for guiding: normalize and convert to torch tensor
    """
    ac_norm_stats = policy.action_normalization_stats
    device = policy.policy.device
    actions = TensorUtils.to_tensor(actions)
    actions = TensorUtils.to_device(actions, device)

    if ac_norm_stats is not None:
        assert len(ac_norm_stats.keys()) == 1, "Only one action key is supported for now"

        ac_key = list(ac_norm_stats.keys())[0]
        # ensure obs_normalization_stats are torch Tensors on proper device
        ac_norm_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(ac_norm_stats), device))
        # limit normalization to obs keys being used, in case environment includes extra keys
        actions_dict = {ac_key: actions}
        actions_dict = ObsUtils.normalize_dict(actions_dict, normalization_stats=ac_norm_stats)
        actions = actions_dict[ac_key]

    return actions

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None,
            init_state_dict = None, guide_config = None, **kwargs):
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
        init_state_dict (dict): state dictionary to reset the environment

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    # Use separate function to rollout open loop bc rnn and diffusion policy TODO make this more robust

    # if (policy.policy.global_config.ALGO_NAME == "bc"):
    #     if policy.policy.algo_config.rnn.enabled:
    #         if policy.policy.algo_config.rnn.open_loop:
    #             return rollout_open_loop_bc_rnn(policy, env, horizon, render, video_writer, video_skip, return_obs, camera_names, **kwargs)
    if (policy.policy.global_config.ALGO_NAME == "diffusion_policy"):
        return rollout_diffusion_policy(policy, env, horizon, render, video_writer, video_skip, return_obs,
                                        camera_names, init_state_dict=init_state_dict, guide_config = guide_config, **kwargs)
    else:
        raise Exception("This script only works with Diffusion policy at the moment")



def run_trained_agent(args, **kwargs):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)  # either on-screen or video but not both
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
    if args.control_freq is not None:
        ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"] = args.control_freq

    

    if kwargs["joint_position_control"]:
        from robomimic.robosuite_configs.paths import joint_position_nadun as jp_path
        joint_controller_fp = jp_path()
        controller_configs = json.load(open(joint_controller_fp))
        ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"] = controller_configs
    else:
        ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['control_delta'] = kwargs.get("control_delta",
                                                                                                    True)
    
    # Set torque and kp
    ckpt_dict["env_metadata"]['env_kwargs']['torque_scale'] = kwargs.get("torque_scale", 1.0)
    ckpt_dict["env_metadata"]['env_kwargs']['controller_configs']['kp'] = kwargs['kp']

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
        # Added to fix GPU randomness
        torch.cuda.manual_seed(args.seed)
        # Added to remove torch algorithmic randomness
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    # maybe load initialization states for environment

    if args.init_states_path is not None:
        with open(args.init_states_path, "rb") as f:
            init_states = pickle.load(f)
    else:
        init_states = None

    if args.config is not None:
        guide_config = config_factory("guided_diffusion_policy", dic=None).guide
        with open(args.config, "rb") as f:
            guide_config_dict = json.load(f)["guiding"]
        with guide_config.values_unlocked():
            guide_config.update(guide_config_dict)
        
        if guide_config.consistency_loss.enabled:
            if guide_config.consistency_loss.loss_type == "sparc":
                if policy.action_normalization_stats is not None:
                    guide_config.consistency_loss.unlock_keys()
                    action_dict = policy.action_normalization_stats[list(policy.action_normalization_stats.keys())[0]]
                    guide_config.consistency_loss.action_scale = action_dict["scale"]
                    guide_config.consistency_loss.action_offset = action_dict["offset"]
                    guide_config.consistency_loss.lock_keys()

    else:
        guide_config = None

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # do we write the rollout trajs?
    write_dataset = (args.dataset_path is not None)

    return_obs = kwargs.get("return_obs", False)

    rollout_stats = []
    rollout_trajs = {}
    start = time.time()
    c_freq = ckpt_dict["env_metadata"]["env_kwargs"]["control_freq"]
    print(f"Evaluating control frequency: {c_freq}")
    for i in tqdm(range(rollout_num_episodes)):
        # if (i % 20) == 0:
        #     print(f"Running rollout episode: {i}")
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
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            # return_obs=(write_dataset and args.dataset_obs) or return_obs,
            camera_names=args.camera_names,
            init_state_dict=init_state_dict,
            guide_config = guide_config,
            **kwargs
        )
        stats["time_taken_in_run_agent"] = time.time() - start_episode
        traj['kwargs'] = kwargs
        rollout_stats.append(stats)
        rollout_trajs[f'demo_{i}'] = traj

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)

    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats[f"Time for {rollout_num_episodes} demos"] = time.time() - start

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        rollout_results = {
            "stats": rollout_stats,
            "rollouts": rollout_trajs,
        }
        with open(args.dataset_path, 'wb') as f:
            pickle.dump(rollout_results, f)
        print("Wrote dataset trajectories and stats to {}".format(args.dataset_path))

    return avg_rollout_stats, rollout_stats, rollout_trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        # default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/square/square_cfg_bm_true_model_epoch_1000.pth",
        # default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/can/can_cfg_bm_off_epoch_1000.pth",
        default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/lift/lift_cfg_bm_off_epoch_1000.pth",
        required=False,
        help="path to saved checkpoint pth file",
    )

    parser.add_argument(
        '--config',
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/guide_template/base_cfg_weight_0.json",
        help="Json file that configures the guiding algorithm for the diffusion policy. This sets only guiding parameters, not the model architecture."
    )

    parser.add_argument(
        '--speed_up_factor',
        type=int,
        default=1,
        help="how much time to speed up"
    )

    parser.add_argument(
        '--init_states_path',
        type=str,
        # default="/home/wjung85/Repo/projects/FastIL/eval_scenarios/square_init_states.pkl",
        # default="/home/wjung85/Repo/projects/FastIL/eval_scenarios/can_init_states.pkl",
        default="/home/wjung85/Repo/projects/FastIL/eval_scenarios/lift_init_states.pkl",
        help="Pickle file that has a list of initialization states for the environment, for consistent eval"
    )

    parser.add_argument(
        '--N_eval',
        type=int,
        default=3,
        help="Number of evaluation of stochastic policy"
    )

    parser.add_argument(
        '--video_dir',
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/videos",
        help="Directory to save videos"
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/logs",
        help="Directory to save the logs"
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
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
        '--debug',
        type=bool,
        default=False,
        help="Run Debug Mode"
    )

    args = parser.parse_args()
    
    # Ta = Td + Tne (closing loop frequency should be equivalent for fair comparison)
    Td  = 4  # steps of delay for inference
    Tne = 4 # steps of executing the actions before inference

    if args.debug:
        args.N_eval = 1
        args.horizon = 400
        args.n_rollouts = 1

    kwargs = { # kwargs for the model
              "return_action_sequence": True,
              # controller args: using osc-absolute control
              "receding_horizon": True,
              "fast_control_freq" : 20*args.speed_up_factor,
              "slow_control_freq" : 20*args.speed_up_factor,
              "kp": 250,
              "torque_scale": 3.0,
              "control_delta": False,
              "joint_position_control": False, 
              "osc_control": True,
              "noisy_eval" : False,
              "execute_n_actions": Tne, "inf_delay": Td,
              'return_obs': True, 'save_high_dim_obs': False,
              }
    
    args.horizon = args.horizon * args.speed_up_factor

    with open(args.config, "r") as f:
        guide_config = json.load(f)
        exp_name = guide_config["experiment"]["name"]
    
    if args.video_dir is None:
        if args.debug:
            args.video_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', f'videos/hparam_cfg_binary_mask_compare_{args.speed_up_factor}'))
        else:
            args.video_dir = os.path.abspath(os.path.join(os.path.dirname(args.agent), '..', 'videos/hparam'))
        os.makedirs(args.video_dir, exist_ok=True)\
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    agent_name = os.path.basename(args.agent).split(".")[0]
    
    for iter in range(args.N_eval):
        filename = f"agent_{agent_name}_guide_{exp_name}_iter_{iter}_{datetime.datetime.now()}"
        args.dataset_path = os.path.join(args.log_dir, f"{filename}.pkl")
        args.video_path = os.path.join(args.video_dir, f"{filename}.mp4")

        run_trained_agent(args, **kwargs)

    
