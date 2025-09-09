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
import os
import json
import h5py
import imageio
import sys
import time
from collections import deque
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from termcolor import cprint
import torch
from wandb.wandb_agent import agent

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.utils.buffer_utils import Rate
# from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS

# TODO: delete later
from robomimic.dev.multi_processing.shared_memory.shared_memory_util import ArraySpec

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=None, real=False, rate_measure=None, rollout_rate=20, receding_horizon=False,
            guide_mode=None, demo=None):
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
        rate_measure: if provided, measure rate of action computation and do not play actions in environment

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    rate = Rate(rollout_rate, name="rollout_rate")

    # Maybe rollout demos actions instead
    if demo is not None:
        rollout_demo = True
        actions = demo['absolute_actions'][:]
        horizon = actions.shape[0]
    else:
        rollout_demo = False

    policy.start_episode()
    obs = env.reset()
    state_dict = dict()
    print(f'HORIZON IS {horizon}')
    if real:
        pass
        # input("ready for next eval? hit enter to continue")
    else:
        state_dict = env.get_state()
        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], obs=[],
                inference_obs=[], inference_action_seq=[], inference_t=[], initial_state_dict=state_dict)

    previous_step = time.time()
    control_loop_times = deque(maxlen=100)

    if return_obs:
        # store observations too
        traj.update(dict(next_obs=[]))
    try:
        t_rollout_start = time.time()
        t_rollout_end = None
        for step_i in range(horizon):
            # HACK: some keys on real robot do not have a shape (and then they get frame stacked)
            t_start = time.time()

            for k in obs:
                if len(obs[k].shape) == 1:
                    obs[k] = obs[k][..., None]

            # TODO: hacking for the parameters input to policy
            # get action from policy, inference is run in a separate thread
            kwargs = {'get_tp': False, 'dt': 1/env.control_rate_hz}
            if rollout_demo:
                act = actions[step_i]
                inference_data = None
            else:
                act, inference_data = policy(ob=obs, parallel_inference=receding_horizon, guide_mode=guide_mode, **kwargs) # dim=(7,), inferen data = {obs, action_seq}

            # print(f"action{act[:3]}, state {obs['eef_pose'][-1, :3]}")
            # print(f"xyz error {act[:3]-obs['eef_pose'][-1, :3]}")

            array_spec_list = []
            for o in obs:

                spec = ArraySpec(name=o, shape=obs[o].shape[1:], dtype=obs[o].dtype)
                array_spec_list.append(spec)
            # TODO: only for timing the lift cube task. Delete later
            if (step_i > 90) and (obs['eef_pose'][-1, 2] > 0.28):
                print(f"Current z axis: {obs['eef_pose'][-1, 2]}")
                t_rollout_end = time.time()
                break

            if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (policy.policy.global_config.algo_name != "diffusion_policy"):
                # joint impedance actions and diffusion policy actions are absolute in the real world
                act = np.clip(act, -1., 1.)

            if rate_measure is not None:
                rate_measure.measure()
                print("time: {}s".format(t2 - t1))
                # dummy reward and done
                r = 0.
                done = False
                next_obs = obs
            else:
                # play action $
                t_env_step = time.time()
                next_obs, r, done, _ = env.step(act)
                # print(f"t env step now is: {1.0/(time.time()-t_env_step)}")
                current = time.time()
                control_time = current - previous_step
                previous_step = current
                control_loop_times.append(control_time)
                # print(f"time for env step {current - t_env_step}")
                # print(f"One iteration rate: {1/(control_time):.2f}, Time between env steps : {control_time:.4f}")
                # print(f"Time taken in step: {time.time() - before_step}")
                # print(f"Infer + control freq : {1/(time.time() - previous_step)}")

                if type(r) == tuple: # TODO: tuple comes from panda_deoxys
                    r = 0

            # import pdb; pdb.set_trace()
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
                        if real:
                            # for real robot, log observations instead of rendering
                            img = np.array(obs[cam_name][-1,...])
                            if img.shape[0] == 3: # reorder to HxWxC
                                img = np.transpose(img, (1,2,0))
                            video_img.append(img) # take the latest observation
                        else:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            # CZY add DP inference datafor debug parallel inference
            if inference_data is not None:
                traj["inference_obs"].append(ObsUtils.unprocess_obs_dict(inference_data["obs"]))
                traj["inference_action_seq"].append(inference_data["action"])
                traj["inference_t"].append(inference_data["t"])
            else:
                cprint(f"Warning: inference data log is None", color="yellow")

            if not real:
                traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))
            else:
                # Keep only low dim states
                if 'agentview_image' in obs:
                    del obs['agentview_image']
                if 'robot0_eye_in_hand_image' in obs:
                    del obs['robot0_eye_in_hand_image']
                # NOTE: hack remove stacked observations
                # import pdb; pdb.set_trace()
                for k, v in obs.items():
                    if len(v.shape) == 2:
                        obs[k] = v[-1, :]
                traj["obs"].append(obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if not real:
                state_dict = env.get_state()

            # TODO: the env has its own sleep so we probably dont need a sleep here
            t_step = time.time() - t_start
            cprint(f"step rate is {1/t_step}", "yellow")

            # rate.sleep()

        if t_rollout_end==None:
            t_rollout_end = time.time()
        print(f"Rollout time {t_rollout_end - t_rollout_start} Total env step: {step_i}")

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    policy.reset()

    stats = dict(
        Return=total_reward,
        Horizon=(step_i + 1),
        Success_Rate=float(success),
        Exception_Rate=float(got_exception),
        time=(time.time() - rollout_timestamp),
    )

    #### Mod for returned DP inference data
    if len(traj["inference_obs"]) > 0:
        traj["inference_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["inference_obs"])

    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset
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

    print(f"Average time taken for control loop : {sum(control_loop_times)/len(control_loop_times)}")
    return stats, traj

def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    if args.rollout_demo is not None:
        demo_rollout = True
        demo_file = h5py.File(args.rollout_demo, 'r')['data']
        if args.n_rollouts is None:
            args.n_rollouts = len(demo_file.keys())
    else:
        demo_rollout = False
    rate_measure = None
    if args.hz is not None:
        import RobotTeleop
        from RobotTeleop.utils import Rate, RateMeasure, Timers
        rate_measure = RateMeasure(name="control_rate_measure", freq_threshold=args.hz)
    
    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    if args.dp_eval_steps is not None:
        assert algo_name == "diffusion_policy"
        log_warning("setting @num_inference_steps to {}".format(args.dp_eval_steps))

        # HACK: modify the config, then dump to json again and write to ckpt_dict
        tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        with tmp_config.values_unlocked():
            if tmp_config.algo.ddpm.enabled:
                tmp_config.algo.ddpm.num_inference_timesteps = args.dp_eval_steps
            elif tmp_config.algo.ddim.enabled:
                tmp_config.algo.ddim.num_inference_timesteps = args.dp_eval_steps
            else:
                raise Exception("should not reach here")

        ckpt_dict['config'] = tmp_config.dump()

    # TODO: hack to test horiozn=12
    tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    # print(tmp_config['algo/horizon/action_horizon'])
    tmp_config['algo']['horizon']['action_horizon'] = 12
    ckpt_dict['config'] = tmp_config.dump()
    # ckpt_dict['config/algo/horizon/action_horizon'] = 12

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True) # TODO: for testing multiprocessing

    print("=======================================================================")
    print(f"Device being used : {device}")
    print("=======================================================================")

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        # read horizon from config
        rollout_horizon = config.experiment.rollout.horizon

    # HACK: assume absolute actions for now if using diffusion policy on real robot
    if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(env_meta=ckpt_dict["env_metadata"]):
        ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # Auto-fill camera rendering info if not specified
    # TODO: why is this done?
    # if args.camera_names is None:
    #     # We fill in the automatic values
    #     env_type = EnvUtils.get_env_type(env=env)
    #     args.camera_names = DEFAULT_CAMERAS[env_type]
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)

    if is_real_robot:
        # on real robot - log some warnings
        need_pause = False
        if "env_name" not in ckpt_dict["env_metadata"]["env_kwargs"]:
            log_warning("env_name not in checkpoint...proceed with caution...")
            need_pause = True
        if ckpt_dict["env_metadata"]["env_name"] != "EnvRealPandaGPRS":
            # we will load EnvRealPandaGPRS class by default on real robot even if agent was collected with different class
            log_warning("env name in metadata appears to be class ({}) different from EnvRealPandaGPRS".format(ckpt_dict["env_metadata"]["env_name"]))
            need_pause = True
        # if need_pause:
        #     ans = input("continue? (y/n)")
        #     if ans != "y":
        #         exit()

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    # rollout_horizon = 1000
    for i in tqdm(range(rollout_num_episodes)):
        try:
            if demo_rollout: # TODO for debugging
                demo_name = f"demo_{i}"
                demo = demo_file[demo_name]
                print(f"running demo {demo_name}")
            else:
                demo = None
            stats, traj = rollout(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
                rate_measure=rate_measure,
                rollout_rate=args.rollout_rate,
                receding_horizon=args.receding_horizon,
                guide_mode=args.guide_mode,
                demo=demo
            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C catched, stop execution")
                print("env rate measure")
                print(env.rate_measure)
                ans = input("success? (y / n)")
                rollout_stats.append((1 if ans == "y" else 0))
                print("*" * 50)
                print("have {} success out of {} attempts".format(np.sum(rollout_stats), len(rollout_stats)))
                print("*" * 50)
                continue
            else:
                sys.exit(0)
        
        if is_real_robot:
            print("TERMINATE WITHOUT KEYBOARD INTERRUPT...")
            ans = input("success? (y / n)")
            stats['Success'] = True if ans == "y" else False
            # rollout_stats.append((1 if ans == "y" else 0))

        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            # ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if "obs" in traj:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            # if args.dataset_obs:
            #     for k in traj["obs"]:
            #         ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            #         ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # DP inference data, with action sequence and input observation
            # import pdb; pdb.set_trace()
            if traj["inference_obs"] is not None:
                for k in traj["inference_obs"]:
                    ep_data_grp.create_dataset("inference_obs/{}".format(k), data=np.array(traj["inference_obs"][k]))
                ep_data_grp.create_dataset("inference_action_seq", data=np.array(traj["inference_action_seq"]))
                ep_data_grp.create_dataset("inference_t", data=np.array(traj["inference_t"]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60. # total time taken for rollouts in minutes
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"]) # number of episodes attempted
    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if args.json_path is not None:
        json_f = open(args.json_path, "w")
        json_f.write(stats_json)
        json_f.close()

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
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
        default=10,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
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
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=None,
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

    # Dump a json of the rollout results stats to the specified path
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="(optional) dump a json of the rollout results stats to the specified path",
    )

    # Dump a file with the error traceback at this path. Only created if run fails with an error.
    parser.add_argument(
        "--error_path",
        type=str,
        default=None,
        help="(optional) dump a file with the error traceback at this path. Only created if run fails with an error.",
    )

    # TODO: clean up this arg
    # If provided, do not run actions in env, and instead just measure the rate of action computation
    parser.add_argument(
        "--hz",
        type=int,
        default=None,
        help="If provided, do not run actions in env, and instead just measure the rate of action computation and raise warnings if it dips below this threshold",
    )

    # TODO: clean up this arg
    # If provided, set num_inference_timesteps explicitly for diffusion policy evaluation
    parser.add_argument(
        "--dp_eval_steps",
        type=int,
        default=None,
        help="If provided, set num_inference_timesteps explicitly for diffusion policy evaluation",
    )

    parser.add_argument(
        "--rollout_rate",
        type=int,
        default=80,
        help="If provided, set the rate of rollout",
    )

    parser.add_argument(
        "--receding_horizon",
        action='store_true',
        default=False,
        help="use receding horizon for DP inference, DP run in parallel thread",
    )

    parser.add_argument(
        '--rollout_demo',
        type=str,
        default=None,
        help="demo to rollout instead of policy"
    )

    parser.add_argument(
        "--guide_mode",
        type=str,
        default=None,
        help="use guide mode for diffusion",
    )

    args = parser.parse_args()

    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot/pick_cube_osc_desired_pos_franka_1106_3/20241107154505/models/model_epoch_200.pth"
    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/pick_cube_osc_desired_pose_1110_ep1000.pt" # this one is dangerous
    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot/pick_cube_joint_reached_pos_shift_franka_1120/20241120164630/models/model_epoch_300.pth"

    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot/pick_cube_osc_desired_pos_franka_1128_remove_idle/20241128200426/models/model_epoch_2000.pth"
    args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot/pick_cube_joint_reached_pos_franka_1208_remove_idle/20241218035013/models/model_epoch_2000.pth"
    # args.agent = "/home/mbronars/rohan/trained_models/models/model_epoch_2000.pth"
    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/dp_trained_models/real_robot/pick_cube_joint_reached_pos_franka_1208_remove_idle/20241208041135/models/latest.pth"
    # args.agent = "/home/mbronars/zhenyang/robomimic-nadun/robomimic/../dp_trained_models/real_robot/wiping_osc_desired_pos_franka_1227/20241227212152/models/latest.pth"

    # args.receding_horizon = True
    args.horizon = 400
    args.n_rollouts = 1
    # args.guide_mode = "inpaint"
    # args.dataset_path = "/home/mbronars/zhenyang/demos/pick_cube_osc_1218_60hz_seq_FF_act12.hdf5"
    # args.dataset_path = "/home/mbronars/zhenyang/demos/test_1218.hdf5"
    # args.rollout_demo = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo.hdf5"

    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            # write traceback to file
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e

    # import threading
    # print("Cleaning active threads and terminate the program")
    # for thread in threading.enumerate():
    #     print(f"Thread name: {thread.name}, Daemon: {thread.daemon}")
    #     if thread.name == "MainThread":
    #         continue
        # thread.join()
