import numpy as np
import click
import time
import h5py
from collections import deque

from Xlib.xobject.colormap import rgb_res
from glfw import FALSE
from termcolor import cprint
from multiprocessing.managers import SharedMemoryManager
import imageio
import json
from tqdm import tqdm
from line_profiler_pycharm import profile

import torch

# Import robomimic assets
from robomimic.dev.multi_processing.shared_memory.shared_memory_util import ArraySpec
from robomimic.dev.multi_processing.envs.rl2_robot_env_process import RL2RobotEnv
from robomimic.utils.time_utils import precise_sleep
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

obs_key_map = {
    "robot_timestamp": "robot_timestamp",
    "joint_positions": "joint_positions",
    "joint_velocities": "joint_velocities",
    "joint_positions_desired": "joint_positions_desired",
    "joint_velocities_desired": "joint_velocities_desired",
    "ee_twist_desired": "ee_twist_desired",
    "ee_pose_desired": "ee_pose_desired",
    "eef_pos": "eef_pos",
    "eef_quat": "eef_quat",
    "eef_axis_angle": "eef_axis_angle",
    "eef_pose": "eef_pose",
    # gripper
    "gripper_state": "gripper_state",
    "gripper_position": "gripper_position",
    "gripper_velocity": "gripper_velocity",
    "gripper_force": "gripper_force",
    "gripper_timestamp": "gripper_timestamp"
}

@click.command()
@click.option('--agent', type=str, required=True, help='Path to saved checkpoint pth file')
@click.option('--n_rollouts', type=int, default=10, help='Number of rollouts')
@click.option('--horizon', type=int, default=200, help='(Optional) override maximum horizon of rollout from the one in the checkpoint')
@click.option('--env', type=str, default=None, help='(Optional) override name of env from the one in the checkpoint, and use it for rollouts')
@click.option('--render', is_flag=True, help='On-screen rendering')
@click.option('--video_path', type=str, default=None, help='(Optional) render rollouts to this video file path')
@click.option('--video_skip', type=int, default=5, help='Render frames to video every n steps')
@click.option('--camera_names', type=str, multiple=True, default=None, help='(Optional) camera name(s) to use for rendering on-screen or to video')
@click.option('--wrist_camera', type=str, required=True, help='none, dual, right, left')
@click.option('--dataset_path', type=str, default=None, help='(Optional) if provided, an hdf5 file will be written at this path with the rollout data')
@click.option('--dataset_obs', is_flag=True, help='Include possibly high-dimensional observations in output dataset hdf5 file')
@click.option('--seed', type=int, default=None, help='(Optional) set seed for rollouts')
@click.option('--json_path', type=str, default=None, help='(Optional) dump a json of the rollout results stats to the specified path')
@click.option('--error_path', type=str, default=None, help='(Optional) dump a file with the error traceback at this path')
@click.option('--hz', type=int, default=None, help='If provided, measure the rate of action computation')
@click.option('--dp_eval_steps', type=int, default=None, help='If provided, set num_inference_timesteps explicitly for diffusion policy evaluation')
@click.option('--rollout_rate', type=int, default=20, help='If provided, set the rate of rollout')
@click.option('--receding_horizon', is_flag=True, default=False, help='Use receding horizon for DP inference, DP run in parallel thread')
@click.option('--action_scheduling', is_flag=True, default=False, help='Use action scheduling (with timestamp)')
@click.option('--rollout_demo', type=str, default=None, help='Demo to rollout instead of policy')
@click.option('--guide_mode', type=str, default=None, help='Use guide mode for diffusion')
def main(agent, n_rollouts, horizon, env, render, video_path, video_skip, camera_names, wrist_camera,
         dataset_path, dataset_obs, seed, json_path, error_path, hz, dp_eval_steps,
         rollout_rate, receding_horizon, action_scheduling, rollout_demo, guide_mode):
    """Main function for running trained agent evaluation.
    1. Launch the robot environment, which returns the observations and receive actions for the inference process.
    2. Run the inference thread in the rollout, which will perform the inference and return the actions for the robot environment.
    3. Perform temporal ensembling (here or still using the rolloutpolicy implementation?)
    """

    # NOTE: need to change accordingly
    camera_config_dict = {"agentview": {"sn": "001039114912",
                                        "type": "Kinect",
                                        "resize": True,
                                        "resize_resolution": (128, 128)},
                          "wrist": {"sn": 14620168,
                                    "fps": 60.0,
                                    "resize": True,
                                    "resize_resolution": (128, 128),
                                    "camera_pos": wrist_camera}
                          }

    write_video = (video_path is not None)
    input("Press Enter to start...")
    if rollout_demo is not None:
        demo_rollout = True
        demo_file = h5py.File(rollout_demo, 'r')['data']
        if n_rollouts is None:
           n_rollouts = len(demo_file.keys())
    else:
        demo_rollout = False
    rate_measure = None
    if hz is not None:
        from RobotTeleop.utils import Rate, RateMeasure, Timers
        rate_measure = RateMeasure(name="control_rate_measure", freq_threshold=hz)

    if dp_eval_steps is not None:
        raise ValueError("Haven't implemented")

    # create environment, not from saved ckpt here
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    # need to be before ckpt loading, TODO: fix
    env = RL2RobotEnv(
        # env setup
        shm_manager=shm_manager,
        frequency=rollout_rate,
        n_obs_steps=2, # config.algo.horizon.observation_horizon, # TODO: change accrodingly
        obs_key_map=obs_key_map,
        # camera setup
        camera_name="agentview",
        camera_config_dict=camera_config_dict,
        save_depth_obs=False,
        max_obs_buffer_size=30,
        # robot control setup
        controller_type="OSC_POSE",
        control_rate_robot=100, # TODO: make sure this higher than env rate
        robot_latency=0.0,
        verbose=False
    )
    env.start()

    # load ckpt dict and get algo name for sanity checks
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=agent)

    # TODO: HACK: modify the config, then dump to json again and write to ckpt_dict
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    with config.values_unlocked():
        config.algo.horizon.action_horizon = 32 # change action horizon
    ckpt_dict['config'] = config.dump()

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("=======================================================================")
    print(f"Device being used : {device}")
    print("=======================================================================")

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # maybe set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in tqdm(range(n_rollouts)):
        try:
            if demo_rollout:  # TODO for debugging
                demo_name = f"demo_{i}"
                demo = demo_file[demo_name]
                print(f"running demo {demo_name}")
            else:
                demo = None
            stats, traj = rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                return_obs=(write_dataset and dataset_obs),
                camera_names=camera_names,
                real=True,
                rate_measure=rate_measure,
                rollout_rate=rollout_rate,
                receding_horizon=receding_horizon,
                action_scheduling=action_scheduling,
                guide_mode=guide_mode,
                demo=demo
            )
        except KeyboardInterrupt:
            print("ctrl-C catched, stop execution")
            print("env rate measure")
            print(env.rate_measure)
            ans = input("success? (y / n)")
            rollout_stats.append((1 if ans == "y" else 0))
            print("*" * 50)
            print("have {} success out of {} attempts".format(np.sum(rollout_stats), len(rollout_stats)))
            print("*" * 50)
            continue

        print("TERMINATE WITHOUT KEYBOARD INTERRUPT...")
        ans = input("success? (y / n)")
        stats['Success'] = True if ans == "y" else False

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

            # DP inference data, with action sequence and input observation
            # import pdb; pdb.set_trace()
            if traj["inference_obs"] is not None:
                for k in traj["inference_obs"]:
                    ep_data_grp.create_dataset("inference_obs/{}".format(k), data=np.array(traj["inference_obs"][k]))
                ep_data_grp.create_dataset("inference_action_seq", data=np.array(traj["inference_action_seq"]))
                ep_data_grp.create_dataset("inference_t", data=np.array(traj["inference_t"]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]  # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]  # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    env.stop()

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60.  # total time taken for rollouts in minutes
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"])  # number of episodes attempted
    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if json_path is not None:
        json_f = open(json_path, "w")
        json_f.write(stats_json)
        json_f.close()

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)  # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(dataset_path))

# @profile
def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=None, real=False, rate_measure=None, rollout_rate=20, receding_horizon=False, action_scheduling=False,
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
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectory.
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
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

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
        input("ready for next eval? hit enter to continue")
    else:
        state_dict = env.get_state()
        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)

    video_count = 0  # video frame counter
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], obs=[],
                inference_obs=[], inference_action_seq=[], inference_t=[], initial_state_dict=state_dict)

    if return_obs:
        # store observations too
        traj.update(dict(next_obs=[]))
    try:
        t_rollout_start = time.time()
        t_rollout_end = None
        dt_fast = 1 / rollout_rate
        dt_ori  = 1/ 20.0
        get_obs = True # only used for action scheduling, to adaptively change the inference rate

        # TODO: hacking for the parameters input to policy, important for action scheduling and vel
        kwargs = {'get_tp': False, 'dt': 1.0 / env.frequency, 'vel_approx': False} # whether approximate vel for execution

        for step_i in range(horizon):
            timestamp = None
            t_start = time.time()

            # Get new observation Deal with the shape
            if get_obs:
                obs = env.get_observation(rgb_seq="chw") # for inference
                print(f"****************************************************retrieving observation*********** with {time.time()-t_start}s")
                for k in obs:
                    if len(obs[k].shape) == 1:
                        obs[k] = obs[k][..., None]
            else:
                obs = None

            # from PIL import Image
            # if obs['wrist_image'].shape[1] == 3:
            #     wrist_im = Image.fromarray(np.transpose(obs['wrist_image'][-1,...], (1,2,0)))
            #     agent_im = Image.fromarray(np.transpose(obs['agentview_image'][-1,...], (1,2,0)))
            #     wrist_im.save("wrist.png")
            #     agent_im.save("agent.png")
            #     print(f"$$$$$$$$$$ save image")

            t_act = time.time()
            if action_scheduling:
                act, get_obs, inference_data = policy(ob=obs, parallel_inference=False, action_scheduling=True, guide_mode=guide_mode, **kwargs)
                timestamp, act = act
            else:
                act, inference_data = policy(ob=obs, parallel_inference=receding_horizon,  guide_mode=guide_mode, **kwargs) # dim=(7,), inferen data = {obs, action_seq}

            t_finish_act = time.time()
            print(f"=======action retrieval time {t_finish_act - t_act}")

            # TODO: set angular vel to 0 for reached model
            # act[9:12] = np.zeros(3)

            # print(f"act {act[:6]} {act[6:]}")
            # TODO: HACK action scaling for position for reached positions
            # ratio_xyz = np.array([1.00393191, 1.33003807, 1.01112061])
            # ratio_xyz = np.array([1.0393191, 1.33003807, 1.021112061])
            # scale_xyz = act[:3] * ratio_xyz
            # scale_xyz = np.clip(scale_xyz, -np.abs(act[:3])-0.01, np.abs(act[:3])+0.01) # limit the change to 1cm
            # act[:3] = scale_xyz

            # print(f"obs time in rollout loop {obs['timestamp'][-1]}")
            # print(f"current pose {obs['eef_pose'][-1,:]}")
            # print(f"action predited {act[:7]}")
            if rollout_demo:
                act = actions[step_i]
                inference_data = None

            if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (policy.policy.global_config.algo_name != "diffusion_policy"):
                # joint impedance actions and diffusion policy actions are absolute in the real world
                act = np.clip(act, -1., 1.)

            # print(f"action is {act[:6]} at time {timestamp}")

            dt = dt_fast
            # logic for slowing down TODO: distinguish between action scheduling and normal exe?
            # if act.shape[1] == 8: # with the precision label
            #     precision = act[-1, -1]
            #     if precision > 0.7:
            #         dt = dt_ori
            #     else:
            #         dt = dt_fast

            if timestamp is None:
                timestamp = time.time() + dt

            env.exec_actions(act, timestamps=timestamp)
            success = env.is_success()["task"]

            print(f"$$$$$ exec action {time.time() - t_finish_act}")

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        if real and (obs is not None):
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

            # CZY add DP inference data for debug parallel inference
            if inference_data is not None:
                traj["inference_obs"].append(ObsUtils.unprocess_obs_dict(inference_data["obs"]))
                traj["inference_action_seq"].append(inference_data["action"])
                traj["inference_t"].append(inference_data["t"])
            else:
                cprint(f"Warning: inference data log is None", color="yellow")

            if not real:
                traj["states"].append(state_dict["states"])

            if obs is not None:
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

            t_step = dt - (time.time() - t_start)
            precise_sleep(t_step)
            cprint(f"step rate is {1/(time.time()-t_start)}", "yellow")

        if t_rollout_end==None:
            t_rollout_end = time.time()
        # print(f"Rollout time {t_rollout_end - t_rollout_start} Total env step: {step_i}")

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    stats = dict(
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

    # print(f"Average time taken for control loop : {sum(control_loop_times)/len(control_loop_times)}")
    return stats, traj


if __name__ == '__main__':
    main()
