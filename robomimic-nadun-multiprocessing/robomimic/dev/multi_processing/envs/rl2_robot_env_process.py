import time
from typing import Any, Dict, Optional
from copy import deepcopy
from collections import deque

import numpy as np
import math
from multiprocessing.managers import SharedMemoryManager

from scipy.constants import precision

from robomimic.envs.env_base import EnvBase
import robomimic.utils.transform_utils as TransUtils
import robomimic.utils.action_utils as AcUtils

from robomimic.dev.multi_processing.envs.panda_deoxy_process import PandaRobot
from robomimic.dev.multi_processing.envs.kinect_camera_process import KinectCamera
from robomimic.dev.multi_processing.envs.zed_camera_process import ZedCamera
from robomimic.dev.multi_processing.envs.robotipq_2f_gripper_process import Robotiq2FingerGripper

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot
from termcolor import  cprint

class RL2RobotEnv(EnvBase):
    # TODO: the robot environment is abstracted from the robot client to make it possible to use different robots
    # with the same env. Create process for the observation and controller.
    # TODO: add support for video recording later
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        frequency: float = 30.0, # frequency of the stacked obs
        n_obs_steps: int = 2,
        obs_key_map: Dict[str, str] = {},
        # camera setup
        camera_name: str = "agentview",
        camera_cap_fps: float = 30.0,
        camera_config_dict: Dict = {},
        save_depth_obs = False,
        max_obs_buffer_size: int = 30,
        # robot control setup
        controller_type: str = "OSC_POSE",
        control_rate_robot: float = 60.0, # default to 60Hz. control process rate, how fast we send to Deoxys
        robot_latency: float = 0.0,
        # gripper control setup
        control_rate_gripper: float = 100.0,
        gripper_latency: float = 0.0,
        verbose: bool = False,
        # Currently unused to maintain compatibility with robomimic
        env_name=None,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
        use_depth_obs=False,

    ) -> None:

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        print(f"camera config {camera_config_dict}")
        robot = PandaRobot(shm_manager=shm_manager, controller_type=controller_type, control_rate=control_rate_robot, get_max_k=max_obs_buffer_size, verbose=verbose)
        gripper = Robotiq2FingerGripper(shm_manager=shm_manager, frequency=control_rate_gripper, get_max_k=max_obs_buffer_size, verbose=verbose)
        camera = KinectCamera(shm_manager=shm_manager, config=camera_config_dict['agentview'], cam_name=camera_name, get_max_k=max_obs_buffer_size, verbose=verbose)
        # TODO: add camera configs to a consistent dict, and add API
        wrist_camera = ZedCamera(shm_manager=shm_manager, config=camera_config_dict['wrist'], cam_name="wrist", get_max_k=max_obs_buffer_size, verbose=verbose)

        self._robot = robot
        self._gripper = gripper
        self._camera = camera
        self._wrist_camera = wrist_camera

        self.max_obs_buffer_size = max_obs_buffer_size
        self.save_depth_obs = save_depth_obs
        self.frequency = frequency
        self.video_capture_fps = camera_cap_fps
        self.n_obs_steps = n_obs_steps
        self.obs_key_map = obs_key_map
        self.robot_latency = robot_latency
        self.gripper_latency = gripper_latency
        # self.robot_step_times = deque(maxlen=30)
        # self.obs_times = deque(maxlen=30)
        # self.env_step_times = deque(maxlen=30)

    # ===== Start and stop API ===== #
    @property
    def is_ready(self):
        return self._wrist_camera.is_ready and self._camera.is_ready and self._robot.is_ready and self._gripper.is_ready
    
    def start(self, wait: bool=True):
        self._robot.start(wait=False)
        self._gripper.start(wait=False)
        self._camera.start(wait=False)
        self._wrist_camera.start(wait=False)
        if wait:
            self.start_wait()
    
    def start_wait(self):
        self._robot.start_wait()
        self._gripper.start_wait()
        self._camera.start_wait()
        self._wrist_camera.start_wait()

    def stop(self, wait: bool=True):
        self._robot.stop(wait=False)
        self._gripper.stop(wait=False)
        self._camera.stop(wait=False)
        self._wrist_camera.stop(wait=False)
        if wait:
            self.stop_wait()
    
    def stop_wait(self):
        self._robot.stop_wait()
        self._gripper.stop_wait()
        self._camera.stop_wait()
        self._wrist_camera.stop_wait()

    # def start_recording(self):
    #     self._robot.start_recording()
    #     # TODO: add gripper and camera, add episode number
    #
    # def stop_recording(self):
    #     # TODO: very hack now, test
    #     robot_data = self._robot.stop_recording() # list of dict
    #     # import robomimic.utils.tensor_utils as TensorUtils
    #     # robot_data = TensorUtils.list_of_flat_dict_to_dict_of_list(robot_data)
    #
    #     # dataset_path = "/home/mbronars/zhenyang/rollout_data/0126-oven-robot.hdf5"
    #     # data_writer = h5py.File(dataset_path, "w")
    #     # data_grp = data_writer.create_group("data")
    #     # # ep_data_grp = data_grp.create_group("demo_{}".format(i))
    #     # ep_data_grp = data_grp.create_group("demo_0")
    #     # ep_data_grp.create_dataset("eef_pose", data=np.array(robot_data['eef_pose']))
    #     # ep_data_grp.create_dataset("robot_timestamp", data=np.array(robot_data['robot_timestamp']))
    #     # ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    #     return robot_data

    # ===== Environment API ===== #
    def exec_actions(self, actions: np.ndarray,
                     timestamps: np.ndarray,
                     compensate_latency: bool = False):
        
        """Step the environment forward. Only new actions (future timestamps) will be scheduled.

        Args:
            actions shpae: TxD
            actions: when OSC POSE, sequence of [pos, axis_angle, gripper_act]
            actions: when JOINT IMPEDANCE, sequence of [joints, gripper_act]
            timestamps: (T,) scheduled timestamps for the actions
            compensate_latency: if True, compensate for the latency of the robot and gripper

        Returns:
            obs: observation from the environment.
        """
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            if isinstance(timestamps, (int, float)):
                timestamps = np.array([timestamps]) # for single number
            else:
                timestamps = np.array(timestamps)

        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, axis=0) # to match TxD dimension requiremnt

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        if new_actions.shape[0] == 0:
            cprint("No available new actions for execution", "red")
            return 0
        new_timestamps = timestamps[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            if new_actions.shape[1] > 12: # contain vel
                r_action = new_actions[i, :12]
                g_action = new_actions[i, 12]
            else:
                r_action = new_actions[i, :6]
                g_action = new_actions[i, 6]
            r_latency = self.robot_latency if compensate_latency else 0.0
            g_latency = self.gripper_latency if compensate_latency else 0.0
            self._robot.schedule_waypoint(r_action, new_timestamps[i] - r_latency)
            self._gripper.schedule_waypoint(g_action, new_timestamps[i] - g_latency, rescale=True) # map [-1,1] to [1,100]
        # NOTE: add action record here

    def reset(self):
        self._gripper.reset()
        success = self._robot.reset()
        if success:
            time.sleep(3.0) # wait for new observation coming in (essential to keep this amount)
            return self.get_observation()
        else:
            raise ValueError("Reset Robot Env Failed")

    def get_low_dim_observation(self, time_span: float) -> Dict[str, Any]:
            """Get observation from the environment.
            Input: rgb_seq: saved rgb format
                   time_span: the observation history you want to get in (s). Number of obs will be calcuated by time_span and freq

            Returns:
                obs: observation from the environment.
            """
            assert self.is_ready
            observations = {}

            # get data
            # 30 Hz, camera_receive_timestamp
            gripper_frequency = 30.0

            # 100 hz, robot_receive_timestamp
            last_robot_data = self._robot.get_all_state()
            # both have more than n_obs_steps data

            # 30 hz, gripper_receive_timestamp
            last_gripper_data = self._gripper.get_all_state()

            # align gripper obs timestamps (slower one)
            dt = 1 / self.frequency
            num_obs = time_span / dt
            this_timestamps = last_gripper_data['gripper_timestamp']
            last_timestamp = np.max(this_timestamps)
            if time.time() - last_timestamp > 1.0:
                raise ValueError("Gripper Observation Older than 1s")

            # timestamps that used for all other sensors alignment
            obs_align_timestamps = last_timestamp - (np.arange(num_obs)[::-1] * dt)

            # align gripper obs
            gripper_timestamps = last_gripper_data['gripper_timestamp']
            this_timestamps = gripper_timestamps
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            gripper_obs_raw = dict()
            for k, v in last_gripper_data.items():
                if k in self.obs_key_map:
                    gripper_obs_raw[self.obs_key_map[k]] = v

            gripper_obs = dict()
            for k, v in gripper_obs_raw.items():
                gripper_obs[k] = v[this_idxs]

            # align robot obs
            robot_timestamps = last_robot_data['robot_timestamp']
            this_timestamps = robot_timestamps
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            robot_obs_raw = dict()
            for k, v in last_robot_data.items():
                if k in self.obs_key_map:
                    robot_obs_raw[self.obs_key_map[k]] = v

            robot_obs = dict()
            for k, v in robot_obs_raw.items():
                robot_obs[k] = v[this_idxs]

            # return obs
            obs_data = dict(robot_obs)
            obs_data.update(gripper_obs)
            obs_data['timestamp'] = obs_align_timestamps
            return obs_data

    def get_observation(self, rgb_seq="hwc") -> Dict[str, Any]:
        """Get observation from the environment.
        Input: rgb_seq: saved rgb format

        Returns:
            obs: observation from the environment.
        """
        assert self.is_ready
        observations = {}

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency)) + 2 # margin
        last_camera_data = self._camera.get(k=k)
        last_wrist_camera_data = self._wrist_camera.get(k=k)

        # 100 hz, robot_receive_timestamp
        last_robot_data = self._robot.get_all_state()
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = self._gripper.get_all_state()

        # align camera obs timestamps
        dt = 1 / self.frequency
        # last_timestamp = np.max([x['timestamp'][-1] for x in last_camera_data.values()])
        this_timestamps = last_camera_data['timestamp']
        last_timestamp = np.max(this_timestamps)
        if time.time() - last_timestamp > 1.0:
            raise ValueError("Camera Observation Older than 1s")

        # timestamps that used for all other sensors alignment
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        this_idxs = list()

        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        # remap key
        camera_obs[f'{self._camera.cam_name}_image'] = last_camera_data['rgb'][this_idxs]
        if rgb_seq == "chw":
            if len(camera_obs[f'{self._camera.cam_name}_image'].shape) == 3:
                camera_obs[f'{self._camera.cam_name}_image'] = np.transpose(
                    camera_obs[f'{self._camera.cam_name}_image'], (2, 0, 1))
            elif len(camera_obs[f'{self._camera.cam_name}_image'].shape) == 4:
                camera_obs[f'{self._camera.cam_name}_image'] = np.transpose(
                    camera_obs[f'{self._camera.cam_name}_image'], (0, 3, 1, 2))
            else:
                raise ValueError("Camera Obs has wrong shape")

        # align wrist camera obs timestamps
        this_timestamps = last_wrist_camera_data['timestamp']
        if time.time() - np.max(this_timestamps) > 1.0:
            raise ValueError("Wrist Camera Observation Older than 1s")
        wrist_camera_obs = dict()
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        # remap key
        wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'] = last_wrist_camera_data['rgb'][this_idxs]
        if rgb_seq == "chw":
            if len(wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'].shape) == 3:
                wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'] = np.transpose(
                    wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'], (2, 0, 1))
            elif len(wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'].shape) == 4:
                wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'] = np.transpose(
                    wrist_camera_obs[f'{self._wrist_camera.cam_name}_image'], (0, 3, 1, 2))
            else:
                raise ValueError("Camera Obs has wrong shape")

        # align robot obs
        robot_timestamps = last_robot_data['robot_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # align gripper obs
        gripper_timestamps = last_gripper_data['gripper_timestamp']
        this_timestamps = gripper_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        
        gripper_obs_raw = dict()
        for k, v in last_gripper_data.items():
            if k in self.obs_key_map:
                gripper_obs_raw[self.obs_key_map[k]] = v
        
        gripper_obs = dict()
        for k, v in gripper_obs_raw.items():
            gripper_obs[k] = v[this_idxs]

        # accumulate obs, reocrder
        # if self.robot_obs_accumulator is not None:
        #     self.robot_obs_accumulator.put(
        #         robot_obs_raw,
        #         robot_timestamps
        #     )
        # if self.gripper_obs_accumulator is not None:
        #     self.gripper_obs_accumulator.put(
        #         gripper_obs_raw,
        #         gripper_timestamps
        #     )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(dict(wrist_camera_obs))
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    # ===== methods for compatibility with robomimic, not implemented yet ===== #
    def step(self, action):
        self.exec_actions(action, timestamps=np.array([time.time()]))
        return self.get_observation()
    
    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        if self._robot.controller_type == "OSC_POSE":
            return 6
        elif self._robot.controller_type == "JOINT_IMPEDANCE":
            return 7
        assert False, "Controller type not supported"

    @property
    def controller_type(self):
        return self._robot.controller_type
    @property
    def version(self):
        """
        Returns version of environment (str).
        This is not an abstract method, some subclasses do not implement it
        """
        return None

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        # we don't wrap any env
        return self

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        return self.get_observation()

    def render(self, mode="human", height=None, width=None, camera_name=None):
        """Render"""
        return

    def get_state(self):
        """Get environment simulator state, compatible with @reset_to"""
        return

    def get_reward(self):
        """
        Get current reward.
        """
        return

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # real robot environments don't usually have a success check - this must be done manually
        return {"task": False}

    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7

    def name(self):
        """
        Returns name of environment name (str).
        """
        # TODO implement this if needed
        return "PandaDeoxysSimple"

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        from robomimic.envs.env_base import EnvType
        return EnvType.PANDA_DEOXYS_SIMPLE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return

    def create_for_data_processing(
        cls,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            camera_names ([str]): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
            render (bool or None): optionally override rendering behavior. Defaults to False.
            render_offscreen (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_image_obs (bool or None): optionally override rendering behavior. The default value is True if
                @camera_names is non-empty, False otherwise.
            use_depth_obs (bool): if True, use depth observations

        Returns:
            env (EnvBase instance)
        """
        return

    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()


if __name__ == "__main__":

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
        "gripper_timestamp": "gripper_timestamp"
    }

    ################# Args Setup ##################
    policy_cmd_rate = 20 # how fast we get cmd from policy
    # NOTE: higher control rate for process control can lead to smaller position error
    robot_control_rate = 80 # how fast we send out interpolated command in the process
    replay_num = 2 # 10 for exp data collection
    # action_key = "commanded"
    action_key = "reached"
    est_vel = True
    save = False

    import h5py
    import os
    import json
    # rollout_demo = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo.hdf5"
    rollout_demo = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo_remove_idle.hdf5"
    rollout_demo = "/home/mbronars/zhenyang/demos/oven_bowl_demo_0115/_demo.hdf5"
    # rollout_demo = "/home/mbronars/zhenyang/demos/speed_stacking_52_0118/stacking_only.hdf5"

    demo_file = h5py.File(rollout_demo, 'r')['data']
    demo = demo_file['demo_2']
    actions_commanded = demo['absolute_actions'][:]
    if action_key == "commanded":
        actions = actions_commanded
    elif action_key == "reached":
        actions = np.concatenate((demo['obs/eef_pose'][:], actions_commanded[:, 6:7]), axis=1)

    if est_vel:
        dt = 1.0 / policy_cmd_rate
        linear_vel = AcUtils.compute_velocity_base_frame(actions[:, :3], dt=dt, max_vel=1.5)
        omega = AcUtils.compute_omega_base_frame(actions[:, 3:6], dt=dt)
        twist_base_frame = np.hstack((linear_vel, omega))
    else:
        # set default eef pose vel to 0
        twist_base_frame = np.zeros((actions.shape[0], 6))

    print(f"actions shape {actions.shape}, twist shpae {twist_base_frame.shape}")
    horizon = actions.shape[0]
    folder = "/home/mbronars/zhenyang/rollout_data/replay"
    # controller_type = config["controller_type"]
    base_name = os.path.basename(rollout_demo)
    save_path = f"{folder}/0125_{policy_cmd_rate}HZ_{action_key}_replay_{base_name}_Kp_300_500.hdf5"
    controller_type = "OSC_POSE"

    with SharedMemoryManager() as shm_manager:
        serial_number = "001039114912"
        camera_config = {'agentview': {"sn": serial_number, "type": "Kinect", "resize": True, "resize_resolution": (128, 128)},
                         'wrist': {"sn": 14620168, "fps": 60.0, "camera_pos": "left"}}

        env = RL2RobotEnv(
            # env setup
            shm_manager=shm_manager,
            frequency=30,
            n_obs_steps=1,
            obs_key_map=obs_key_map,
            # camera setup
            camera_name="agentview",
            camera_cap_fps=30,
            camera_config_dict=camera_config,
            save_depth_obs=False,
            max_obs_buffer_size=30,
            # robot control setup
            controller_type=controller_type,
            control_rate_robot=robot_control_rate,
            robot_latency=0.0,
            verbose=False
        )
        env.start(wait=True)

        dt = 1 / policy_cmd_rate
        avr_pos_errs = []
        avr_rot_errs = []
        ### for loop for several replay times for data collection. set to 1 by default
        for num in range(replay_num):
            error = []
            rot_error = []
            data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}

            key = input("start demo replaying? (y/n)")
            if key=='y':
                env.reset()
                for i in range(horizon):
                    action = actions[i,]
                    action = np.hstack((action[:6], twist_base_frame[i, :], action[6:]))
                    env.exec_actions(action, timestamps=np.array([dt+time.time()])) # NOTE: important to align the time correctly!!
                    obs = env.get_observation()
                    # print(f"timet at {obs['robot_timestamp']} error is {action - obs['eef_pose']}")
                    # print(f"action is {action}")
                    error.append(np.squeeze(np.abs(action[:3] - obs['eef_pose'][-1,:3])))
                    rot_error.append(TransUtils.geodesic_distance(action[3:6], obs['eef_axis_angle'][-1, :]))

                    data['action'].append(action)
                    data['ee_states'].append(obs['eef_pose'])
                    data['joint_states'].append(obs['joint_positions'])
                    data['gripper_states'].append(obs['gripper_position'])
                    time.sleep(dt)

                avg_pos_error = np.mean(np.array(error), axis=0)
                avg_rot_error = np.mean(np.abs(np.array(rot_error)))
                avr_rot_errs.append(avg_rot_error)
                avr_pos_errs.append(avg_pos_error)
                print(f"avg position error {avg_pos_error} m")
                print(f"avg rot geodesic error {avg_rot_error} rad")

            ############# Save the dataset to subgroup
            if not save:
                print("Not saving the trajectory")
            else:
                with h5py.File(save_path, "a") as h5py_file: # a for modifying the original dataset
                    config_dict = {
                        # "controller_cfg": EasyDict(config["controller_cfg"]),
                        "controller_type": controller_type,
                    }
                    grp = h5py_file.create_group(f"data/demo_{num}")
                    grp.attrs["config"] = json.dumps(config_dict)

                    grp.create_dataset("actions", data=np.array(data["action"]))
                    grp.create_dataset("ee_states", data=np.array(data["ee_states"]))
                    grp.create_dataset("joint_states", data=np.array(data["joint_states"]))
                    # grp.create_dataset("gripper_states", data=np.array(data["gripper_states"]))
                    grp.create_dataset("avr_pos_error", data=avg_pos_error)
                    grp.create_dataset("avr_rot_error", data=avg_rot_error)
                    success = input("replay success?y/n")
                    if success == 'y':
                        grp.create_dataset("success", data=True)
                    else:
                        grp.create_dataset("success", data=False)
                    print(f"Finish replay trajectory saving at {save_path}")

        print(f"avg pos error {np.mean(avr_pos_errs)}, avg rot error {np.mean(avr_rot_errs)}")
        env.stop(wait=True)