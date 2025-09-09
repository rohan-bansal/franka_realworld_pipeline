import numpy as np


from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot

from deoxys.utils import YamlConfig, transform_utils

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB


class EnvGello(EB.EnvBase):
    def __init__(
            self, 
            # matching with what we stored in run_env_minimal_tactile_v2.py
            env_name,
            control_freq=1, 
            camera_dict=None,
            gripper_type=None,
            # for robomimic
            postprocess_visual_obs=True,
            **kwargs
        ):
        # create cameras
        cam_dict = self._create_cameras_from_config(camera_dict)

        # create interface with gello_software
        self.robot = PandaRobot("OSC_POSE", gripper_type=gripper_type)
        self.env = RobotEnv(
            self.robot, 
            control_rate_hz=control_freq, 
            camera_dict=cam_dict,
            save_depth_obs=True
        )

        self.postprocess_visual_obs = postprocess_visual_obs

        ObsUtils.initialize_obs_utils_with_obs_specs(
            obs_modality_specs={
                "obs": {
                    "low_dim": ["joint_positions", "gripper_position"],
                    "rgb": ["agentview_image", "hand_in_eye_image"],
                }
            })
        
        self.max_open_distance = 0.039  # Maximum distance in meters when fully open
        self.min_closed_distance = 0.009  # Minimum distance in meters when fully closed

        self.task_emb = np.load("/mnt/data2/mfm/workspace/Franka/gello_software/checkpoints/task_emb/task_emb_bert2.npy", allow_pickle=True).item()["pick up the cube and place it on the other cube"]

    def _create_cameras_from_config(self, cam_config_dict):
        # create cameras
        cam_dict = {}
        if cam_config_dict is not None:
            for cam in cam_config_dict:
                cam_config = cam_config_dict[cam]
                if cam_config["type"] == "Zed":
                    from gello.cameras.zed_camera import ZedCamera
                    cam_dict[cam] = ZedCamera(cam, cam_config)
                elif cam_config["type"] == "RealSense":
                    from  gello.cameras.realsense_camera import RealSenseCamera
                    cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'])
                elif cam_config["type"] == "Kinect":
                    from gello.cameras.kinect_camera import KinectCamera
                    cam_dict[cam] = KinectCamera(cam, cam_config)
        return cam_dict

    def reset(self):
        self.robot.reset()
        return self.get_observation()
    
    def step(self, action):
        # convert action: (pos, axis angle) to (pos, quat)
        assert len(action) == 7
        pos = action[:3]
        axisangle = action[3:6]
        gripper_act = action[-1]
        action = np.array(pos.tolist() + transform_utils.axisangle2quat(axisangle).tolist() + [gripper_act])

        print("DEBUG EnvGello.step: action =", action)
        
        assert len(action) == self.robot.num_dofs()

        self.env.step(action)
        return self.get_observation(), 0, False, {}
    
    def get_observation(self):
        di = self.env.get_obs()
        # print(di.keys())
        # print(ObsUtils.OBS_KEYS_TO_MODALITIES)
        # print(di["gripper_position"], di["gripper_position"].shape)
        # post-process visual obs
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES):
                ret[k] = di[k]
                if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb") or ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb_tactile"):
                    # print("processing rgb", k)
                    # print("old", ret[k].shape)
                    # ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
                    # print("new", ret[k].shape)
                    pass
                elif (k == "gripper_position"):
                    gripper_position = di["gripper_position"]
                    # if not isinstance(gripper_position, np.ndarray):
                    #     gripper_position = np.array([gripper_position])
                    # distance = self.max_open_distance - gripper_position * (self.max_open_distance - self.min_closed_distance)
        

                    ret["gripper_position"] =  np.array([gripper_position, -gripper_position])
                    # print(ret["gripper_position"].shape)
                # NOTE(VS) FileUtils.policy_from_checkpoint() would have initialized ObsUtils using the config
            elif (k == "joint_positions") and ("joint_position" in ObsUtils.OBS_KEYS_TO_MODALITIES):
                # TODO(VS) handling this separately for now; we should return matching keys from env
                #  wrapper in the robot codebase (gello); currently the gello env does not return "joint_position"
                ret["joint_position"] = di["joint_positions"]

        ret["task_emb"] = self.task_emb
        # print("task emb dim", ret["task_emb"].shape)

        # print(di["agentview_image"].shape)
        # print(di["hand_in_eye_image"].shape)
        # ret["image"] = di["agentview_image"]
        # ret["image"] = ObsUtils.process_obs(obs=ret["image"], obs_key="agentview_image")
        # ret["hand_in_eye_image"] = di["hand_in_eye_image"]
        # ret["joint_position"] = di["joint_positions"]
        # ret["eef_pos"] = di["eef_pos"]
        # ret["eef_quat"] = di["eef_quat"]
        # ret["gripper_state"] = di["gripper_position"]
        # ['agentview_image: (2, 360, 640, 3)', 'joint_positions: (2, 7)', 'gripper_position: (2, 2)', 'task_emb: (2, 768)', 'timesteps: (2, 1)', 'actions: (2, 7)']
        # print(ret.keys())
        # print(list(f"{frame}: {ret[frame].shape}" for frame in ret))
        return ret


#########################
# everything below is set just to make EB.EnvBase happy
    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        return

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
        return

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return

    @property
    def version(self):
        """
        Returns version of environment (str).
        This is not an abstract method, some subclasses do not implement it
        """
        return None

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return

    @classmethod
    def create_for_data_processing(cls, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            camera_names ([str]): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards

        Returns:
            env (EnvBase instance)
        """
        return

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return
    
    @property
    def base_env(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return
    