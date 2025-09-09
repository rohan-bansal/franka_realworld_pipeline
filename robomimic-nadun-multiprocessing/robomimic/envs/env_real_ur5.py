"""
Real robot env wrapper for UR5 robot
"""
#Library imports
import json
import time

import numpy as np


#Robot learning specific imports
import robomimic.envs.env_base as EB
import RobotTeleop
from RobotTeleop.configs.real_ur5_config import RealUR5ServerConfig
import RobotTeleop.utils as U
from RobotTeleop.utils import Rate, RateMeasure, Timers
import robomimic.utils.obs_utils as ObsUtils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy

class EnvRealUR5(EB.EnvBase):
    def __init__(self,
                 env_name="RealUR5Env",
                 render=False,
                 render_offscreen=False,
                 use_image_obs=True,
                 use_depth_obs=False,
                 control_freq=40.0,
                 postprocess_visual_obs=True,
                 debug=False,
                 config=None,
                 obs_shapes = None,
                 data = None):

        #TODO maybe pass in Env Config to init function
        if config is None:
            self.config = RealUR5ServerConfig()
        else:
            self.config = config()  # config should be a class

        ###Setup robot
        from RobotTeleop.robots.ur5_ros_interface import RealUR5Robot
        self.robot_interface = RealUR5Robot(config=self.config)

        self.env_name = env_name
        self.act_dim = self.config.robot.act_dim


        #For control rate
        self.control_freq = self.config.robot.control_freq
        self.rate = Rate(self.control_freq)
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * self.control_freq))
        self.timers = Timers(history=100, disable_on_creation=False)

        #TODO populate kwargs dict properly. Check if needed
        self.camera_names_to_sizes  = {}
        for cam_name in self.config.camera_dict:
            self.camera_names_to_sizes[cam_name] = self.config.camera_dict[cam_name]['image_size']
        self._init_kwargs = {}
        self.postprocess_visual_obs = postprocess_visual_obs

        self.debug  =debug

        self.obs_shapes = obs_shapes

    def step(self, action, need_obs=True, **kwargs):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take, should be in [-1, 1]
            need_obs (bool): if False, don't return the observation, because this
                can involve copying image data around. This allows for more
                flexibility on when observations are retrieved.

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """

        # TODO maybe just use the step function in ur5_ros_interface.py in roboteleop
        # TODO put this assert back after fixing
        # assert len(action.shape) == 1 and action.shape[0] == self.action_dimension, "action has incorrect dimensions"
        # rate-limiting
        self.rate.sleep()
        self.rate_measure.measure()

        self.timers.tic("real_ur5_step")
        self.robot_interface.step(action, **kwargs)


        obs = None
        if need_obs:
            obs = self.get_observation()
        r = self.get_reward()
        done = self.is_done()

        self.timers.toc("real_ur5_step")

        return obs, r, done, {}

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self.robot_interface.reset()
                # TODO: check if the rate measure variable is needed
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * self.control_freq))
        print("Reset the robot interface")

        return self.get_observation()

    def reset_to(self, state):
        """
        Reset to a specific state. On real robot, we visualize the start image,
        and a human should manually reset the scene.

        Reset to a specific simulator state.

        Args:
            state (dict): initial state that contains:
                - image (np.ndarray): initial workspace image

        Returns:
            None
        """
        #TODO the camera name is hard coded in for now, change this if necessart
        assert "top_down_image" in state
        ref_img = cv2.cvtColor(state["front_image"], cv2.COLOR_RGB2BGR)

        print("\n" + "*" * 50)
        print("Reset environment to image shown in left pane")
        print("Press 'c' when ready to continue.")
        print("*" * 50 + "\n")
        while(True):
            # read current image
            cur_img = self.robot_interface.get_camera_frame(camera_name="top_down_image")
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)

            # concatenate frames to display
            img = np.concatenate([ref_img, cur_img], axis=1)

            # display frame
            cv2.imshow('initial state alignment window', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                break

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            raise Exception("on-screen rendering not supported currently")
        if mode == "rgb_array":
            assert (height is None) and (width is None), "cannot resize images"
            assert camera_name in self.camera_names_to_sizes, "invalid camera name"
            return self.robot_interface.get_camera_frame(camera_name=camera_name)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.
        """
        self.timers.tic("get_observation")
        observation = self.robot_interface.save_state(obs)
        observation = self.robot_interface.postprocess_obs(observation, obs_shapes=self.obs_shapes)
        return observation

    ### BELOW ARE DEFAULT FUNCTIONS REQUIRED BY ROBOMIMIC
    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        on real robot this returns numpy.zero
        """
        return dict(states=np.zeros(1))
        # raise Exception("Real robot has no simulation state.")

    def get_reward(self):
        """
        Get current reward. No actual reward for real robot env until completion
        """
        return 0.

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """

        # real robot environments don't usually have a success check - this must be done manually
        return {"task": False}

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.act_dim

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        # return self._env_name

        # for real robot. ensure class name is stored in env meta (as env name) for use with any external
        # class registries
        return self.__class__.__name__

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.UR5_REAL_TYPE

    #TODO implement this correctly
    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        # return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))
        # raise NotImplementedError()

    @classmethod
    def create_for_data_processing(self, cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvRealPanda instance)
        """

        # initialize obs utils so it knows which modalities are image modalities
        assert self.camera_names_to_sizes is not None
        image_modalities = list(self.camera_names_to_sizes.keys())
        obs_modality_specs = {
            "obs": {
                "low_dim": [],  # technically unused, so we don't have to specify all of them
                "image": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            postprocess_visual_obs=False,
            **kwargs,
        )
        raise NotImplementedError()

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        # we don't wrap any env
        return self

    def set_obs_shapes(self, obs_shapes):
        self.obs_shapes = obs_shapes

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self.config, sort_keys=True, indent=4)

    #TODO implement env shutdown
    def close(self):
        """
        Clean up env
        """
        # for c_name in self.cr_interfaces:
        #     self.cr_interfaces[c_name].stop()
        # self.robot_interface.close()
        pass

#For debugging only
if __name__ == '__main__':
    env = EnvRealUR5()





