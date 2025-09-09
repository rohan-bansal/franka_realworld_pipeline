import json
import copy
import math
from collections import OrderedDict

import numpy as np
from collections.abc import Iterable
from libero.utils.env_utils import make_libero_env
from robosuite.wrappers import Wrapper
from atm.utils.env_real_gello import EnvGello
from collections import deque
from copy import deepcopy

class FrameStackWrapper(Wrapper):

    valid_obs_types = ["image"]

    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames, mask, cameras):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert num_frames > 1, "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(num_frames)

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None
        self.masks = mask
        self.cameras = cameras

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history
    
    def _stack_obs(self, obs):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=0)
        return obs_dict

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        frames = { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }
        print(list(f"{frame}: {frames[frame].shape}" for frame in frames))

        return self._stack_obs(frames)

    def cache_obs_history(self):
        self.obs_history_cache = deepcopy(self.obs_history)

    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None

    def reset(self):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset()
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        print("FrameStackWrapper: ", obs["agentview_image"].shape)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        print("FrameStackWrapper: ", obs_ret["agentview_image"].shape)
        return obs_ret, r, done, info

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)

class ObservationWrapper(Wrapper):

    valid_obs_types = ["image"]

    def __init__(self, env, masks, cameras):
        super(ObservationWrapper, self).__init__(env)
        self.masks = masks
        self.cameras = cameras

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._stack_obs(obs)
        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self._stack_obs(obs)
        return obs_dict, reward, done, info

    def _stack_obs(self, obs):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=0)
        return obs_dict


class LiberoImageUpsideDownWrapper(Wrapper):
    def __init__(self, env):
        super(LiberoImageUpsideDownWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs, reward, done, info
    
class ImageUpsideDownWrapper(Wrapper):
    def __init__(self, env):
        super(ImageUpsideDownWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["eye_in_hand_image"] = obs["eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["eye_in_hand_image"] = obs["eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs, reward, done, info

class RealObservationWrapper(ObservationWrapper):
    valid_obs_types = ["image"]

    def __init__(self, env):
        super().__init__(env, None, ["agentview", "hand_in_eye"])

    def _stack_obs(self, obs, axis=0):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=1)  # (b, v, h, w, c)
        return obs_dict

class LiberoObservationWrapper(ObservationWrapper):
    valid_obs_types = ["image"]

    def __init__(self, env, mask, cameras):
        super().__init__(env, mask, cameras)

    def _stack_obs(self, obs, axis=0):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=1)  # (b, v, h, w, c)
        return obs_dict

class LiberoSuccessWrapper(Wrapper):
    def __init__(self, env):
        super(LiberoSuccessWrapper, self).__init__(env)
        self.success = None

    def reset(self):
        obs = self.env.reset()
        self.success = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.success is None:
            self.success = done
        else:
            assert len(self.success) == len(done)
            self.success = [self.success[i] or done[i] for i in range(len(done))]
        info["success"] = list(self.success)
        return obs, reward, done, info


def build_env():
    """
    Build the rollout environment.
    Args:
        img_size: The resolution of the pixel observation.
        env_type: The type of environment benchmark. Choices: ["libero"].
        env_meta_fn: The path to robommimic meta data, which is used to specify the robomimic environments.
        env_name: The name to specify the environments.
        obs_types: The observation types in the returned obs dict in Robomimic
        render_gpu_ids:  The available GPU ids for rendering the images
        vec_env_num: The number of parallel environments
        seed: The random seed environment initialization.

    Returns:
        env: A gym-like environment.
    """
    # if isinstance(img_size, Iterable):
    #     assert len(img_size) == 2
    #     img_h = img_size[0]
    #     img_w = img_size[1]
    # else:
    #     img_h = img_w = img_size

    env = EnvGello(env_name="realworld_gello", 
                    camera_dict={
                        "agentview": {"sn": "213722070937", "type": "RealSense", "width": 640, "height": 360},
                    },
                    gripper_type="robotiq"
    )
    # env = FrameStackWrapper(env, 2, None, ["agentview"])
    # env = RealObservationWrapper(env)

    return env