"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import numpy as np
from collections import deque

from termcolor import cprint

import robomimic.envs.env_base as EB
from bisect import bisect_left

class EnvWrapper(object):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr

def find_closest_numbers_indices(A, B):
    results = []
    for b in B:
        # Find the position to insert `b` in `A` using binary search
        pos = bisect_left(A, b)

        # Compare neighbors to find the closest element
        if pos == 0:  # `b` is smaller than or equal to the smallest element in `A`
            closest_index = 0
        elif pos == len(A):  # `b` is greater than the largest element in `A`
            closest_index = len(A) - 1
        else:
            # Check the nearest two elements
            left = A[pos - 1]
            right = A[pos]
            closest_index = pos - 1 if abs(left - b) <= abs(right - b) else pos

        # Store the result
        results.append(closest_index)

    return results

class FrameStackWrapper(EnvWrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames):
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
        self.acceleration_factor = 10 # determine queue length according to how fast we accelerate
        self.frame_t_interval = 0.05 # 20hz

        ### TODO: add action padding option + adding action to obs to include action history in obs ###

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None
        # self.obs_history_long = None #
        self.last_obs_timestamp = 0

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
                [init_obs[k][None] for _ in range(self.num_frames*self.acceleration_factor)],
                maxlen=self.num_frames*self.acceleration_factor,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Zhenyang Mod on Dec 9: change the return according to timestamp we have from the queue.
        To match the 20hz observation learning.
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        # return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }
        if "agentview_dev_timestamp" not in self.obs_history.keys():
            # original code, but specify the num_frames of the observation
            # TODO: fix this for normal use. Field in obs_history all queue for SAIL setup.
            all_obs = {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
            for k in all_obs:
                all_obs[k] = all_obs[k][-self.num_frames:]
            return all_obs
        else:
            timestamp_queue = self.obs_history['agentview_dev_timestamp']
            latest_t = timestamp_queue[-1]
            earliest_t = latest_t - (self.num_frames-1)*self.frame_t_interval
            # Timestamps of target observations stack
            obs_timestamp = np.linspace(earliest_t, latest_t, num=self.num_frames, endpoint=True)

            try:
                # find the timestamp that 0.05s later, but it is wrong since we want the obs displacement at 20hz
                id = find_closest_numbers_indices(np.array(timestamp_queue), obs_timestamp)
                id = [-2, -1] # using the latest
                # print(f"id used for obs {id}")
                return {k: np.concatenate(self.obs_history[k], axis=0)[id] for k in self.obs_history}
            except:
                cprint("Can't get the correct IDs", "red")
                return {k: np.concatenate(self.obs_history[k][:-self.num_frames], axis=0) for k in self.obs_history}

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

    def step(self, action, **kwargs):
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
        obs, r, done, info = self.env.step(action, **kwargs)
        # TODO: we are doing this before we don't need obs every time for certain models
        if obs is not None:
            self.update_obs(obs, action=action, reset=False)
            # update frame history
            # import pdb; pdb.set_trace()

            ## Only update observation history if new observation coming in. Tailor for acceleration
            from termcolor import cprint
            # if "agentview_timestamp" in obs.keys():
            # if "time" in obs.keys():
            #     # only update if we get a new timestamp. Don't append observation of same timestep
            #     # import pdb; pdb.set_trace()
            #     if (obs["time"] - self.last_obs_timestamp) >= 0.05: # TODO: force 20hz, need to change to a variable
            #         cprint(f"current timestamp {obs['timestamp']}, prev {self.last_obs_timestamp}", "green")
            #         for k in obs:
            #             # if k == "agentview_timestamp":
            #             #     print(f"appending keys {obs[k]}")
            #             # make sure to have leading dim of 1 for easy concatenation
            #             self.obs_history[k].append(obs[k][None])
            #         # self.last_obs_timestamp = obs["agentview_timestamp"]
            #         self.last_obs_timestamp = obs['timestamp']

            # # import pdb; pdb.set_trace()
            # if "agentview_dev_timestamp" in obs.keys():
            #     # only update if we get a new timestamp. Don't append observation of same timestep
            #     # import pdb; pdb.set_trace()
            #     if (obs["agentview_dev_timestamp"] - self.last_obs_timestamp) >= 0.05: # TODO: force 20hz, need to change to a variable
            #         # cprint(f"current timestamp {obs['agentview_dev_timestamp']}, prev {self.last_obs_timestamp}", "green")
            #         for k in obs:
            #             self.obs_history[k].append(obs[k][None])
            #         self.last_obs_timestamp = obs['agentview_dev_timestamp']
            # else:
            #     for k in obs:
            #         # make sure to have leading dim of 1 for easy concatenation
            #         self.obs_history[k].append(obs[k][None])
            # import pdb; pdb.set_trace()
            if "agentview_dev_timestamp" in obs.keys():
                for k in obs:
                    self.obs_history[k].append(obs[k][None])
                self.last_obs_timestamp = obs['agentview_dev_timestamp']

        obs_ret = self._get_stacked_obs_from_history()
        # print(f"obs test {obs_ret['timesteps']}")
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