"""
This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
"""
import copy
import textwrap
from copy import deepcopy
from collections import OrderedDict, deque

import torch.nn as nn
import torch
import threading
import time

from termcolor import cprint
import numpy as np

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.transform_utils as TransUtils
from robomimic.dev.multi_processing.pose_traj_interpolator import PoseTrajectoryInterpolator
from robomimic.utils.buffer_utils import ObsBufferUpdater, ActionQueue, Rate
from robomimic.utils.torch_utils import axis_angle_to_matrix
from robomimic.utils.transform_utils import interpolate_rotations

# mapping from algo name to factory functions that map algo configs to algo class names
REGISTERED_ALGO_FACTORY_FUNCS = OrderedDict()


def register_algo_factory_func(algo_name):
    """
    Function decorator to register algo factory functions that map algo configs to algo class names.
    Each algorithm implements such a function, and decorates it with this decorator.

    Args:
        algo_name (str): the algorithm name to register the algorithm under
    """
    def decorator(factory_func):
        REGISTERED_ALGO_FACTORY_FUNCS[algo_name] = factory_func
    return decorator


def algo_name_to_factory_func(algo_name):
    """
    Uses registry to retrieve algo factory function from algo name.

    Args:
        algo_name (str): the algorithm name
    """
    return REGISTERED_ALGO_FACTORY_FUNCS[algo_name]


def algo_factory(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.

    Args:
        algo_name (str): the algorithm name

        config (BaseConfig instance): config object

        obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

        ac_dim (int): dimension of action space

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    assert algo_name == config.algo_name

    # use algo factory func to get algo class and kwargs from algo config
    factory_func = algo_name_to_factory_func(algo_name)
    algo_cls, algo_kwargs = factory_func(config.algo)

    # create algo instance
    return algo_cls(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
        **algo_kwargs
    )


class Algo(object):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        for k in obs_key_shapes:
            if "obs" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.obs.values() for obs_key in modality]:
                self.obs_shapes[k] = obs_key_shapes[k]
            if "goal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.goal.values() for obs_key in modality]:
                self.goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in [obs_key for modality in self.obs_config.modalities.subgoal.values() for obs_key in modality]:
                self.subgoal_shapes[k] = obs_key_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        """
        raise NotImplementedError

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets[k][i], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k])
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k], optimizer=self.optimizers[k])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        return batch

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        obs_keys = ["obs", "next_obs", "goal_obs"]
        for k in obs_keys:
            if k in batch and batch[k] is not None:
                batch[k] = ObsUtils.process_obs_dict(batch[k])
                if obs_normalization_stats is not None:
                    batch[k] = ObsUtils.normalize_dict(batch[k], obs_normalization_stats=obs_normalization_stats)
        return batch

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Does some operations (like channel swap, uint8 to float conversion, normalization)
        after @process_batch_for_training is called, in order to ensure these operations
        take place on GPU.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader. Assumed to be on the device where
                training will occur (after @process_batch_for_training
                is called)

            obs_normalization_stats (dict or None): if provided, this should map observation
                keys to dicts with a "mean" and "std" of shape (1, ...) where ... is the
                default shape for the observation.

        Returns:
            batch (dict): postproceesed batch
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_dict(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])

        recurse_helper(batch)
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.nets.eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return self.nets.state_dict()

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict)

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        return "{} (\n".format(self.__class__.__name__) + \
               textwrap.indent(self.nets.__repr__(), '  ') + "\n)"

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    """
    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    """
    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    """
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal (dict): predicted subgoal
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy

        Returns:
            current subgoal (dict): predicted subgoal
        """
        raise NotImplementedError


class RolloutPolicy(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """
    def __init__(self, policy, obs_normalization_stats=None, action_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats
        self.action_normalization_stats = action_normalization_stats

        ## for the support of temporal ensembling
        self.inference_thread_initialized = False
        self.thread_lock = threading.Lock()
        self._running = False
        self._rate = Rate(20.0, name="Rollout inference", log_warning=True)
        print(f"********** Rollout Policy inference runs at 20hz **************")

        self.Ta = self.policy.algo_config.horizon.action_horizon
        self.To = self.policy.algo_config.horizon.observation_horizon

        self.reset()

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()
        self.reset()

    def reset(self):
        if hasattr(self, 'obs_queue'):
            self.obs_queue.clear()
        else:
            self.obs_queue = deque(maxlen=self.To)

        if hasattr(self, 'action_queue'):
            self.action_queue.clear()
        else:
            self.action_queue = deque(maxlen=self.Ta)

        if hasattr(self, 'action_id_queue'):
            self.action_id_queue.clear()
        else:
            self.action_id_queue = deque(maxlen=self.Ta)

        if hasattr(self, 'action_queue_normalized'):
            self.action_queue_normalized.clear()
        else:
            self.action_queue_normalized = deque(maxlen=self.Ta)

        # variables for action scheduling support
        self.action_queue_timestamped = ActionQueue()
        self.action_sequence_normalized_last = None
        self.final_action_timestamp = None
        self.time_budget_inference = 0.12 # TODO: est. duration for get_obs() + inference
        self.is_new_obs = True
        self.last_inference_timestamp = time.time()
        self.during_inference = False
        self.timestamps_last = None

        self.data_logged = None
        self.queue_call_count = 0
        self.temp_var = None

        self.action_step_count = 0 # counter of how many times action is called/return
        self.stop_inference_thread()

    def stop_inference_thread(self):
        """
        Stop the inference thread when the policy is deleted.
        """
        if hasattr(self, '_running'):
            self._running = False
            if hasattr(self, 'inference_thread') and self.inference_thread is not None:
                time.sleep(0.2)
                if self.inference_thread is not None:
                    print(f"stopping thread {self.inference_thread}")
                    self.inference_thread.join()
                self.inference_thread = None
                self.inference_thread_initialized = False

    def get_action_receding_horizon(self, obs_dict, goal_dict=None, **kwargs):
        # with inference thread and temporal ensemble
        # we only accept int speedup

        # import pdb; pdb.set_trace()
        with self.thread_lock:
            # update shared input buffer
            self.obs_dict = obs_dict
            self.goal_dict = goal_dict
            self.kwargs = kwargs
            # self.speedup = speedup

            if not self.inference_thread_initialized:
                self._running = True
                self.inference_thread = threading.Thread(target=self.run_inference_thread)
                self.inference_thread.start()
                self.inference_thread_initialized = True

        while True:
            with self.thread_lock:
                # Waiting for the first action
                if len(self.action_queue) > 0:
                    return self.get_temporal_ensemble_action(dt=kwargs.get('dt', False)), TensorUtils.clone(self.data_logged)
            time.sleep(0.001)

    def get_action_scheduling(self, obs_dict, goal_dict=None, **kwargs):
        # with inference thread and temporal ensemble
        # we only accept int speedup

        # import pdb; pdb.set_trace()
        t_enter = time.time()
        print(f"----------enter action scheduling at {t_enter}")
        with self.thread_lock:
            # update shared input buffer
            self.obs_dict = obs_dict
            self.goal_dict = goal_dict
            if obs_dict is not None:
                self.is_new_obs = True

            self.dt = kwargs.get('dt', None)
            assert self.dt is not None
            self.action_scheduling = True # save timestamped action queue in thread
            self.kwargs = kwargs
            # self.speedup = speedup

            if not self.inference_thread_initialized:
                self._running = True
                self.inference_thread = threading.Thread(target=self.run_inference_thread)
                self.inference_thread.start()
                self.inference_thread_initialized = True

        t_copy = time.time()
        print(f"ccccopying the dict {t_copy - t_enter}, obs dict is None {obs_dict is None}")

        while True:
            if not self.action_queue_timestamped.is_empty():
                # Waiting for the first action
                t_before_action = time.time()
                with self.thread_lock:
                    action = self.action_queue_timestamped.pop_next_action() # (timestamp, action)

                t_action = time.time()
                print(f"gettting action with {t_before_action - t_action}")
                time_to_final_action = self.final_action_timestamp - action[0]
                if time_to_final_action < self.time_budget_inference and not self.during_inference:
                    get_obs = True
                else:
                    get_obs = False

                # NOTE: debug print
                print(f"time to final {time_to_final_action:.3f} get obs {get_obs} curr time {time.time():.3f} final time {self.final_action_timestamp:.3f} act time {action[0]:.3f}")

                if action[0] > time.time():
                    # print(f"time delta is {action[0]-time.time()}, current time {time.time()}, schedule time {action[0]}")
                    return action, get_obs, TensorUtils.clone(self.data_logged)

                if self.action_queue_timestamped.is_empty():
                    action = (None, action[1]) # set timestamp to None, avoid infinite loop (not getting new observation/timestamp)
                    cprint(f"Warning, fail to retrieve future action in the queue", "yellow")
                    return action, get_obs, TensorUtils.clone(self.data_logged) # if we come to last action, then time_to_final will < time_budget
            time.sleep(0.001) # 0.0001 originally don't tune down, otherwise new action might come in faster than dropping

    def get_temporal_ensemble_action(self, m=0.2, dt=False):
        """
        We predict action sequence at each inference step, e.g. at 20hz.
        In the queue we will have #Ta action sequences, and we blend the predictions for each time step together.
        Following ACT paper.
        w_i = exp(-m*i)
        action = sum_{i=0}^{Ta} w_i * action_sequence_j[i], where i is the time step.
        where action_sequence_j is the j-th action sequence in the queue.
        Param:
         - dt: whether to use Feedforward controller. If true, also get the next action, and then perform forward euler to get the velocity approximation
        """

        id_array = np.array(self.action_id_queue)
        id_step = np.where(id_array==self.action_step_count)

        if id_step[0].shape[0] == 0:
            cprint("No match action for given timestep", "yellow")

        num_action_seqs = id_step[0].shape[0] # row dim, length

        # if num_action_seqs == 0:
        #     ## Infernce too slow and exhaust the actions
        #     time.sleep(1.0)
        #     # TODO: make this a while loop and further fix
        #
        #     id_array = np.array(self.action_id_queue)
        #     id_step = np.where(id_array == self.action_step_count)
        #
        #     if id_step[0].shape[0] == 0:
        #         cprint("No match action for given timestep", "yellow")
        #
        #     num_action_seqs = id_step[0].shape[0]  # row dim, length

        # reverse order let old action has more weight, smoother transform. forward order let new action has more weight, more responsive
        # Following ACT, we are using reverse order
        forward_order = np.arange(num_action_seqs)  # [large -> small] -> [a_t, a_t-1, a_t-2, ...]
        reverse_order = np.flip(forward_order, axis=0)  # [small -> large] -> [a_t, a_t-1, a_t-2, ...] (ACT in action weight order)
        # weights = np.exp(-m * reverse_order)
        weights = np.exp(-m*forward_order)
        weights = weights / weights.sum()

        action_to_blend = [] # NOTE: for debugging
        axis_angle_list = []
        num_action, action_dim = self.action_queue[0].shape
        action = np.zeros(action_dim)  # [action Dim]
        for i in range(num_action_seqs):
            action_seq_id = id_step[0][i]
            action_id = id_step[1][i]
            action_i = self.action_queue[action_seq_id][action_id, :]
            action_i = TensorUtils.to_numpy(action_i)
            action += weights[i] * action_i
            action_to_blend.append(action_i)

            if i==0 or i==1:
                axis_angle_list.append(action_i[3:6])

        #### we use the oldest 2 rotation prediction, and get the middle rotation by interpolation ****
        axis1, angle1 = TransUtils.vec2axisangle(axis_angle_list[0])
        axis2, angle2 = TransUtils.vec2axisangle(axis_angle_list[-1])
        # don't need to handle the ambiguity for axis angle when we change it to rotation matrix
        R1 = TransUtils.axisangle2mat(axis1, angle1)
        R2 = TransUtils.axisangle2mat(axis2, angle2)
        # using interpolation as a way to do average
        interpolated_rotation = TransUtils.interpolate_rotations(R1, R2, 2)
        assert interpolated_rotation.shape[0]==3, "Number of rotation not matched"
        mean_axis, mean_angle = TransUtils.mat2axisangle(interpolated_rotation[1])
        mean_axis_angle_vec = TransUtils.axisangle2vec(mean_axis, mean_angle)

        action[3:6] = mean_axis_angle_vec
        # TODO: deal with discrete gripper action with deadzone
        if not hasattr(self, 'previous_gripper'):
            self.previous_gripper = action[-1]
        else:
            deadzone = 0.2
            if np.abs(action[-1]) < deadzone:
                # 0 for decision with low confidence.
                action[-1] = self.previous_gripper
                self.previous_gripper = action[-1]
            else:
                action[-1] = 1.0 if action[-1] > 0.0 else -1.0

        ### Get next action if possible, and then get forward velocity
        if dt:
            next_id_step = np.where(id_array==(self.action_step_count+1))
            if next_id_step[0].shape[0] == 0:
                # no future action present, set vel=0
                velocity = np.zeros(6)
            else:
                next_num_action_seqs = next_id_step[0].shape[0]  # row dim, length
                # reverse order let old action has more weight, smoother transform. forward order let new action has more weight, more responsive
                # Following ACT, we are using reverse order
                forward_order = np.arange(next_num_action_seqs)  # [large -> small] -> [a_t, a_t-1, a_t-2, ...]
                reverse_order = np.flip(forward_order,
                                        axis=0)  # [small -> large] -> [a_t, a_t-1, a_t-2, ...] (ACT in action weight order)
                # weights = np.exp(-m * reverse_order)
                weights = np.exp(-m * forward_order)
                weights = weights / weights.sum()
                next_axis_angle_list = []
                next_action = np.zeros(action_dim)  # [action Dim]
                for i in range(next_num_action_seqs):
                    next_action_seq_id = next_id_step[0][i]
                    next_action_id = next_id_step[1][i]
                    next_action_i = self.action_queue[next_action_seq_id][next_action_id, :]
                    next_action_i = TensorUtils.to_numpy(next_action_i)
                    next_action += weights[i] * next_action_i

                    if i == 0 or i == 1:
                        next_axis_angle_list.append(next_action_i[3:6])

                #### we use the oldest 2 rotation prediction, and get the middle rotation by interpolation ****
                naxis1, nangle1 = TransUtils.vec2axisangle(next_axis_angle_list[0])
                naxis2, nangle2 = TransUtils.vec2axisangle(next_axis_angle_list[-1])
                # don't need to handle the ambiguity for axis angle when we change it to rotation matrix
                nR1 = TransUtils.axisangle2mat(naxis1, nangle1)
                nR2 = TransUtils.axisangle2mat(naxis2, nangle2)
                # using interpolation as a way to do average
                next_interpolated_rotation = TransUtils.interpolate_rotations(nR1, nR2, 2)
                assert next_interpolated_rotation.shape[0] == 3, "Number of rotation not matched"
                nmean_axis, nmean_angle = TransUtils.mat2axisangle(next_interpolated_rotation[1])
                nmean_axis_angle_vec = TransUtils.axisangle2vec(nmean_axis, nmean_angle)

                next_action[3:6] = nmean_axis_angle_vec

                xyz_traj = np.concatenate((action[:3][np.newaxis], next_action[:3][np.newaxis]), axis=0)
                axis_angle_traj = np.concatenate((action[3:6][np.newaxis], next_action[3:6][np.newaxis]), axis=0)
                linear_vel = AcUtils.compute_velocity_base_frame(xyz_traj, dt=dt, max_vel=1.5)
                omega = AcUtils.compute_omega_base_frame(axis_angle_traj, dt=dt)
                twist_base_frame = np.hstack((linear_vel, omega))

                action = np.hstack((action[:6], twist_base_frame, action[-1])) # pos + vel + gripper

        self.queue_call_count += 1
        self.action_step_count += 1

        # import pdb; pdb.set_trace()
        # if self.temp_var == None:
        #     self.temp_var = action_i[3:6]
        # action[3:6] = self.temp_var
        return action # [np.newaxis, :]

    def get_temporal_ensemble_action_ori(self, m=0.2):
        """
        We predict action sequence at each inference step, e.g. at 20hz.
        In the queue we will have #Ta action sequences, and we blend the predictions for each time step together.
        Following ACT paper.
        w_i = exp(-m*i)
        action = sum_{i=0}^{Ta} w_i * action_sequence_j[i], where i is the time step.
        where action_sequence_j is the j-th action sequence in the queue.
        """

        num_action_seqs = len(self.action_queue) - self.queue_call_count # number of action sequences used to calculate the weighted sum of actions
        if num_action_seqs <= 0:
            # this only happens when the queue is not full
            cprint(
                f"Warning: action queue len {len(self.action_queue)} is less than queue call count {self.queue_call_count}, temporal ensemble  disabled, using the latest action sequence",
                "yellow")
            num_action_seqs = 1  # only use the latest action sequence in this special case, can be improved

        # reverse order let old action has more weight, smoother transform. forward order let new action has more weight, more responsive
        # Following ACT, we are using reverse order
        forward_order = np.arange(num_action_seqs)  # [large -> small] -> [a_t, a_t-1, a_t-2, ...]
        reverse_order = np.flip(forward_order, axis=0)  # [small -> large] -> [a_t, a_t-1, a_t-2, ...] (ACT in action weight order)
        # weights = np.exp(-m * reverse_order)
        weights = np.exp(-m*forward_order)
        weights = weights / weights.sum()

        # for whatever number of action sequences in the queue, do the weighted sum
        # action_sequence: [num_action_seqs, action_dim]
        num_action, action_dim = self.action_queue[0].shape
        action = np.zeros(action_dim)  # [action Dim]
        axis_ref = None # used for determined the direction of axis
        axis_avg = np.zeros(3)
        angle_avg = 0.0
        axis_angle_list = []
        for i in range(num_action_seqs):
            # for action sequence generated at time t-i, we take the i-th action (i.e. align with the current time step t)
            # queue_call_count is the offset for t, used when the queue is not updated in time.
            action_id = i + self.queue_call_count
            if action_id >= num_action:
                action_id = num_action - 1
                cprint(
                    f"Warning: running out of action length, query action id {action_id} num of action {num_action} queue call count {self.queue_call_count}",
                    "yellow")
            action_i = self.action_queue[-i - 1][action_id, :]
            action_i = TensorUtils.to_numpy(action_i)
            action += weights[i] * action_i

            # if i == num_action_seqs or i ==num_action_seqs-1:
            if i==0 or i==1:
                axis_angle_list.append(action_i[3:6])

            ### Curation for axis angle #### Wrong, we can't do avg for rotation
            # axis, angle = TransUtils.vec2axisangle(action_i[3:6])
            # # import pdb; pdb.set_trace()
            # if axis_ref is None:
            #     axis_ref = axis
            # else:
            #     if np.dot(axis_ref, axis) < 0.0:
            #         print("reverse")
            #         # if axis is in "reverse" direction to the ref one, change to same direction
            #         axis = -axis
            #         angle = -angle
            #
            # print(f"axis {axis}, angle {angle}")
            # axis_avg += weights[i] * axis
            # angle_avg += weights[i] * angle
            ###################################

        ### Curation for axis angle
        # normalize the axis, and regularize angle to 0-2pi TODO: check
        # axis_norm = axis_avg / np.linalg.norm(axis_avg)
        # angle_reg = angle_avg % (2*np.pi)
        # axis_angle = axis_norm * angle_reg
        # action[3:6] = axis_angle
        # print(f"*****action {axis_norm} angle {angle_reg}")
        # TODO: need testing.......

        #### we use the oldest 2 rotation prediction, and get the middle rotation by interpolation ****
        axis1, angle1 = TransUtils.vec2axisangle(axis_angle_list[0])
        axis2, angle2 = TransUtils.vec2axisangle(axis_angle_list[-1])
        # don't need to handle the ambiguity for axis angle when we change it to rotation matrix
        R1 = TransUtils.axisangle2mat(axis1, angle1)
        R2 = TransUtils.axisangle2mat(axis2, angle2)
        # using interpolation as a way to do average
        interpolated_rotation = TransUtils.interpolate_rotations(R1, R2, 2)
        assert interpolated_rotation.shape[0]==3, "Number of rotation not matched"
        mean_axis, mean_angle = TransUtils.mat2axisangle(interpolated_rotation[1])
        mean_axis_angle_vec = TransUtils.axisangle2vec(mean_axis, mean_angle)

        action[3:6] = mean_axis_angle_vec

        self.queue_call_count += 1

        # import pdb; pdb.set_trace()
        # if self.temp_var == None:
        #     self.temp_var = action_i[3:6]
        # action[3:6] = self.temp_var
        return action # [np.newaxis, :]

    def run_inference_thread(self):

        # denote timestamp and action for inpainting with action scheduling
        action_seq_buffer = None
        action_seq_time_buffer = None
        # TODO: change this for other action type accordingly
        # offset = self.action_normalization_stats['absolute_actions']["offset"][0]
        # scale = self.action_normalization_stats['absolute_actions']["scale"][0]

        while self._running:
            try:
                # cprint("running inference", "blue")
                # t1 = time.time()
                with self.thread_lock:
                    # action sequence: [B, Dim, Ta]
                    if self.obs_dict != None and self.is_new_obs:
                        obs_dict_copy = TensorUtils.clone(self.obs_dict)
                        obs_timestamp = copy.deepcopy(self.obs_dict['timestamp'])  # keep it numpy
                        self.is_new_obs = False
                        print(f"Performing inference at {time.time()}")
                    else:
                        # no new observation, continue
                        # time.sleep(self.final_action_timestamp - self.time_budget_inference - time.)
                        time.sleep(0.001)
                        continue

                    goal_dict_copy = TensorUtils.clone(self.goal_dict)
                    kwargs_copy = copy.deepcopy(self.kwargs)
                    self.during_inference = True

                    # print(f"obs keys {obs_dict_copy.keys()}")
                    # print(f"inference obs dict timestamp {obs_dict_copy['agentview_timestamp']}")

                    # # TODO: for testing the predicted action as observation: test key is right
                    # if len(self.action_queue) > 0:
                    #     action_seq_copy = self.action_queue[-1].detach().clone()
                    #     print(f"action seq copy: {action_seq_copy}")
                    #     obs_dict_copy["actions"] = action_seq_copy[:self.To, :].unsqueeze(0) # first To actions, which is being executed during current inference
                    # else:
                    #     cprint("Warning: action queue is empty, can't use action as observation", "yellow")
                # t2 = time.time()

                #### Inpainting for action scheduling
                if self.action_scheduling:
                    ### Hacky version without interpolation. Select the closest timestamp and actions as initial action for inpainting
                    ### Works like charm for 20hz
                    t1 = obs_timestamp[-1, -1]

                    # ## TODO: Hack for policy trained with reached position, but no timeshift
                    timeshift_num = 0 # default = 0, send action ahead num > 0, send action lagging (dont' do this) num < 0
                    action_timestamped = np.linspace(t1 - timeshift_num*self.dt, t1 + self.dt * (self.Ta - 1 - timeshift_num),
                                                     num=self.Ta)  # last obs time align with first action time
                    # print(action_timestamped)

                    if action_seq_time_buffer is None:
                        # No previous action, disable guiding
                        kwargs_copy['guide_mode'] = None
                    else:
                        idx = np.abs(action_seq_time_buffer-t1).argmin()

                        # print(f"id is {idx} prev time {action_seq_time_buffer[1]} and {action_seq_time_buffer[2]} obs time {t1}")
                        kwargs_copy['initial_actions'] = self.action_queue_normalized[-1][
                                                         idx:,
                                                         :]  # 1 is for time alignment, align [t0,t1] of next prediction to previous prediction [t1,t2]

                    ### Interpolation version, haven't worked yet due to axis angle problem in interpolation and vec2axisangle
                    # # get action timestamp for next action prediction
                    # t1 = obs_timestamp[-1, -1]
                    # # print(f"obs time {t1}, current time {time.time()}")
                    # action_timestamped = np.linspace(t1, t1 + self.dt * (self.Ta - 1),
                    #                                  num=self.Ta)  # last obs time align with first action time
                    #
                    # if kwargs_copy['guide_mode'] == "inpaint":
                    #     t1 = time.time()
                    #     if action_seq_time_buffer is None:
                    #         # No previous action, disable guiding
                    #         kwargs_copy['guide_mode'] = None
                    #     else:
                    #         # create interpolator based on previous action seq and timestamp
                    #         pose_interp = PoseTrajectoryInterpolator(
                    #             times = action_seq_time_buffer,
                    #             poses = action_seq_buffer[:, :6],
                    #         )
                    #
                    #         # determine the inpaint length based on the overlapping
                    #         ta_prev_action = action_seq_time_buffer[-1]
                    #         initial_actions = [] # for inpainting
                    #
                    #         # print(f"act timestamp {action_timestamped} prev timestamp {action_seq_buffer} obs {obs_timestamp}")
                    #         for i in range(self.Ta):
                    #             # check overlapping for prev and current action
                    #             if action_timestamped[i] > ta_prev_action:
                    #                 break
                    #             action_i = np.concatenate((pose_interp(action_timestamped[i]), action_seq_buffer[i][-1:]))
                    #             initial_actions.append(action_i)
                    #
                    #         print(f"initial actions from interp: {initial_actions}")
                    #
                    #         if len(initial_actions) == 0:
                    #             cprint("Waraning: No overlapped actions for inpainting", "yellow")
                    #             kwargs_copy['guide_mode'] = None
                    #         else:
                    #             # normalize the initial actions
                    #             prev_sign = action_seq_buffer[i][3]
                    #             initial_actions = np.array(initial_actions)
                    #             for i in range(initial_actions.shape[0]):
                    #                 axis, angle = TransUtils.vec2axisangle(initial_actions[i, 3:6]) # angle will > 0
                    #                 # TODO: dirty hack. Output axis angle is [-3, 0.14, -0.075], while interpolator gives us [3, -0.14, ]
                    #                 if axis[0] < 0: # all wx in dataset < 0
                    #                     axis = - axis
                    #                 vec = TransUtils.axisangle2vec(axis, angle)
                    #                 initial_actions[i, 3:6] = vec
                    #                 # if axis[0]*prev_sign < 0: # different sign
                    #                 #     axis = -axis
                    #                 #     angle = np.pi - angle
                    #                 #     vec = TransUtils.axisangle2vec(axis, angle)
                    #                 #     initial_actions[i, 3:6] = vec
                    #
                    #             initial_actions_norm = (initial_actions-offset) / scale
                    #
                    #             print(f"initial actions converted: {initial_actions[:2]}") #, \n\nnormed actions {initial_actions_norm},"
                    #                   # (f"\n\n prev_action_norm {self.action_queue_normalized[-1]} \n\n"
                    #             print(f"prev action {self.action_queue[-1][:2]}")
                    #             # obs_dict = {
                    #             #     'eef_pos': initial_actions[:, :3],
                    #             #     'eef_axis_angle': initial_actions[:, 3:6]
                    #             # }
                    #             # obs_dict_norm = self._prepare_observation(obs_dict)
                    #             # initial_actions_norm = np.concatenate((obs_dict_norm['eef_pos'][0,...].to('cpu').numpy(), obs_dict_norm['eef_axis_angle'][0,...].to('cpu').numpy()), axis=1)
                    #             kwargs_copy['initial_actions'] = torch.tensor(initial_actions_norm, device=self.policy.device)
                    #     print(f"process time {time.time() - t1}")
                else:
                    ### Inpainting for seq and temporal ensembling. copy data and run inference outside thread lock
                    if len(self.action_queue) > 0:
                        with self.thread_lock:
                            start_step = self.action_step_count
                            # import pdb; pdb.set_trace()
                            start_id = np.where(np.array(self.action_id_queue[-1]) == start_step)[0][0]
                        inpaint_len = 2
                        kwargs_copy['initial_actions'] = self.action_queue_normalized[-1][
                                                         start_id:start_id + inpaint_len,
                                                         :]  # 1 is for time alignment, align [t0,t1] of next prediction to previous prediction [t1,t2]
                    else:
                        kwargs_copy['guide_mode'] = None

                action_sequence = self.policy.get_action_sequence(obs_dict_copy, goal_dict_copy,
                                                              **kwargs_copy)  # 50ms to run

                # t3 = time.time()
                # Save original normalized action sequence for inpainting
                with self.thread_lock:
                    # # prefill the queue with first action sequence
                    if len(self.action_queue_normalized) == 0:
                        if len(action_sequence.shape) == 3:
                            self.action_queue_normalized.extend(
                                [action_sequence[-1] for _ in range(self.action_queue_normalized.maxlen)])
                        else:
                            self.action_queue_normalized.extend([action_sequence for _ in range(self.action_queue_normalized.maxlen)])

                        self.action_id_queue.extend([list(range(0, self.Ta)) for _ in range(self.action_queue_normalized.maxlen)])
                    else:
                        if len(action_sequence.shape) == 3:
                            self.action_queue_normalized.append(action_sequence[-1])
                        else:
                            self.action_queue_normalized.append(action_sequence)

                    self.action_id_queue.append(list(range(self.action_step_count, self.action_step_count+self.Ta)))

                #### Denormalize to make sure the temporal ensembling works for the axis angle #####
                ac = TensorUtils.to_numpy(action_sequence)
                # print(f"normalized actions {ac}")
                if self.action_normalization_stats is not None:
                    action_keys = self.policy.global_config.train.action_keys
                    action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
                                     self.action_normalization_stats}
                    ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
                    ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
                    action_config = self.policy.global_config.train.action_config
                    for key, value in ac_dict.items():
                        this_format = action_config[key].get('format', None)
                        if this_format == 'rot_6d':
                            rot_6d = torch.from_numpy(value).unsqueeze(0)
                            rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                            ac_dict[key] = rot
                    ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
                action_sequence = ac
                # print(f"unnormalized actions {action_sequence}")
                ####################################

                #### Make sure the axis angle always > 0
                # import  pdb; pdb.set_trace()
                # b, act_len, action_dim = ac.shape
                #
                # ac_copy = copy.deepcopy(ac)
                # for i in range(act_len):
                #     axis_angle = TransUtils.convert_sign_axis_angle(ac[0, i, 3:6])
                #     ac[0, i, 3:6] = axis_angle
                #
                # print(f"bef {ac_copy[0,0,3:6]} after {ac[0,0,3:6]}")
                #####################

                # delete images to save memory, keep low dim
                if 'agentview_image' in obs_dict_copy:
                    del obs_dict_copy['agentview_image']
                if 'robot0_eye_in_hand_image' in obs_dict_copy:
                    del obs_dict_copy['robot0_eye_in_hand_image']
                if 'wrist_image' in obs_dict_copy:
                    del obs_dict_copy['wrist_image']

                #### Get velocity estimation here ####
                if len(action_sequence.shape) == 3:
                    xyz_traj = action_sequence[-1, :, :3]
                    axis_angle_traj = action_sequence[-1, :, 3:6]
                else:
                    xyz_traj = action_sequence[:, :3]
                    axis_angle_traj = action_sequence[:, 3:6]
                linear_vel = AcUtils.compute_velocity_base_frame(xyz_traj, dt=self.dt, max_vel=0.5)
                omega = AcUtils.compute_omega_base_frame(axis_angle_traj, dt=self.dt, max_omega=0.2)
                velocity_sequence = np.hstack((linear_vel, omega))

                ##### Write new action data to queue #########
                # with self.thread_lock:
                #     # prefill the queue with first action sequence
                #     if len(self.action_queue) == 0:
                #         if len(action_sequence.shape) == 3:
                #             self.action_queue.extend(
                #                 [action_sequence[-1] for _ in range(self.action_queue.maxlen)])
                #         else:
                #             self.action_queue.extend([action_sequence for _ in range(self.action_queue.maxlen)])
                #     else:
                #         if len(action_sequence.shape) == 3:
                #             self.action_queue.append(action_sequence[-1])
                #         else:
                #             self.action_queue.append(action_sequence)
                #
                #     self.data_logged = {'t': time.time(), 'obs': TensorUtils.to_numpy(obs_dict_copy),
                #                         'goal': TensorUtils.to_numpy(goal_dict_copy),
                #                         'action': action_sequence}
                #
                #     #### Add timestamped actions ####
                #     if self.action_scheduling:
                #         if len(action_sequence.shape) == 3:
                #             action_sequence = action_sequence[-1]
                #         action_vel_sequence = np.hstack((action_sequence[:, :6], velocity_sequence, action_sequence[:, 6:])) # pose + vel + gripper + others (precision)
                #         for t, action in zip(action_timestamped, action_vel_sequence):
                #             # print(f"test {t}, act {action}")
                #             if t > time.time() or self.action_queue_timestamped.is_empty():
                #                 # don't need "old action". But will need to have when queue is empty
                #                 self.final_action_timestamp = t  # use timestamp final action put into the action queue. {action_timestamped} is not accurate when have condition
                #                 self.action_queue_timestamped.add_action(t, action)
                #
                #     self.queue_call_count = 0  # reset queue call count
                #     self.during_inference = False


                # prefill the queue with first action sequence
                if len(self.action_queue) == 0:
                    if len(action_sequence.shape) == 3:
                        with self.thread_lock:
                            self.action_queue.extend(
                                [action_sequence[-1] for _ in range(self.action_queue.maxlen)])
                    else:
                        with self.thread_lock:
                            self.action_queue.extend([action_sequence for _ in range(self.action_queue.maxlen)])
                else:
                    if len(action_sequence.shape) == 3:
                        with self.thread_lock:
                            self.action_queue.append(action_sequence[-1])
                    else:
                        with self.thread_lock:
                            self.action_queue.append(action_sequence)

                #### Add timestamped actions ####
                if self.action_scheduling:
                    if len(action_sequence.shape) == 3:
                        action_sequence = action_sequence[-1]
                    action_vel_sequence = np.hstack((action_sequence[:, :6], velocity_sequence, action_sequence[:, 6:])) # pose + vel + gripper + others (precision)
                    for t, action in zip(action_timestamped, action_vel_sequence):
                        # print(f"test {t}, act {action}")
                        if t > time.time() or self.action_queue_timestamped.is_empty():
                            # don't need "old action". But will need to have when queue is empty
                            self.final_action_timestamp = t  # use timestamp final action put into the action queue. {action_timestamped} is not accurate when have condition
                            with self.thread_lock:
                                self.action_queue_timestamped.add_action(t, action)

                with self.thread_lock:
                    self.data_logged = {'t': time.time(), 'obs': TensorUtils.to_numpy(obs_dict_copy),
                                        'goal': TensorUtils.to_numpy(goal_dict_copy),
                                        'action': action_sequence}
                self.queue_call_count = 0  # reset queue call count
                self.during_inference = False

                if self.action_scheduling:
                    action_seq_buffer = copy.deepcopy(action_sequence)
                    action_seq_time_buffer = copy.deepcopy(action_timestamped)

                # run DP inference at a constant rate
                # t4 = time.time()

                # print(f" data copy time {t2-t1}, infer time {t3-t2}, postprocess time {t4-t3}")
                if not self.action_scheduling:
                    self._rate.sleep()

            except Exception as e:
                print(f"Error in DP inference thread: {e}")
                break

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        if "timestamp" in ob.keys():
            # don't convert timestamp, otherwise precision will change
            timestamp = ob['timestamp']
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.policy.device))
            # limit normalization to obs keys being used, in case environment includes extra keys
            ob = { k : ob[k] for k in self.policy.global_config.all_obs_keys }
            ob = ObsUtils.normalize_dict(ob, normalization_stats=obs_normalization_stats)

        if "timestamp" in ob.keys():
            # don't convert timestamp, otherwise precision will change
            ob['timestamp'] = timestamp
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def get_action_seq_scheduling(self, ob, goal=None, batch=True, return_all_pred=False, **kwargs):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.
        Updated on Jan 26, only for action scheduling (similar to UMI style)

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
            goal (dict): goal observation
            batch (bool): whether the obs has batch dimension. If yes, we squeeze after preparing obs. Designed for
        """
        ob = self._prepare_observation(ob)

        if goal is not None:
            goal = self._prepare_observation(goal)

        # TODO maybe move this to the class that calls rollout policy
        if kwargs.get("inpaint_first_action", False):
            if self.action_normalization_stats is not None:
                action_keys = self.policy.global_config.train.action_keys
                action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
                                 self.action_normalization_stats}
                ac_dict = AcUtils.vector_to_action_dict(kwargs["first_action"], action_shapes=action_shapes, action_keys=action_keys)
                ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)

                ac_key = list(ac_dict)[0]
                kwargs["first_action"] = ac_dict[ac_key]

        ### Inpainting and guiding ###
        # if self.action_sequence_normalized_last is None:
        #     # No previous action, disable guiding
        #     kwargs['guide_mode'] = None
        # else:
        #     t_last_obs = ob['timestamp'][-1, -1]
        #     idx = np.abs(self.timestamps_last - t_last_obs).argmin()
        #     Tf = self.policy.algo_config.future_action_condition.horizon # length of action condition
        #     print(f"id new {idx} Tf {Tf}")
        #     if len(self.action_sequence_normalized_last.shape) == 3:
        #         kwargs['initial_actions'] = self.action_sequence_normalized_last[-1, idx:, :]  # 1 is for time alignment, align [t0,t1] of next prediction to previous prediction [t1,t2]
        #         # TODO: use a fix length here
        #         if idx + Tf > self.action_sequence_normalized_last.shape[-2]:
        #             kwargs['guide_actions'] = self.action_sequence_normalized_last[-1, -Tf:, :]
        #             action_guide_t = self.timestamps_last[-Tf:]
        #         else:
        #             kwargs['guide_actions'] = self.action_sequence_normalized_last[-1, idx: idx+Tf, :]
        #             action_guide_t = self.timestamps_last[idx: idx+Tf]
        #     else:
        #         kwargs['initial_actions'] = self.action_sequence_normalized_last[idx:, :]
        #         # TODO: use a fix length here
        #         if idx + Tf > self.action_sequence_normalized_last.shape[-2]:
        #             kwargs['guide_actions'] = self.action_sequence_normalized_last[-Tf:, :]
        #             action_guide_t = self.timestamps_last[-Tf:]
        #         else:
        #             kwargs['guide_actions'] = self.action_sequence_normalized_last[idx: idx+Tf, :]
        #             action_guide_t = self.timestamps_last[idx: idx + Tf]

            ########## for testing action guiding #########
            # action_guide, _ = self.denormalize_actions(kwargs['guide_actions'][None,...], return_all_pred)
            # print(f"shape {action_guide.shape} normed shape {kwargs['guide_actions'].shape} ori shape {self.action_sequence_normalized_last.shape}")
            # print("action guide and action t", action_guide_t, "guiding actions", action_guide)

        action_sequence_normalized = self.policy.get_action_sequence(ob, goal, **kwargs)
        action_sequence, ac_all = self.denormalize_actions(action_sequence_normalized, return_all_pred)

        if len(action_sequence.shape) == 3:
            action_sequence = action_sequence[-1]

        dt_ori = kwargs.get('dt_ori', None)
        dt_fast = kwargs.get('dt_fast', None)
        slow_down = kwargs.get('slow_down', None)
        assert dt_ori is not None and dt_fast is not None

        ##### Get velocity sequence #####
        # TODO: change the slow part velocity to 0 according to label
        xyz_traj = action_sequence[:, :3]
        axis_angle_traj = action_sequence[:, 3:6]
        linear_vel = AcUtils.compute_velocity_base_frame(xyz_traj, dt=dt_fast, max_vel=0.5)
        omega = AcUtils.compute_omega_base_frame(axis_angle_traj, dt=dt_fast, max_omega=0.7)
        velocity_sequence = np.hstack((linear_vel, omega))

        #### Get timestamped and velocity (according to the gripper) ####
        t_obs = ob['timestamp'][-1, -1]
        timestamps = np.linspace(t_obs, t_obs + dt_fast * (self.Ta - 1), num=self.Ta)

        if slow_down:
            # gripper heuristic
            gripper_action = action_sequence[:, 6]

            # if gripper state change, use dt_ori for the whole act seq
            if np.any(gripper_action > 0.8) and np.any(gripper_action < -0.8):
                print(f"*************************** slow down actiavat ************")
                timestamps = np.linspace(t_obs, t_obs + dt_ori * (self.Ta - 1), num=self.Ta)
                velocity_sequence = np.zeros_like(velocity_sequence)

            # # using awe label gripper heuristic
            # dt_seq = dt_fast * np.ones(self.Ta)
            # awe_label = action_sequence[:, -1]
            # print(f"awe label {awe_label}")
            # dt_seq[awe_label > 0.9] = dt_ori
            # t_seq = np.cumsum(dt_seq)
            # timestamps_seq = t_obs + t_seq
            # print(f"timestamps seq {timestamps_seq}")

        ##### Get desired action sequence #####
        action_sequence_desired = np.hstack((action_sequence[:, :6], velocity_sequence, action_sequence[:, 6:]))

        ##### Store Current Value for inpainting or Guiding #####
        assert action_sequence_desired.shape[0] == timestamps.shape[0]
        self.action_sequence_normalized_last = action_sequence_normalized.detach()
        self.action_sequence_last = action_sequence
        self.timestamps_last = timestamps

        return timestamps, action_sequence_desired

    def __call__(self, ob, goal=None, batch=True, parallel_inference=False, action_scheduling=False, return_all_pred=False, **kwargs):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
            goal (dict): goal observation
            batch (bool): whether the obs has batch dimension. If yes, we squeeze after preparing obs. Designed for
        """

        assert not (parallel_inference and action_scheduling) # can only do one of this now
        if ob is not None:
            ob = self._prepare_observation(ob)
            with self.thread_lock:
                self.new_obs = True

        if goal is not None:
            goal = self._prepare_observation(goal)

        # TODO maybe move this to the class that calls rollout policy
        if kwargs.get("inpaint_first_action", False):
            if self.action_normalization_stats is not None:
                action_keys = self.policy.global_config.train.action_keys
                action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
                                 self.action_normalization_stats}
                ac_dict = AcUtils.vector_to_action_dict(kwargs["first_action"], action_shapes=action_shapes, action_keys=action_keys)
                ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)

                ac_key = list(ac_dict)[0]
                kwargs["first_action"] = ac_dict[ac_key]

        # import pdb; pdb.set_trace()
        if parallel_inference:
            ac, inference_data = self.get_action_receding_horizon(ob, goal_dict=goal, **kwargs)
        elif action_scheduling:
            action = self.get_action_scheduling(ob, goal_dict=goal, **kwargs) # (time, action)
            return action
        else:
            # Default case
            ## TODO adding in kwargs for action sequence, make this more robust later
            # TODO if we get a sequence of actions from the policy, make sure to unnormalize the whole sequence properly
            # ac, inference_data = self.policy.get_action(obs_dict=ob, goal_dict=goal, parallel_inference=parallel_inference, **kwargs)
            ac, next_ac = self.policy.get_action(obs_dict=ob, goal_dict=goal, **kwargs)
            inference_data = None

            # TODO: since we do repeated sampling, change this now
            ac, ac_all = self.denormalize_actions(ac, return_all_pred)
            next_ac, next_ac_all = self.denormalize_actions(next_ac, return_all_pred)

            ### Test the velocity approximation!
            dt = kwargs.get('dt', None)
            vel_approx = kwargs.get('vel_approx', False)
            if (dt is not None) and vel_approx:
                # we also calculating the one step forward vel for the robot
                xyz_traj = np.concatenate((ac[:3][np.newaxis], next_ac[:3][np.newaxis]), axis=0)
                axis_angle_traj = np.concatenate((ac[3:6][np.newaxis], next_ac[3:6][np.newaxis]), axis=0)
                linear_vel = AcUtils.compute_velocity_base_frame(xyz_traj, dt=dt, max_vel=1.5)
                omega = AcUtils.compute_omega_base_frame(axis_angle_traj, dt=dt)
                twist_base_frame = np.hstack((linear_vel, omega))
                ac = np.hstack((ac[:6], twist_base_frame, ac[-1])) # pos + vel + gripper

        if return_all_pred:
            return ac, ac_all
        else:
            return ac, inference_data

    def denormalize_actions(self, ac, return_all_pred):
        if ac.shape[0] == 1:
            ac = TensorUtils.to_numpy(ac[0])
            ac_all = None
        else:
            ac_all = TensorUtils.to_numpy(ac)
            ac = TensorUtils.to_numpy(ac[0])
        if self.action_normalization_stats is not None:
            action_keys = self.policy.global_config.train.action_keys
            action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
                             self.action_normalization_stats}
            ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
            ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
            action_config = self.policy.global_config.train.action_config
            for key, value in ac_dict.items():
                this_format = action_config[key].get('format', None)
                if this_format == 'rot_6d':
                    rot_6d = torch.from_numpy(value).unsqueeze(0)
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                    ac_dict[key] = rot
            ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
            if return_all_pred and ac_all is not None:
                ac_dict = AcUtils.vector_to_action_dict(ac_all, action_shapes=action_shapes, action_keys=action_keys)
                ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
                action_config = self.policy.global_config.train.action_config
                for key, value in ac_dict.items():
                    this_format = action_config[key].get('format', None)
                    if this_format == 'rot_6d':
                        rot_6d = torch.from_numpy(value).unsqueeze(0)
                        rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                        ac_dict[key] = rot
                ac_all = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)

            return ac, ac_all

    def __del__(self):
        self.stop_inference_thread()
        del self.policy
        print("*******Deleted rollout policy**********")


# import multiprocessing
# class RolloutPolicy(object):
#     """
#     Wraps @Algo object to make it easy to run policies in a rollout loop.
#     """
#     def __init__(self, policy, obs_normalization_stats=None, action_normalization_stats=None):
#         """
#         Args:
#             policy (Algo instance): @Algo object to wrap to prepare for rollouts
#             obs_normalization_stats (dict): optionally pass a dictionary for observation normalization.
#         """
#         self.policy = policy
#         self.obs_normalization_stats = obs_normalization_stats
#         self.action_normalization_stats = action_normalization_stats
#
#         self.inference_process_initialized = False
#         self.process_lock = multiprocessing.Lock()
#         self._running = multiprocessing.Value('b', False)  # Shared flag to control process
#         self._rate = Rate(20.0, name="Rollout inference", log_warning=True)
#         print(f"********** Rollout Policy inference runs at 20hz **************")
#
#         self.Ta = self.policy.algo_config.horizon.action_horizon
#         self.To = self.policy.algo_config.horizon.observation_horizon
#
#         self.action_step_count = 0  # Counter for actions
#         self.reset()
#
#     def start_episode(self):
#         """
#         Prepare the policy to start a new rollout.
#         """
#         self.policy.set_eval()
#         self.policy.reset()
#         self.reset()
#
#     def reset(self):
#         if hasattr(self, 'obs_queue'):
#             self.obs_queue.clear()
#         else:
#             self.obs_queue = deque(maxlen=self.To)
#
#         if hasattr(self, 'action_queue'):
#             self.action_queue.clear()
#         else:
#             self.action_queue = deque(maxlen=self.Ta)
#
#         if hasattr(self, 'action_id_queue'):
#             self.action_id_queue.clear()
#         else:
#             self.action_id_queue = deque(maxlen=self.Ta)
#
#         if hasattr(self, 'action_queue_normalized'):
#             self.action_queue_normalized.clear()
#         else:
#             self.action_queue_normalized = deque(maxlen=self.Ta)
#
#         self.data_logged = None
#         self.queue_call_count = 0
#         self.temp_var = None
#
#         self.action_step_count = 0
#
#         self.stop_inference_process()
#
#     def stop_inference_process(self):
#         """
#         Stop the inference process when the policy is deleted.
#         """
#         if hasattr(self, '_running') and self._running.value:
#             self._running.value = False
#             if hasattr(self, 'inference_process') and self.inference_process is not None:
#                 time.sleep(0.2)
#                 if self.inference_process is not None:
#                     print(f"Stopping process {self.inference_process}")
#                     self.inference_process.join()
#                 self.inference_process = None
#                 self.inference_process_initialized = False
#
#     def get_action_receding_horizon(self, obs_dict, goal_dict=None, **kwargs):
#         with self.process_lock:
#             # Update shared input buffer
#             self.obs_dict = obs_dict
#             self.goal_dict = goal_dict
#             self.kwargs = kwargs
#
#             if not self.inference_process_initialized:
#                 self._running.value = True
#                 self.shared_action_queue = multiprocessing.Queue()
#                 self.inference_process = multiprocessing.Process(
#                     target=self.run_inference_process, args=(self.shared_action_queue, self._running, self.process_lock))
#                 self.inference_process.start()
#                 self.inference_process_initialized = True
#
#         while True:
#             try:
#                 if not self.shared_action_queue.empty():
#                     action = self.shared_action_queue.get()
#                     return self.get_temporal_ensemble_action(action), TensorUtils.clone(self.data_logged)
#             except Exception as e:
#                 print(f"Error retrieving action: {e}")
#             time.sleep(0.0001)
#
#     def get_temporal_ensemble_action(self, m=0.2):
#         """
#         We predict action sequence at each inference step, e.g. at 20hz.
#         In the queue we will have #Ta action sequences, and we blend the predictions for each time step together.
#         Following ACT paper.
#         w_i = exp(-m*i)
#         action = sum_{i=0}^{Ta} w_i * action_sequence_j[i], where i is the time step.
#         where action_sequence_j is the j-th action sequence in the queue.
#         """
#
#         id_array = np.array(self.action_id_queue)
#         id_step = np.where(id_array == self.action_step_count)
#
#         if id_step[0].shape[0] == 0:
#             cprint("No match action for given timestep", "yellow")
#
#         num_action_seqs = id_step[0].shape[0]  # row dim, length
#
#         if num_action_seqs == 0:
#             ## Infernce too slow and exhaust the actions
#             time.sleep(1.0)
#             # TODO: make this a while loop and further fix
#
#             id_array = np.array(self.action_id_queue)
#             id_step = np.where(id_array == self.action_step_count)
#
#             if id_step[0].shape[0] == 0:
#                 cprint("No match action for given timestep", "yellow")
#
#             num_action_seqs = id_step[0].shape[0]  # row dim, length
#
#         print(id_array)
#         print(f"action step count {self.action_step_count} num action steps {num_action_seqs}")
#         # reverse order let old action has more weight, smoother transform. forward order let new action has more weight, more responsive
#         # Following ACT, we are using reverse order
#         forward_order = np.arange(num_action_seqs)  # [large -> small] -> [a_t, a_t-1, a_t-2, ...]
#         reverse_order = np.flip(forward_order,
#                                 axis=0)  # [small -> large] -> [a_t, a_t-1, a_t-2, ...] (ACT in action weight order)
#         # weights = np.exp(-m * reverse_order)
#         weights = np.exp(-m * forward_order)
#         weights = weights / weights.sum()
#
#         action_to_blend = []  # NOTE: for debugging
#         axis_angle_list = []
#         num_action, action_dim = self.action_queue[0].shape
#         action = np.zeros(action_dim)  # [action Dim]
#         for i in range(num_action_seqs):
#             action_seq_id = id_step[0][i]
#             action_id = id_step[1][i]
#             action_i = self.action_queue[action_seq_id][action_id, :]
#             action_i = TensorUtils.to_numpy(action_i)
#             action += weights[i] * action_i
#             action_to_blend.append(action_i)
#
#             if i == 0 or i == 1:
#                 axis_angle_list.append(action_i[3:6])
#
#         #### we use the oldest 2 rotation prediction, and get the middle rotation by interpolation ****
#         axis1, angle1 = TransUtils.vec2axisangle(axis_angle_list[0])
#         axis2, angle2 = TransUtils.vec2axisangle(axis_angle_list[-1])
#         # don't need to handle the ambiguity for axis angle when we change it to rotation matrix
#         R1 = TransUtils.axisangle2mat(axis1, angle1)
#         R2 = TransUtils.axisangle2mat(axis2, angle2)
#         # using interpolation as a way to do average
#         interpolated_rotation = TransUtils.interpolate_rotations(R1, R2, 2)
#         assert interpolated_rotation.shape[0] == 3, "Number of rotation not matched"
#         mean_axis, mean_angle = TransUtils.mat2axisangle(interpolated_rotation[1])
#         mean_axis_angle_vec = TransUtils.axisangle2vec(mean_axis, mean_angle)
#
#         action[3:6] = mean_axis_angle_vec
#
#         self.queue_call_count += 1
#         self.action_step_count += 1
#
#         # import pdb; pdb.set_trace()
#         # if self.temp_var == None:
#         #     self.temp_var = action_i[3:6]
#         # action[3:6] = self.temp_var
#         return action  # [np.newaxis, :]
#
#     def run_inference_process(self, shared_action_queue, shared_flag, lock):
#         """
#         Inference process logic, adapted from the threading version.
#
#         Args:
#             shared_action_queue (multiprocessing.Queue): Queue to send actions back to the main process.
#             shared_flag (multiprocessing.Value): A shared flag to control process termination.
#             lock (multiprocessing.Lock): A lock for synchronizing access to shared data.
#         """
#         while shared_flag.value:
#             try:
#                 with lock:
#                     # Clone input data for thread safety
#                     obs_dict_copy = TensorUtils.clone(self.obs_dict)
#                     goal_dict_copy = TensorUtils.clone(self.goal_dict)
#                     kwargs_copy = copy.deepcopy(self.kwargs)
#
#                 # Perform inference (50ms operation)
#                 action_sequence = self.policy.get_action_sequence(
#                     obs_dict_copy, goal_dict_copy, **kwargs_copy
#                 )
#
#                 with lock:
#                     # Prefill action queue with the first sequence if empty
#                     if len(self.action_queue_normalized) == 0:
#                         if len(action_sequence.shape) == 3:
#                             self.action_queue_normalized.extend(
#                                 [action_sequence[-1] for _ in range(self.action_queue_normalized.maxlen)]
#                             )
#                         else:
#                             self.action_queue_normalized.extend(
#                                 [action_sequence for _ in range(self.action_queue_normalized.maxlen)]
#                             )
#
#                         self.action_id_queue.extend(
#                             [list(range(0, self.Ta)) for _ in range(self.action_queue_normalized.maxlen)]
#                         )
#                     else:
#                         if len(action_sequence.shape) == 3:
#                             self.action_queue_normalized.append(action_sequence[-1])
#                         else:
#                             self.action_queue_normalized.append(action_sequence)
#
#                     self.action_id_queue.append(
#                         list(range(self.action_step_count, self.action_step_count + self.Ta))
#                     )
#
#                 # Denormalize action sequence if normalization stats are available
#                 action_sequence = self.denormalize_action_sequence(action_sequence)
#
#                 # Delete images from observation to save memory
#                 if 'agentview_image' in obs_dict_copy:
#                     del obs_dict_copy['agentview_image']
#                 if 'robot0_eye_in_hand_image' in obs_dict_copy:
#                     del obs_dict_copy['robot0_eye_in_hand_image']
#
#                 # Add the latest action to the queue
#                 with lock:
#                     if len(self.action_queue) == 0:
#                         if len(action_sequence.shape) == 3:
#                             self.action_queue.extend(
#                                 [action_sequence[-1] for _ in range(self.action_queue.maxlen)]
#                             )
#                         else:
#                             self.action_queue.extend(
#                                 [action_sequence for _ in range(self.action_queue.maxlen)]
#                             )
#                     else:
#                         if len(action_sequence.shape) == 3:
#                             self.action_queue.append(action_sequence[-1])
#                         else:
#                             self.action_queue.append(action_sequence)
#
#                     # Reset the call count
#                     self.queue_call_count = 0
#
#                     # Log the data
#                     self.data_logged = {
#                         't': time.time(),
#                         'obs': TensorUtils.to_numpy(obs_dict_copy),
#                         'goal': TensorUtils.to_numpy(goal_dict_copy),
#                         'action': action_sequence,
#                     }
#
#                 # Send the action sequence to the shared queue for the main process
#                 shared_action_queue.put(action_sequence)
#
#                 # Sleep to maintain the desired inference rate
#                 self._rate.sleep()
#
#             except Exception as e:
#                 print(f"Error in inference process: {e}")
#                 break
#
#     def _prepare_observation(self, ob):
#         """
#         Prepare raw observation dict from environment for policy.
#
#         Args:
#             ob (dict): single observation dictionary from environment (no batch dimension,
#                 and np.array values for each key)
#         """
#         ob = TensorUtils.to_tensor(ob)
#         ob = TensorUtils.to_batch(ob)
#         ob = TensorUtils.to_device(ob, self.policy.device)
#         ob = TensorUtils.to_float(ob)
#         if self.obs_normalization_stats is not None:
#             # ensure obs_normalization_stats are torch Tensors on proper device
#             obs_normalization_stats = TensorUtils.to_float(
#                 TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.policy.device))
#             # limit normalization to obs keys being used, in case environment includes extra keys
#             ob = {k: ob[k] for k in self.policy.global_config.all_obs_keys}
#             ob = ObsUtils.normalize_dict(ob, normalization_stats=obs_normalization_stats)
#         return ob
#
#     def __repr__(self):
#         """Pretty print network description"""
#         return self.policy.__repr__()
#
#     def __call__(self, ob, goal=None, parallel_inference=False, return_all_pred=False, **kwargs):
#         """
#         Produce action from raw observation dict (and maybe goal dict) from environment.
#
#         Args:
#             ob (dict): single observation dictionary from environment (no batch dimension,
#                 and np.array values for each key)
#             goal (dict): goal observation
#         """
#
#         ob = self._prepare_observation(ob)
#         if goal is not None:
#             goal = self._prepare_observation(goal)
#
#         # TODO maybe move this to the class that calls rollout policy
#         if kwargs.get("inpaint_first_action", False):
#             if self.action_normalization_stats is not None:
#                 action_keys = self.policy.global_config.train.action_keys
#                 action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
#                                  self.action_normalization_stats}
#                 ac_dict = AcUtils.vector_to_action_dict(kwargs["first_action"], action_shapes=action_shapes,
#                                                         action_keys=action_keys)
#                 ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
#
#                 ac_key = list(ac_dict)[0]
#                 kwargs["first_action"] = ac_dict[ac_key]
#
#         # import pdb; pdb.set_trace()
#         if parallel_inference:
#             ac, inference_data = self.get_action_receding_horizon(ob, goal_dict=goal, **kwargs)
#         else:
#             # Default case
#             ## TODO adding in kwargs for action sequence, make this more robust later
#             # TODO if we get a sequence of actions from the policy, make sure to unnormalize the whole sequence properly
#             # ac, inference_data = self.policy.get_action(obs_dict=ob, goal_dict=goal, parallel_inference=parallel_inference, **kwargs)
#             ac, inference_data = self.policy.get_action(obs_dict=ob, goal_dict=goal, **kwargs)
#
#             # TODO: since we do repeated sampling, change this now
#             if ac.shape[0] == 1:
#                 ac = TensorUtils.to_numpy(ac[0])
#                 ac_all = None
#             else:
#                 ac_all = TensorUtils.to_numpy(ac)
#                 ac = TensorUtils.to_numpy(ac[0])
#             if self.action_normalization_stats is not None:
#                 action_keys = self.policy.global_config.train.action_keys
#                 action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
#                                  self.action_normalization_stats}
#                 ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
#                 ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
#                 action_config = self.policy.global_config.train.action_config
#                 for key, value in ac_dict.items():
#                     this_format = action_config[key].get('format', None)
#                     if this_format == 'rot_6d':
#                         rot_6d = torch.from_numpy(value).unsqueeze(0)
#                         rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
#                         ac_dict[key] = rot
#                 ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
#                 if return_all_pred and ac_all is not None:
#                     ac_dict = AcUtils.vector_to_action_dict(ac_all, action_shapes=action_shapes,
#                                                             action_keys=action_keys)
#                     ac_dict = ObsUtils.unnormalize_dict(ac_dict,
#                                                         normalization_stats=self.action_normalization_stats)
#                     action_config = self.policy.global_config.train.action_config
#                     for key, value in ac_dict.items():
#                         this_format = action_config[key].get('format', None)
#                         if this_format == 'rot_6d':
#                             rot_6d = torch.from_numpy(value).unsqueeze(0)
#                             rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
#                             ac_dict[key] = rot
#                     ac_all = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
#
#         if return_all_pred:
#             return ac, ac_all
#         else:
#             return ac, inference_data
#
#     def denormalize_action_sequence(self, action_sequence):
#         """
#         Denormalize the action sequence for proper usage.
#         """
#         ac = TensorUtils.to_numpy(action_sequence)
#         if self.action_normalization_stats is not None:
#             action_keys = self.policy.global_config.train.action_keys
#             action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in
#                              self.action_normalization_stats}
#             ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes,
#                                                     action_keys=action_keys)
#             ac_dict = ObsUtils.unnormalize_dict(ac_dict,
#                                                 normalization_stats=self.action_normalization_stats)
#             action_config = self.policy.global_config.train.action_config
#             for key, value in ac_dict.items():
#                 this_format = action_config[key].get('format', None)
#                 if this_format == 'rot_6d':
#                     rot_6d = torch.from_numpy(value).unsqueeze(0)
#                     rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
#                     ac_dict[key] = rot
#             ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
#         action_sequence = ac
#         return action_sequence
#
#     def __del__(self):
#         self.stop_inference_process()
#         del self.policy
#         print("*******Deleted rollout policy**********")