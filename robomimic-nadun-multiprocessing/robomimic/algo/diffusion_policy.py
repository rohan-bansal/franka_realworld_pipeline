"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import copy
import numpy as np
import threading
import time

from termcolor import cprint

import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from scipy.interpolate import CubicSpline

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.buffer_utils import ObsBufferUpdater, Rate

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.dev.dataset_processing.enforce_axis_angle_consistency import process_demo_file
from robomimic.dev.hparam.metrics.smoothness_metrics import sparc_torch
from robomimic.utils.deco_utils import time_profile
from tqdm import tqdm

NULL_TOKEN = -1
COND_TOKEN = 1

@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


def blur_noise(naction, blur_dim, blur_length):
    """

    Args:
        naction:
        blur_dim:
        blur_length:

    Returns:

    """
    # TODO: assumption we only blur the first 'blur_length' actions
    # TODO this does not work, smoothing needs to be done better

    B = naction.shape[0]
    t = naction.shape[1]
    D = naction.shape[2]

    naction_blurred = np.ones(naction.shape) * -5

    blur_dim = min(blur_dim, D)
    blur_length = min(t, blur_length)

    X = np.linspace(0, blur_length, blur_length, dtype=int)

    naction = naction.detach().cpu().numpy()

    # Smooth the parts that need to be smoothed
    for b in range(B):  # For every batch
        for d in range(blur_dim):  # For the action dim
            spline = CubicSpline(X, naction[b, :blur_length, d])
            # smooth_noise = spline(X)
            naction_blurred[b, :blur_length, d] = spline(X)

    import pdb
    pdb.set_trace()



    print()


class ExponentialWeightedMSELoss(nn.Module):
    def __init__(self, lambda_=0.5, reduction='none'):
        """
        Args:
            lambda_ (float): Exponential decay rate for weighting.
        """
        super(ExponentialWeightedMSELoss, self).__init__()
        self.lambda_ = lambda_
        self.reduction = reduction


    def forward(self, X, Y):
        """
        Compute the exponentially weighted MSE loss between X and Y.

        Args:
            X (Tensor): The input tensor.
            Y (Tensor): The target tensor you are trying to match.

        Returns:
            Tensor: The weighted loss.
        """
        # Calculate the exponential weights for each element based on their index
        weights = torch.exp(-self.lambda_ * torch.arange(X.size(0), dtype=torch.float, device=X.device))

        out = (X-Y)**2 * weights[None, :, None]
        if self.reduction == 'none':
            return out
        else:
            sum_squared_diff = torch.sum(out, dim=-1)
            loss = torch.mean(sum_squared_diff, dim=-1)
            return loss

class SparcLoss():
    def __init__(self, amp_th=0.05, fc=10, padlevel=0, reduction='none', action_scale=1.0, action_offset=0.0, Td=4, dt=0.05, **kwargs):
        # Fourier transform parameters
        self.amp_th = amp_th
        self.fc = fc
        self.padlevel = padlevel
        self.reduction = reduction
        self.action_scale  = action_scale
        self.action_offset = action_offset

        # TODO: make this to be decided by the policy
        self.dt = dt
        self.Td = Td

        self.log = {}

    def unnormalize_actions(self, actions):
        """
        Unnormalize actions to the original scale
        """
        scale = torch.tensor(self.action_scale, device=actions.device, dtype=actions.dtype)
        offset = torch.tensor(self.action_offset, device=actions.device, dtype=actions.dtype)

        actions_orig = scale * actions + offset

        return actions_orig

    def __call__(self, actions, recent_actions):
        """
        Compute the sparc loss between actions and recent_actions

        Args:
            actions (Tensor): The clean, normalized actions to be executed. (B, Tp-1, Da)
            recent_actions (Tensor): The clean, normalizedrecent actions to be used for consistency. (B, TH, Da)

        Returns:
            Tensor: The sparc loss.

        NOTE:
        - Make sure that the actions are not unnormalized here - we are looking at the correct
        - Try to change the actions to be cut out at `Td: Td + Ta` instead of `Td:`
        """
        assert actions.shape[-2] == 31
        assert recent_actions.shape[-2] == 36 # (ws-1) * Ta

        B = actions.shape[0]

        # concat x1((i-ws)*Ta: i*Ta), x2(Td: Td + Ta)
        # actions_all = torch.cat([recent_actions, actions[:, self.Td:, :]], dim=-2)
        # TODO: Ta is 12, bad practice
        actions_all = torch.cat([recent_actions, actions[:, self.Td+12:, :]], dim=-2)
        actions_all = self.unnormalize_actions(actions_all)

        # Compute the movements of the robot
        poses_all      = actions_all[:, :, :6] # (B, T, 6)
        twists_all     = TorchUtils.pose_to_twist(poses_all, dt=self.dt) # (B, T-1, Da)
        movements      = torch.norm(twists_all, dim=-1) # (B, Tp)

        # Compute the sparc loss (- for maximization)
        sparc_loss, fft, fft_filtered  = sparc_torch(movements, dt = self.dt, fc=self.fc, padlevel=self.padlevel, amp_th=self.amp_th, return_data=True)
        # sparc_loss  = sparc_torch(movements, dt = self.dt, fc=self.fc, padlevel=self.padlevel, amp_th=self.amp_th, return_data=False)
        sparc_loss    = -sparc_loss

        self.log = {
            "sparc_loss": sparc_loss,
            "fft": fft,
            "fft_filtered": fft_filtered,
            "actions": actions,
            "recent_actions": recent_actions,
            "actions_all": actions_all,
            "movements": movements
        }

        if self.reduction == 'none':
            if B == 1:
                sparc_loss = sparc_loss.unsqueeze(0)
            return sparc_loss[:, None, None]
        else:
            return torch.mean(sparc_loss, dim=-1)

class DiffusionPolicyUNet(PolicyAlgo):

    def __init__(self, *args, **kwargs):
        super(DiffusionPolicyUNet, self).__init__(*args, **kwargs)
        ## Support for inference threads
        self.inference_thread_initialized = False
        self.thread_lock = threading.Lock()
        self._running = True
        self._rate = Rate(20.0, name="DP inference", log_warning=True)
        print(f"********** DP inference runs at 20hz **************")

        self.Ta = self.algo_config.horizon.action_horizon
        self.To = self.algo_config.horizon.observation_horizon

        self.obs_queue = deque(maxlen=self.To)
        self.action_queue = deque(maxlen=self.Ta)
        self.data_logged = None
        # number of same action queue calls, i.e. when the queue is the same (no new action sequence added)
        # how many times it is called to generate new action. e.g. from inference t0-t1, maybe 2 or 3 action needs to be generated
        # this is actually the same value of acceleration rate (in ideal sync case)
        self.queue_call_count = 0

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # NOTE: Add CFG-related code here
        # TODO: If variable length inpainting is required, add a mask here
        global_cond_dim = obs_dim * self.algo_config.horizon.observation_horizon
        if "future_action_condition" in self.algo_config.keys() and self.algo_config.future_action_condition.enabled:
            global_cond_dim = global_cond_dim + self.ac_dim * self.algo_config.future_action_condition.horizon
            if "binary_mask" in self.algo_config.future_action_condition.keys() and self.algo_config.future_action_condition.binary_mask:
                global_cond_dim = global_cond_dim + 1

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=global_cond_dim
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(parameters=nets.parameters(), power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self._shadow_nets = copy.deepcopy(self.nets).eval()
        self._shadow_nets.requires_grad_(False)
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

        # For classifier guidance
        self.classifier = None
        self.classifier_scale = 0.0 # how much to weight the classifier guidance

        # For consistency guidance
        # self.consistency_loss = nn.MSELoss(reduction='none')
        # self.consistency_loss = ExponentialWeightedMSELoss()
        # self.consistency_weight = 50.0 # 10.0 is good for mse loss
        # self.eta                = 0.0  # DDPM

        assert self.algo_config.horizon.observation_horizon == self.global_config.train.frame_stack,\
            "the observation horizon needs to be the same as the framestack"

    def set_classifier(self, classifier, classifier_scale):
        self.classifier = classifier
        self.classifier_scale = classifier_scale
    
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
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _get_null_token_actions(self, B, Tf, action_dim):
        """
        Get null token actions for action condition
        """
        if self.algo_config.future_action_condition.null_token == 'zero':
            null_token = torch.zeros((B, Tf*action_dim), device=self.device)
            if "binary_mask" in self.algo_config.future_action_condition.keys() and self.algo_config.future_action_condition.binary_mask:
                binary_mask = NULL_TOKEN * torch.ones((B, 1), device=self.device)
                null_token = torch.cat([null_token, binary_mask], dim=-1)
        else:
            raise ValueError(f"Invalid null token: {self.algo_config.future_action_condition.null_token}")

        return null_token

    def _get_action_condition(self, actions):
        """Helper function to get action condition with optional nullification.

        Args:
            batch (dict): Batch of data containing actions
            B (int): Batch size
            To (int): Observation horizon
            action_dim (int): Action dimension

        Returns:
            torch.Tensor: Action condition tensor
        """

        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        assert actions.ndim == 3 and actions.shape[1] == Tp

        B = actions.shape[0]
        action_dim = self.ac_dim

        p_cond = self.algo_config.future_action_condition.p_cond
        Tf = self.algo_config.future_action_condition.horizon

        if torch.rand(1).item() < p_cond:
            # Create null token of same shape as action_cond
            null_token = self._get_null_token_actions(B, Tf, action_dim)
            return null_token
        else:
            action_cond = actions[:, To-1:To-1+Tf, :] # (B, Tf, D)
            action_cond = action_cond.flatten(start_dim=1) # (B, Tf*D)

            if "binary_mask" in self.algo_config.future_action_condition.keys() and self.algo_config.future_action_condition.binary_mask:
                binary_mask = COND_TOKEN * torch.ones((B, 1), device=self.device)
                action_cond = torch.cat([action_cond, binary_mask], dim=-1)

            return action_cond # (B, Tf*D)

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
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
                'goal': batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)

            # TODO: Add CFG-related code here
            # 1. Drop the action observation condition seldomly with p_cond, nullify the action observation condition
            # 2. Add cumulative binary mask here (B, Tf*D)
            if "future_action_condition" in self.algo_config.keys() and self.algo_config.future_action_condition.enabled:
                action_cond = self._get_action_condition(batch["actions"]) # (B, Tf*D)
                obs_cond = torch.cat([obs_cond, action_cond], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets.parameters())
                
                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    # @time_profile
    def get_action(self, obs_dict, goal_dict=None, **kwargs):
        return self.get_action_sequential(obs_dict, goal_dict, **kwargs)

    def get_action_sequence(self, obs_dict, goal_dict=None, **kwargs):
        return self._get_action_trajectory(obs_dict, goal_dict, **kwargs)

    def get_action_aborted(self, obs_dict, goal_dict=None, parallel_inference=False, **kwargs):
        # import pdb; pdb.set_trace()
        if parallel_inference:
            return self.get_action_receding_horizon(obs_dict, goal_dict, **kwargs)
        else:
            return self.get_action_sequential(obs_dict, goal_dict, **kwargs)

    def get_action_receding_horizon(self, obs_dict, goal_dict=None, **kwargs):
        # with inference thread and temporal ensemble
        with self.thread_lock:
            # update shared input buffer
            self.obs_dict = obs_dict
            self.goal_dict = goal_dict
            self.kwargs = kwargs

            if not self.inference_thread_initialized:
                self.inference_thread = threading.Thread(target=self.run_inference_thread)
                self.inference_thread.start()
                self.inference_thread_initialized = True

        # import pdb; pdb.set_trace()
        while True:
            with self.thread_lock:
                # Waiting for the first action
                if len(self.action_queue) > 0:
                    return self.get_temporal_ensemble_action(), TensorUtils.clone(self.data_logged)
            time.sleep(0.001)

    def get_temporal_ensemble_action(self, m=0.5):
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
            cprint(f"Warning: action queue len {len(self.action_queue)} is less than queue call count {self.queue_call_count}, temporal ensemble  disabled, using the latest action sequence", "red")
            num_action_seqs = 1 # only use the latest action sequence in this special case, can be improved

        # reverse order let old action has more weight, smoother transform. forward order let new action has more weight, more responsive
        # Following ACT, we are using reverse order
        forward_order = torch.arange(num_action_seqs, device=self.device) # [large -> small] -> [a_t, a_t-1, a_t-2, ...]
        reverse_order = forward_order.flip(0) # [small -> large] -> [a_t, a_t-1, a_t-2, ...] (ACT in action weight order)
        weights = torch.exp(-m * reverse_order)
        weights = weights / weights.sum()

        # for whatever number of action sequences in the queue, do the weighted sum
        # action_sequence: [num_action_seqs, action_dim]
        num_action, action_dim = self.action_queue[0].shape
        action = torch.zeros(action_dim, device=self.device) # [action Dim]
        for i in range(num_action_seqs):
            # for action sequence generated at time t-i, we take the i-th action (i.e. align with the current time step t)
            # queue_call_count is the offset for t, used when the queue is not updated in time.
            action_id = i + self.queue_call_count
            if action_id >= num_action:
                action_id = num_action - 1
                cprint(f"Warning: running out of action sequence, action queue len {len(self.action_queue)} queue call count {self.queue_call_count}", "red")
            action_i = self.action_queue[-i-1][action_id,:]
            action += weights[i] * action_i

        self.queue_call_count += 1

        # # TODO: delete this later, for testing that disable angle axis ensembling
        # if self.temp_var == None:
        #     self.temp_var = action_i[3:6]
        # action[3:6] = self.temp_var
        return action.unsqueeze(0)

    def run_inference_thread(self):
        while self._running:
            try:
                with self.thread_lock:
                    # action sequence: [B, Dim, Ta]
                    obs_dict_copy = TensorUtils.clone(self.obs_dict)
                    goal_dict_copy = TensorUtils.clone(self.goal_dict)
                    kwargs_copy = TensorUtils.clone(self.kwargs)

                    # # TODO: for testing the predicted action as observation: test key is right
                    # if len(self.action_queue) > 0:
                    #     action_seq_copy = self.action_queue[-1].detach().clone()
                    #     print(f"action seq copy: {action_seq_copy}")
                    #     obs_dict_copy["actions"] = action_seq_copy[:self.To, :].unsqueeze(0) # first To actions, which is being executed during current inference
                    # else:
                    #     cprint("Warning: action queue is empty, can't use action as observation", "yellow")

                # copy data and run inference outside thread lock
                action_sequence = self._get_action_trajectory(obs_dict_copy, goal_dict_copy, **kwargs_copy) # 50ms to run

                # delete images to save memory, keep low dim
                if 'agentview_image' in obs_dict_copy:
                    del obs_dict_copy['agentview_image']
                if 'robot0_eye_in_hand_image' in obs_dict_copy:
                    del obs_dict_copy['robot0_eye_in_hand_image']

                # add latest action to the queue
                with self.thread_lock:
                    # prefill the queue with first action sequence
                    if len(self.action_queue) ==0:
                        if len(action_sequence.shape) == 3:
                            self.action_queue.extend([action_sequence[-1].clone() for _ in range(self.action_queue.maxlen)])
                        else:
                            self.action_queue.extend([action_sequence.clone() for _ in range(self.action_queue.maxlen)])
                    else:
                        if len(action_sequence.shape) == 3:
                            self.action_queue.append(action_sequence[-1])
                        else:
                            self.action_queue.append(action_sequence)
                    self.queue_call_count = 0 # reset queue call count

                    self.data_logged = {'t': time.time(), 'obs': TensorUtils.to_numpy(obs_dict_copy),
                                        'goal': TensorUtils.to_numpy(goal_dict_copy), 'action': TensorUtils.to_numpy(action_sequence)}

                # run DP inference at a constant rate
                self._rate.sleep()

            except Exception as e:
                print(f"Error in DP inference thread: {e}")
                break

    def get_action_sequential(self, obs_dict, goal_dict=None, **kwargs):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)

        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            # Check if any observation has an extra batch dimension of 1
            # TODO: this is a hack
            has_extra_batch_dim = any(obs_dict[k].ndim == len(self.obs_shapes[k]) + 3 and obs_dict[k].shape[0] == 1
                                    for k in self.obs_shapes)

            # Remove extra batch dimension if present
            if has_extra_batch_dim:
                print("RolloutPolicy has added an extra batch dimension of 1, squeezing it")
                obs_dict = TensorUtils.squeeze(obs_dict, dim=0)


            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, **kwargs)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])

        if kwargs.get("return_action_sequence", False):
            Ta = self.algo_config.horizon.action_horizon
            self.action_queue = deque(maxlen=Ta)
            return action_sequence
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        # TODO: assuming vel=0 for the last actions. Fix this later
        next_action = self.action_queue[0] if len(self.action_queue) != 0 else action

        # [1,Da]
        action = action.unsqueeze(0)
        next_action = next_action.unsqueeze(0)
        return action, next_action

    # @profile
    def _get_action_trajectory(self, obs_dict, goal_dict=None, **kwargs):
        """
        Get action trajectory

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal
            kwargs:
                guide_config (config.Config): guide config
                guide_actions (torch.tensor): guide actions [Tg, Da]

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            self.ema.copy_to(parameters=self._shadow_nets.parameters())
            nets = self._shadow_nets
        
        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }
        # If we use framestack, make sure the input is structured correctly
        # TODO: this needs to be fixed in upstream
        if self.global_config.train.frame_stack > 1:
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k]), f"This obs is wrong {k} . This {inputs['obs'][k].ndim}"
            obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]
        else:
            # no framestack, therefore just get the features from encoder
            obs_features = self.nets['policy']['obs_encoder'](**inputs) # [B, T, D]

        B = obs_features.shape[0]
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # TODO: change this from kwargs
        if kwargs.get('sample_bezier', False):
            # Sample noisy action from bezier-smoothed gaussian
            noisy_action = random_sample_from_bezier(B, N=Tp, D=action_dim, device=self.device)
        else:
        # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        # Set guidance mode for inpainting/action consistency
        guide_config  = kwargs.get("guide_config", None)
        guide_actions = kwargs.get("guide_actions", None)
        use_guiding   = (guide_config is not None) and (guide_config.enabled) and (guide_actions is not None)

        # print(f"^^^^^^^^^^^^^^^^^^{guide_config} ^^^^^^^^^^ {guide_actions} ^^^^^^ {use_guiding}")

        # If model is trained with future action condition, elongate the future action condition
        if "future_action_condition" in self.algo_config.keys() and self.algo_config.future_action_condition.enabled:
            # unconditional action condition
            Tf = self.algo_config.future_action_condition.horizon
            actions_cond_null = self._get_null_token_actions(B, Tf, action_dim)
            obs_cond  = torch.cat([obs_cond, actions_cond_null], dim=-1)

        # default setting for ddim eta
        if self.algo_config.ddim.enabled:
            ddim_eta = 0
        elif self.algo_config.ddpm.enabled:
            ddim_eta = 1
        if guide_config is not None:
            ddim_eta = guide_config.ddim_eta

        # Debugging purpose - log diffusion steps
        self.step_logs = {}

        for k in self.noise_scheduler.timesteps:
            # predict noise
            if not use_guiding or (k > guide_config.timestep_start) or (k < guide_config.timestep_end):
                # No guiding, diffuse normally
                naction = self.step_normal(nets,
                                           naction=naction,
                                           step=k,
                                           obs_cond=obs_cond,
                                           eta=ddim_eta)

            elif guide_config.inpainting.enabled:
                # initial_actions = kwargs['initial_actions'] # actions to inpaint at the start of the trajectory
                naction = self.step_inpainting(nets, 
                                               naction=naction,
                                               step=k,
                                               obs_cond=obs_cond,
                                               eta = ddim_eta,
                                               guide_actions = guide_actions,
                                               **guide_config.inpainting)

            elif guide_config.consistency_loss.enabled:
                naction = self.step_consistency_guidance(nets,
                                                        naction=naction,
                                                        step=k,
                                                        obs_cond=obs_cond,
                                                        eta = ddim_eta,
                                                        guide_actions = guide_actions,
                                                        **guide_config.consistency_loss)

            elif guide_config.cfg.enabled:
                naction = self.step_classifier_free_guidance(nets,
                                                             naction=naction,
                                                             step=k,
                                                             obs_cond=obs_cond,
                                                             eta = ddim_eta,
                                                             guide_actions=guide_actions,
                                                             **guide_config.cfg)

            elif guide_config.classifier.enabled:
                assert self.classifier is not None,\
                    'Classifier must be set in Diffusion Policy to use classifier guidance'
                naction = self.step_classifier_guidance(nets,
                                                        naction=naction,
                                                        step=k,
                                                        obs_cond=obs_cond,
                                                        eta = ddim_eta,
                                                        guide_actions = guide_actions
                                                        **guide_config.classifier)

            else:
                raise NotImplementedError


        # process action using Ta
        start = To - 1
        if kwargs.get("return_Tp", False): # TODO: not sure if correct, changing back for now
            end = Tp
        else:
            end = start + Ta
        action = naction[:,start:end]
        return action

    def step_normal(self, nets, naction, step, obs_cond, **kwargs):
        """
        Reverse and forward diffusion step normally.

        Args:
            nets: (torch.nn.Module) noise prediction model \epsilon(x_{k}, k, obs_cond)
            naction: (torch.tensor) x_{step}
            step: (int) x_{-1} is final, N if full gaussian noise
            obs_cond: (torch.tensor) global conditioning

        Returns:
            naction: (torch.tensor) x_{step-1} or x_{step-k} (for DDIM)
        """
        # Parse kwargs
        eta = kwargs["eta"]

        # Predict noise
        noise_pred = nets['policy']['noise_pred_net'](
            sample=naction,
            timestep=step,
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise) and add noise to generate x_{step-1}
        naction = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=step,
            sample=naction,
            eta=eta
        ).prev_sample

        return naction

    def step_classifier_free_guidance(self, nets, naction, step, obs_cond, **kwargs):
        """
        Reverse and forward diffusion step normally.

        Args:
            nets: (torch.nn.Module) noise prediction model \epsilon(x_{k}, k, obs_cond)
            naction: (torch.tensor) x_{step}
            step: (int) x_{-1} is final, N if full gaussian noise
            obs_cond: (torch.tensor) global conditioning
            kwargs:
                guide_actions (torch.tensor) shape of (Tf, Da)
                cfg config (dict)

        Returns:
            naction: (torch.tensor) x_{step-1} or x_{step-k} (for DDIM)

        Reference:
            1. CFG https://arxiv.org/pdf/2207.12598
        """
        Tf = self.algo_config.future_action_condition.horizon
        # Parse kwargs
        eta = kwargs["eta"]
        guide_actions = kwargs["guide_actions"]
        weight = kwargs["weight"]

        # We assume special form of guiding actions
        assert guide_actions.abs().max() <= 1, "Guide actions should be normalized"
        assert guide_actions.ndim == 2 and guide_actions.shape[-2] == Tf, "Guide actions should be of the same length as the future action condition horizon"
        assert weight >=0, "Weight should be non-negative"

        # Get the obs_cond with the guide_actions
        B = obs_cond.shape[0]
        guide_actions = guide_actions.repeat(B, 1, 1).flatten(start_dim=1)
        obs_cond_with_future_actions = obs_cond.clone()
        if "binary_mask" in self.algo_config.future_action_condition.keys() and self.algo_config.future_action_condition.binary_mask:
            obs_cond_with_future_actions[:, -Tf*self.ac_dim-1:-1] = guide_actions
            obs_cond_with_future_actions[:, -1] = COND_TOKEN * torch.ones((B,), device=self.device)
        else:
            obs_cond_with_future_actions[:, -Tf*self.ac_dim:] = guide_actions

        # Predict unconditional noise
        noise_pred_unconditional = nets['policy']['noise_pred_net'](
            sample=naction,
            timestep=step,
            global_cond=obs_cond
        )

        # Predict conditional noise
        noise_pred_conditional = nets['policy']['noise_pred_net'](
            sample=naction,
            timestep=step,
            global_cond=obs_cond_with_future_actions
        )

        # Classifier-free guidance: Restructure Eq (6) https://arxiv.org/pdf/2207.12598
        noise_pred = noise_pred_unconditional + (1 + weight) * (noise_pred_conditional - noise_pred_unconditional) # (B, Tp, Da)

        # inverse diffusion step (remove noise) and add noise to generate x_{step-1}
        naction = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=step,
            sample=naction,
            eta=eta
        ).prev_sample

        return naction

    def step_inpainting(self, nets, naction, step, obs_cond, **kwargs):
        """
        Reverse and forward Diffusion Process for one timestep with inpainting using repainting.


        Args:
            nets (_type_): noise prediction net
            naction (_type_): action to denoise
            step (_type_): _description_
            obs_cond (_type_): _description_
            initial_cond (_type_): _description_
            n_resample: how many resampling steps to apply (currently unused)
            step_resample: when to start applying resampling
            guide_actions (torch.tensor) hape of (Tg, Da)


        Returns:
            naction: (torch.tensor) x_{step-1} or x_{step-k} (for DDIM)
        """
        with torch.no_grad():
            # Parse variables from kwargs
            initial_actions = kwargs['guide_actions']
            n_resample      = kwargs['n_resample']
            eta             = kwargs['eta']


            # Parse static variables
            To = self.algo_config.horizon.observation_horizon
            Tp = self.algo_config.horizon.prediction_horizon
            k = initial_actions.shape[0]
            B = naction.shape[0]

            init_actions = TensorUtils.to_device(initial_actions, self.device).unsqueeze(0).repeat(B, 1, 1) #[B, k, action_dim] # k=how many to inpaint

            for idx in range(n_resample+1):
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=step,
                    global_cond=obs_cond
                )

                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=step,
                    sample=naction,
                    eta=eta
                ).prev_sample  # x_{step-1}

                prev_step = self._prev_step(step)

            inpaint_mask = torch.ones(init_actions.shape, device=self.device)
            if kwargs.get("blend_inpaint", False):
                blend_dim = kwargs.get("blend_dim", 3)
                blend_mask = (torch.linspace(1, 0, steps=init_actions.shape[1], device=self.device).
                                unsqueeze(0).repeat(naction.shape[0], 1))  # (B, k)
                blend_curve = kwargs.get("blend_curve", "linear")

                if blend_curve == "parabolic":
                    # t^2 * (3-2t)
                    blend_mask = torch.multiply(torch.pow(blend_mask, 2), 3-(torch.multiply(blend_mask, 2)))
                elif blend_curve == "convex":
                    # 2t - t^2
                    blend_mask = torch.multiply(blend_mask, 2) - torch.pow(blend_mask, 2)

                blend_mask = blend_mask.unsqueeze(-1).repeat(1, 1, blend_dim)  # (B, k, action_dim_to_blend)
                # import pdb
                # pdb.set_trace()
                inpaint_mask[..., :blend_dim] = blend_mask

            if prev_step >= 0:
                naction_init = self._forward_noise(init_actions, -1, prev_step)  # (B, k, action_dim)
                # naction_mask  = torch.ones(naction.shape)


                # naction_mask[:, To-1:To-1+k, :] = inpaint_mask
                naction[:,
                To - 1:To - 1 + k] = inpaint_mask * naction_init + (1 - inpaint_mask)*naction[:,
                To - 1:To - 1 + k]  # replace the parts of the action to denoise with the known actions after noising
            else:
                # pass
                if kwargs.get("blend_inpaint_last", False):
                    naction[:, To - 1:To - 1 + k] = inpaint_mask * init_actions + (1 - inpaint_mask) *naction[:,
                To - 1:To - 1 + k]
                else:
                    naction[:, To - 1:To - 1 + k] = init_actions

            # Adding blurring (i.e. spline the inpainted and non inpainted actions)
            # TODO: this does not work yet
            if kwargs.get("blur_inpaint", False):
                # import pdb
                # pdb.set_trace()
                blur_dim = kwargs.get("blur_dim", 3) # which of the parts of the action to blur
                blur_length = kwargs.get("blur_length", initial_actions.shape[1] + 2) # number of actions to spline
                blur_noise(naction, blur_dim, blur_length)
                if prev_step >= 0:
                    naction_init = self._forward_noise(init_actions, -1, prev_step)  # (B, k, action_dim)
                    naction[:, To-1:To-1+k] = naction_init # replace the parts of the action to denoise with the known actions after noising
                else:
                    naction[:, To-1:To-1+k] = init_actions

                # Adding blurring (i.e. spline the inpainted and non inpainted actions)
                if kwargs.get("blur_inpaint", False):
                    blur_dim = kwargs.get("blur_dim", 3) # which of the parts of the action to blur
                    blur_length = kwargs.get("blur_length", initial_actions.shape[1] + 2) # number of actions to spline
                    blur_noise(naction, blur_dim, blur_length)


                # Resample, i.e. reconstruct x_t by adding noise, based on RePaint paper: https://arxiv.org/pdf/2201.09865
                if (idx < n_resample):
                    naction = self._forward_noise(naction, prev_step, step)

            return naction

    def step_classifier_guidance(self, nets, naction, step, obs_cond, classifier_kwargs):
        """
        Reverse and forward diffusion process using classifier guidance based off this paper : https://arxiv.org/pdf/2105.05233
        Returns x_{t-1} after applying classifier guidance.

        Note: the convention in this function is different from the rest of the codebase, to more closely align with the
        equations in https://arxiv.org/pdf/2105.05233

        Args:
            nets:
            naction:
            step:
            obs_cond:
            classifier_kwargs:

        Returns:

        """
        weight = classifier_kwargs["weight"]
        if self.algo_config.ddpm.enabled is True:
            raise Exception("This method of classifier guidance currently only works with DDIM")

        eps = nets['policy']['noise_pred_net'](
            sample=naction,
            timestep=step,
            global_cond=obs_cond
        )

        # setup for equation (12)
        alpha_bar_t = self.noise_scheduler.alphas_cumprod[step]

        # classifier guidance must be derivative of log probability w.r.t x_t (naction)
        classifier_guidance = self.classifier.get_guidance(naction, weight, classifier_kwargs)

        # equation (12)
        eps_hat = eps - torch.sqrt(1 - alpha_bar_t)*classifier_guidance

        naction = self.noise_scheduler.step(
            model_output=eps_hat,
            timestep=step,
            sample=naction,
            eta = self.eta
        ).prev_sample  # x_{step-1}

        raise NotImplementedError

        return naction
    
    def _prev_step(self, step):
        """
        Get previous step according to the scheduler
        
        Args
            step (int): the timestep
        
        Return
            prev_step (int): the previous timestep
        """
        if self.algo_config.ddpm.enabled:
            # DDPM
            prev_step = self.noise_scheduler.previous_timestep(step)  # step - 1
        elif self.algo_config.ddim.enabled:
            # DDIM
            prev_step = step - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps  # step - t 
        else:
            raise NotImplementedError
        
        return prev_step
    
    def _forward_noise(self, naction, from_, to_):
        """
        Forward diffusion: x_{step} ~ p(x_{step}|x_{prev_step}) following noise schedule

        Args:
            naction (torch.tensor) (B, Tp, D)
            prev_step (int)
            step (int)
        """
        assert to_ > from_, "The prev_step should be smaller than the step"
        assert to_ >= 0 , "The step should be equal or greater than 0"

        (B, Tp, Da) = naction.shape
        prev_step = from_
        step      = to_

        if self.algo_config.ddpm.enabled: # resampling for DDPM
            noise_for_resample = torch.randn(B, Tp, self.ac_dim, device=self.device)
            beta               = self.noise_scheduler.betas[step]
            naction            = ((1 - beta) ** 0.5) * naction + (beta ** 0.5) * noise_for_resample
        
        elif self.algo_config.ddim.enabled: # resampling for DDIM
            noise_for_resample  = torch.randn(B, Tp, self.ac_dim, device=self.device)
            alphas_cumprod      = self.noise_scheduler.alphas_cumprod[step]
            prev_alphas_cumprod = self.noise_scheduler.alphas_cumprod[prev_step] if prev_step >= 0 else self.noise_scheduler.final_alpha_cumprod
            sig_scale           = (alphas_cumprod / prev_alphas_cumprod) ** 0.5
            noise_scale         = (1 - alphas_cumprod) ** 0.5 - sig_scale * (1 - prev_alphas_cumprod) ** 0.5
            naction             = sig_scale * naction + noise_scale * noise_for_resample

        else:
            raise NotImplementedError

        return naction
    
    def get_consistency_loss_fn(self, loss_type, **kwargs):
        """
        Load the loss function for consistency loss

        Args:
            loss_type (str): type of loss function

        Returns:
            loss_fn (torch.nn.Module): loss function
        """
        if loss_type == "mse":
            return nn.MSELoss(reduction='none')
        elif loss_type == "weighted_mse":
            return ExponentialWeightedMSELoss(reduction='none')
        elif loss_type == "sparc":
            return SparcLoss(reduction='none', **kwargs)
        else:
            raise NotImplementedError

    def step_consistency_guidance(self, nets, naction, step, obs_cond, **kwargs):
        """
        x_{prev_t} ~ p(x_{prev_t} | x_{t}, c)

        Args:
            nets (torch.nn.module) epsilon(x_t, t)
            naction (torch.tensor) x_t shape of (B, Tp, DA)
            step (int) t
            obs_cond (torch.tensor) c
            consistency_kwargs
                guide_actions (torch.tensor) y shape of (Tg, Da)

        NOTE
            Reference:
                Recurrence: FreeDOM (https://arxiv.org/abs/2303.09833), Section 4.2
                Monte Carlo Smoothed Estimation: LGD (https://proceedings.mlr.press/v202/song23k.html)
                Mean Steering: MPGD (https://arxiv.org/abs//2311.16424)
        """
        eta    = kwargs['eta']
        y      = kwargs['guide_actions'].float()
        # Hyperparameter
        # How many times we want to resample (time-travel)
        N_recur    = kwargs['n_resample']
        # How many number of samples would be used for monte-carlo estimation of conditional score
        N_mc       = kwargs['N_sample_monte_carlo']
        # How large would be the variance of p(x0|xt)  (0 if point estimate, 1 if LGD)
        gamma_mc   = kwargs['std_monte_carlo']
        # Guidance weight
        weight     = kwargs['weight']
        # TODO: set SPARC-related parameters here including action normalization
        # TODO: save the data required for SPARC analysis (clean_actions, noisy_actions, loss_fn), dump at every step for guidance
        loss_fn    = self.get_consistency_loss_fn(**kwargs)

        (B, Tp, D) = naction.shape
        To         = self.algo_config.horizon.observation_horizon
        
        for r in range(N_recur+1):
            with torch.enable_grad():
                # xt
                naction_g = naction.clone().detach().requires_grad_(True)

                # epsilon(xt, t)
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction_g,
                    timestep=step,
                    global_cond=obs_cond
                )

                out_prev = self.noise_scheduler.step(
                    model_output = noise_pred,
                    timestep     = step,
                    sample       = naction_g,
                    eta          = eta
                )
                
                # xtm1 (unconditional sample with variance already added)
                naction_prev = out_prev.prev_sample

                # x0|t
                x_0_hat      = out_prev.pred_original_sample

                # loss calculation L(x0|t, gc)
                start      = To-1
                # TODO: make this `end` to be decided by the loss function
                # end        = To-1 + y.shape[0]

                """
                Monte-carlo for smoothed estimation
                p(y|xt) = 1/Nmc * \sum_{i=1}^{Nmc} (exp(-loss(x0|t + \epsilon_i))
                log_xt_p(y|xt) = 
                """
                noise_mc_std = gamma_mc * ((1 - self.noise_scheduler.alphas_cumprod[step]) ** (0.5))

                noise_mc    = noise_mc_std * torch.randn(N_mc, Tp, D, device=self.device)
                x0_samples  = x_0_hat + noise_mc       # (Nmc, Tp, A)
                x0_to_guide = x0_samples[:, start:] # (Nmc, Ng, D)

                # TODO
                # from robomimic.dev.smoothness_metrics.compute_smoothness_metrics import compute_sparc_torch
                # sparc, x0_samples = compute_sparc_torch(x0_samples, dt=0.05)

                # -loss(x0|t + \epsilon_i)
                loss           = loss_fn(x0_to_guide, y.unsqueeze(0).repeat(N_mc, 1, 1)) # (Nmc, Ng, D)
                loss_per_batch = loss.mean(dim=(1, 2), keepdim=True).squeeze() # (Nmc, 1, 1)

                # Estimate conditional score function: log ( 1 / N_mc * \sum exp(-loss(x0|t + \epsilon_i)))
                avg_logprobs   = torch.logsumexp(-loss_per_batch, dim=0) - math.log(N_mc) # (1, 1, 1)

                grad_xt        = torch.autograd.grad(avg_logprobs, naction_g)[0]

            # x_{prev_t} = sample(x_t, x_{0|t}, t) - rho * grad_xt_L(x0|t, gc)
            naction_prev = naction_prev + weight * grad_xt

            self.step_logs[step.cpu().item()] = {
                "naction"  : naction.detach().cpu().numpy(),
                "loss_log" : loss_fn.log,
                "grad_xt"  : grad_xt.detach().cpu().numpy(),
                "output"   : naction_prev.detach().cpu().numpy()
            }

            # renoise back to x_{t}
            if r < N_recur:
                prev_step   = self._prev_step(step)
                naction     = self._forward_noise(naction_prev, from_ = prev_step, to_ = step)

        return naction_prev

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.load_state_dict(model_dict["ema"])
        

# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
