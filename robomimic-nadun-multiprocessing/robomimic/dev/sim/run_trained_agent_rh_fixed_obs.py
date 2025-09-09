"""
This script provides many useful functions for evaluating a trained agent on the fixed observation sequences.

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
from torch.utils.data import DataLoader

import robomimic.dev.hparam.utils.dataset_utils as DatasetUtils
import robomimic.dev.sim.run_trained_agent_rh_hparam as RunTrainedAgentRH

def maybe_load_guide_config_from_json(config_path:str, **kwargs):
    """
    Load the guide config from the json file and set the guide config
    
    Args:
        config_path (str): Path to the config json file
        **kwargs: Additional arguments that may contain action normalization stats
        
    Returns:
        guide_config: The loaded and configured guide config, or None if config_path is None
    """
    if config_path is not None:
        guide_config = config_factory("guided_diffusion_policy", dic=None).guide
        with open(config_path, "rb") as f:  # Fixed: Use config_path instead of args.config
            guide_config_dict = json.load(f)["guiding"]
        with guide_config.values_unlocked():
            guide_config.update(guide_config_dict)
        
        # Extra setting for consistency loss
        if guide_config.enabled and \
            guide_config.consistency_loss.enabled and \
                guide_config.consistency_loss.loss_type == "sparc":
                action_normalization_stats = kwargs.get("action_normalization_stats", None)  # Fixed: Use consistent naming
                if action_normalization_stats is not None:
                    guide_config.consistency_loss.unlock_keys()
                    action_dict = action_normalization_stats[list(action_normalization_stats.keys())[0]]
                    guide_config.consistency_loss.action_scale = action_dict["scale"]
                    guide_config.consistency_loss.action_offset = action_dict["offset"]
                    guide_config.consistency_loss.lock_keys()
    else:
        guide_config = None

    return guide_config
    

def process_batch_for_policy(policy:RolloutPolicy, batch:dict) -> dict:
    """
    Process the batch for the policy
    
    Args:
        policy (RolloutPolicy): Policy to process batch for
        batch (dict): Batch of data to process
        
    Returns:
        dict: Processed batch with normalized observations and squeezed dimensions
    """
    batch = policy.policy.process_batch_for_training(batch)
    batch = policy.policy.postprocess_batch_for_training(batch, obs_normalization_stats=None)
    batch["obs"] = {key: batch["obs"][key].squeeze(0) for key in batch["obs"].keys()}
    return batch

def unnormalize_actions(policy:RolloutPolicy, actions:torch.Tensor) -> torch.Tensor:
    """
    Unnormalize actions using policy's normalization stats
    
    Args:
        policy (RolloutPolicy): Policy containing normalization stats
        actions (torch.Tensor): Actions to unnormalize, shape (B, T, DA)
        
    Returns:
        torch.Tensor: Unnormalized actions
    """
    ac_norm_stats = policy.action_normalization_stats
    device = policy.policy.device
    actions = TensorUtils.to_tensor(actions)
    actions = TensorUtils.to_device(actions, device)

    if ac_norm_stats is not None:
        assert len(ac_norm_stats.keys()) == 1, "Only one action key is supported for now"

        ac_key = list(ac_norm_stats.keys())[0]
        ac_norm_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(ac_norm_stats), device))
        actions_dict = {ac_key: actions}
        actions_dict = ObsUtils.unnormalize_dict(actions_dict, normalization_stats=ac_norm_stats)
        actions = actions_dict[ac_key]

    return actions

def eval_policy_on_demo_obs(policy:RolloutPolicy, 
                            demo_loader:DataLoader, 
                            guide_config:dict,
                            video_writer,
                            **kwargs) -> tuple:
    """
    Evaluate the policy on demonstration observations
    
    Args:
        policy (RolloutPolicy): Policy to evaluate
        demo_loader (DataLoader): DataLoader containing demonstration data
        **kwargs: Additional arguments including inference delay and execution steps
        
    Returns:
        tuple: (stats dict, trajectory dict) containing evaluation metrics and trajectory data

    NOTE:
    The policy is trained on the following data where t+To-1 is current time
        1. Observation (t:t+To)
        2. Action (t:t+Tp)
    """
    To  = policy.policy.algo_config.horizon.observation_horizon
    Tp  = policy.policy.algo_config.horizon.prediction_horizon
    Td  = kwargs.get('inf_delay', 0)
    Tne = kwargs.get('execute_n_actions', 0)
    Tf  = policy.policy.algo_config.future_action_condition.horizon
    Ta  = Td + Tne

    N_pred = kwargs.get('N_predictions', 1)

    policy.start_episode()
    demo_loader_iter = iter(demo_loader)

    traj = dict(executed_actions=[], multi_preds = [], preds=[], demo_preds=[], states=[], obs=[], diffusion_logs=[])
    
    for step_idx in range(len(demo_loader_iter)):
        if step_idx % Ta == 0:
            batch = next(demo_loader_iter)
            batch = process_batch_for_policy(policy, batch)

            demo_obs = batch["obs"]  # o[t:t+To]
            demo_actions = batch["actions"].squeeze(0)  # a[t:t+Tp], normalized
            demo_prediction = unnormalize_actions(policy, demo_actions)  # a[t:t+Tp], unnormalized

            demo_guide_actions = demo_actions[To-1:To-1+Tf] # a[t+To-1:t+To-1+Tf], normalized

            demo_obs_batched = TensorUtils.unsqueeze_expand_at(demo_obs, size=N_pred, dim=0)

            predictions, multi_predictions = policy(ob=demo_obs_batched,
                                guide_config=guide_config, 
                                guide_actions=demo_guide_actions,
                                return_all_pred=True,
                                **kwargs)  # a[t+To-1:t+To-1+Ta], unnormalized

            # save the trajectory data
            traj['preds'].append(TensorUtils.to_numpy(predictions))
            traj['multi_preds'].append(TensorUtils.to_numpy(multi_predictions))
            traj['demo_preds'].append(TensorUtils.to_numpy(demo_prediction))
            traj['executed_actions'].append(TensorUtils.to_numpy(predictions[:Ta]))
            traj = RunTrainedAgentRH.save_obs(traj, TensorUtils.to_numpy(demo_obs))

    stats = dict(Return=0, Horizon=len(demo_loader_iter), Success_Rate=1.0)
    return stats, traj

def main(args:argparse.Namespace, **kwargs):
    """
    Main evaluation function
    
    Args:
        args (argparse.Namespace): Command line arguments
        **kwargs: Additional arguments passed to evaluation functions
    """
    write_dataset = args.log_path is not None
    write_video = args.video_path is not None

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    policy, validation_loader = DatasetUtils.get_policy_and_validation_dataloader(
        agent_path=args.agent, 
        dataset_path=args.trainset
    )
    
    demo_id_to_demo_loaders = DatasetUtils.split_dataloader_by_demo(validation_loader)

    guide_config = maybe_load_guide_config_from_json(args.guide_config, **kwargs)

    rollout_stats, rollout_trajs = [], {}

    count = 0
    for demo_id, demo_loader in tqdm(demo_id_to_demo_loaders.items()):
        print(f"Evaluating {demo_id}")
        if count >= args.n_rollouts:
            break

        stats, traj = eval_policy_on_demo_obs(
            policy=policy,
            demo_loader=demo_loader,
            video_writer=video_writer,
            guide_config=guide_config,
            **kwargs
        )
        traj['kwargs'] = kwargs
        rollout_stats.append(stats)
        rollout_trajs[demo_id] = traj

        count += 1
    
    if write_dataset:
        rollout_results = {
            "stats": rollout_stats,
            "rollouts": rollout_trajs,
        }
        with open(args.log_path, 'wb') as f:
            pickle.dump(rollout_results, f)
        print(f"Wrote dataset trajectories and stats to {args.log_path}")

    if write_video:
        video_writer.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        # default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/held_out/square/square_cfg_bm_false_model_epoch_1000.pth",
        # default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/cfg_square/square_image_action_horizon_16_additional_bl/cfg_binary_heldout/models/model_epoch_900.pth",
        default="/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/cfg_square/square_image_action_horizon_16_uncond/20250124091457/models/model_epoch_500.pth",
        help="path to saved checkpoint pth file",
    )

    parser.add_argument(
        "--trainset",
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/datasets/square/ph/all_obs_new.hdf5",
        help="path to train set hdf5 file"
    )

    parser.add_argument(
        '--guide_config',
        type=str,
        # default="/home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/square_compare/baseline_config_square_cfg.json",
        default="/home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_weight_0.3.json",
        help="Json file that configures the guiding algorithm for the diffusion policy. This sets only guiding parameters, not the model architecture."
    )
    
    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=5,
        help="evaluate of first `n_rollouts` demonstrations",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/videos/fixed_obs",
        help="(Optional) where to save videos of the different evals"
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="path to save the video"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/wjung85/Repo/projects/FastIL/logs/multimodality",
        help="path to save the dataset"
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        '--N_eval',
        type=int,
        default=1,
        help="number of evaluations"
    )

    parser.add_argument(
        '--N_predictions',
        type=int,
        default=64,
        help="number of predictions"
    )

    # parser.add_argument(
    #     '--multi_action_prediction',
    #     type=bool,
    #     default=False,
    #     help="whether to save the multiple action sequence"
    # )

    args = parser.parse_args()
    
    # Ta = Td + Tne (closing loop frequency should be equivalent for fair comparison)
    Td  = 4  # steps of delay for inference
    Tne = 8 # steps of executing the actions before inference

    kwargs = { # kwargs for the model
              "return_action_sequence": True,
              "noisy_eval" : False,
              "execute_n_actions": Tne, 
              "inf_delay": Td,
              'return_obs': True, 
              'save_high_dim_obs': False,
              "fast_control_freq": 20,
              "N_predictions": args.N_predictions,
              }

    with open(args.guide_config, "r") as f:
        guide_config = json.load(f)
        exp_name = guide_config["experiment"]["name"]

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    agent_name = os.path.basename(args.agent).split(".")[0]
    
    for idx in range(args.N_eval):
        filename_id =f"agent_{agent_name}_guide_{exp_name}_iter_{iter}_{datetime.datetime.now()}"
        args.video_path = os.path.abspath(os.path.join(args.video_dir, f"{filename_id}.mp4"))
        args.log_path = os.path.abspath(os.path.join(args.log_dir, f"{filename_id}.pkl"))

        main(args, **kwargs)