#!/usr/bin/env python3
"""
Script to run a trained ATM policy on the real robot.

This script:
1. Loads a trained ATM model
2. Runs the policy on the real robot in a rollout loop
3. Executes delta actions from the policy with optional target chaining

Target Chaining Support:
- Default: Uses actual robot pose at each timestep for delta action computation
- Target Chaining: Uses previous target pose (except first action in each horizon)
  
Usage:
    python run_trained_policy_atm.py use_target_chaining=true
    python run_trained_policy_atm.py use_target_chaining=false  # default
"""

import hydra
import math
from glob import glob
import pandas as pd
import torch
import time
import h5py  # <--- ADD THIS

import datetime
import torch
torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import torch.distributed as dist
import lightning
from lightning.fabric import Fabric
import gc
import pickle
import os
import cv2


# from einops import rearrange
from omegaconf import DictConfig

import sys
sys.path.insert(0, "/mnt/data2/mfm/workspace/Franka/gello_software")

import numpy as np

from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv

from atm.policy import *
from atm.utils.train_utils import setup_optimizer
# from atm.utils.process_utils import eef_target_pose2action, action2eef_target_pose, controller, pose2mat
# from robosuite.utils.transform_utils import mat2quat
# from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from collections import defaultdict
import pickle
torch.set_float32_matmul_precision('medium')

obs_key_mapping = {
    "gripper_states": "gripper_position",
    "joint_states": "joint_positions",
    "ee_pos": "eef_pos",
    "ee_ori": "eef_quat"
}

cam_dict = {}
camera_config_dict = {"agentview": {"sn" : "001039114912", "type": "Kinect"},
                    "wrist": {"sn": 14620168, "type": "Zed", 'resize': True, 'resize_resolution': (640, 576)}}

class Rate:
    def __init__(self, rate: float, name: str=None, log_warning=False):
        self.last = time.perf_counter()
        self.rate = rate
        self.dt   = 1.0 / self.rate
        self.name = name
        self.log_warning = log_warning

    def sleep(self) -> None:
        update_rate = 1.0 / (time.perf_counter() - self.last)
        # if self.name=="RL2 Robot Env":
        #     print(f"update rate is: {update_rate}")
        if update_rate < self.rate and self.log_warning:
            cprint(f"Warning: {self.name} update rate is {update_rate}Hz, lower than {self.rate}Hz", "red")
        while (self.last + 1.0 / self.rate) > time.perf_counter():
            time.sleep(0.001)
        self.last = time.perf_counter()


def axisangle2quat(vec):
    angle = np.linalg.norm(vec)

    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q

@torch.no_grad()
def rollout(env, policy, dataset_path, action_norms, horizon=None):
    
    cprint("--- RUNNING IN OPEN-LOOP MODE ---", "yellow", attrs=["bold"])
    cprint("Policy will use observations from HDF5 file.", "yellow")
    cprint("Robot will execute actions IGNORING its own state.", "yellow")
    cprint("Press ENTER to continue, CTRL+C to abort.", "yellow")
    input("...")

    policy.eval()
    env._robot.reset()
    policy.reset()
    
    done = False
    step_i = 0

    confirm_first_action = False
    action_queue = []
    input_output = [] # This will log what was commanded vs. what robot observed
    exec_horizon_step = 0

    rate = Rate(20.0, name="rollout_rate", log_warning=True)

    print(f"Loading dataset from: {dataset_path}")
    
    # --- ADDED: Flag to track if first frame is saved ---
    first_frame_saved = False
    
    # Open the dataset
    with h5py.File(dataset_path, 'r') as f:
        
        # Get data handles
        agentview_video = f['root/agentview/video']
        wrist_video = f['root/hand_in_eye/video']
        
        # Load the task embedding (it's static for the whole demo)
        task_emb_data = f['root/task_emb_bert'][:]
        # Keep as NumPy array and add batch dim
        task_emb = task_emb_data[np.newaxis, ...]

        num_steps = f['root/actions'].shape[0]
        if horizon is not None:
            num_steps = min(num_steps, horizon)

        print(f"Will run robot for {num_steps} steps based on dataset obs...")

        try:
            while not done and step_i < num_steps:

                # --- 1. Get Observation from Dataset (Replaces env.get_obs()) ---
                
                # --- MODIFIED: Removed .transpose(1, 2, 0) ---
                # Assumes data is already in HWC format [0, 1] float32
                agentview_rgb_ds = agentview_video[0, step_i].transpose(1, 2, 0) # (C,H,W) -> (H,W,C)
                wrist_rgb_ds = wrist_video[0, step_i].transpose(1, 2, 0)     # (C,H,W) -> (H,W,C)
                
                # --- ADDED START: Save First Frame ---
                if not first_frame_saved:
                    try:
                        output_image_path = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/first_frame_visualization.png"
                        
                        agent_img_bgr = cv2.cvtColor(agentview_rgb_ds, cv2.COLOR_RGB2BGR)
                        wrist_img_bgr = cv2.cvtColor(wrist_rgb_ds, cv2.COLOR_RGB2BGR)
                        
                        # Stack horizontally (agent | wrist)
                        combined_frame = np.hstack((agent_img_bgr, wrist_img_bgr))
                        
                        cv2.imwrite(output_image_path, combined_frame)
                        print(f"Saved first frame visualization to {output_image_path}")
                        first_frame_saved = True # Set flag so we don't save again
                    except Exception as e:
                        cprint(f"Warning: Failed to save first frame. Error: {e}", "red")
                # --- ADDED END: Save First Frame ---

                # (V, H, W, C)
                views = np.stack([agentview_rgb_ds, wrist_rgb_ds], axis=0) 
                # (B, V, H, W, C) - Keep as NumPy array
                rgb = views[np.newaxis, ...] 

                # Load extra states from dataset
                extra_states = {}
                for k in policy.extra_state_keys:
                    # Dataset keys match policy keys
                    data = f[f'root/extra_states/{k}'][step_i]
                    # Keep as NumPy array and add batch dim
                    extra_states[k] = data[np.newaxis, ...]

                # --- 2. Get new actions if queue is empty ---
                if len(action_queue) == 0:
                    try:
                        output_image_path = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/debug_policy_input.png"
                        
                        # rgb shape is (1, 2, H, W, C). Get the two images.
                        agent_img_rgb_float = rgb[0, 0] # Agent view
                        wrist_img_rgb_float = rgb[0, 1] # Wrist view
                        
                        agent_img_bgr = cv2.cvtColor(agent_img_rgb_float, cv2.COLOR_RGB2BGR)
                        wrist_img_bgr = cv2.cvtColor(wrist_img_rgb_float, cv2.COLOR_RGB2BGR)
                        
                        # Stack horizontally (agent | wrist)
                        combined_frame = np.hstack((agent_img_bgr, wrist_img_bgr))
                        
                        cv2.imwrite(output_image_path, combined_frame)
                        exit()
                        print(f"Saved policy input debug image to {output_image_path}")
                    except Exception as e:
                        cprint(f"Warning: Failed to save debug image. Error: {e}", "red")
                    
                    print("Exiting after saving debug image as requested.")
                    
                    actions, _tracks = policy.act(rgb, task_emb, extra_states) 
                    action_queue = list(actions)
                    exec_horizon_step = 0
                    print(f"Step {step_i}: Got {len(action_queue)} new abs actions from policy (using dataset obs)")

                
                # --- 3. Compute Action ---
                # 'actions' is a list of Tensors from the policy
                abs_action = action_queue[exec_horizon_step] 
                abs_action = (action_norms["scale"] * abs_action) + action_norms["offset"]
                abs_action = abs_action.squeeze() # Squeeze and move to CPU/numpy
                
                pos = abs_action[:3]
                rot = abs_action[3:6]
                gripper_action = abs_action[6]
                quat = axisangle2quat(rot)

                abs_action_full = np.concatenate([pos, quat, np.array([gripper_action])])
                
                print(f"Executing action {exec_horizon_step+1}/{len(action_queue)} in horizon (from dataset obs {step_i})")
                print(f"Target position: {pos}")

                if not confirm_first_action:
                    input(f"Intended action (with quat) [enter to rollout]: {abs_action_full}")
                    confirm_first_action = True

                # --- 4. Execute on Real Robot ---
                obs_from_robot, act = env.step(abs_action_full.tolist())
                    
                exec_horizon_step += 1
                if exec_horizon_step >= len(action_queue):
                    action_queue = []
                    exec_horizon_step = 0
                
                # --- 5. Save data (optional) ---
                save_data = {
                    # ... (your save_data dict)
                }
                input_output.append(save_data)
                
                step_i += 1
                rate.sleep()

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Saving recorded data...")
        
        # Save data regardless of how the loop exits
        output_file = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_open_loop_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(input_output, f)
        print(f"Open-loop execution data saved to {output_file}")

    return


def evaluate(fabric, cfg, checkpoint):
    cfg.model_cfg.load_path = checkpoint
    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)

    cfg.optimizer_cfg.params.lr = 0.
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)

    model, optimizer = fabric.setup(model, optimizer)
    if hasattr(model, "setup_ema"):
        model.setup_ema()
    # model.mark_forward_method('act')

    rollout_horizon = cfg.env_cfg.get("horizon", None)

    all_results = []

    action_norms = pickle.load(open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy/1027_atm_dp_absact_50_demos_1444_seed1/action_norm/action_normalization_stats.pkl", "rb"))["actions"]
    
    print("creating real world environment")

    # --- THIS SECTION REMAINS UNCHANGED FROM ORIGINAL ---
    if camera_config_dict is not None:
        for cam in camera_config_dict:
            # cam_config_dict is {"camera_name" : {"sn": int or str, type: "Zed", "RealSense" or "Kinect" +
            #                                                   camera-specific configs}
            cam_config = camera_config_dict[cam]
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam] = ZedCamera(cam, cam_config)
            elif cam_config["type"] == "RealSense":
                from  gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'], enable_depth=True)
            elif cam_config["type"] == "Kinect":
                from gello.cameras.kinect_camera import KinectCamera
                cam_dict[cam] = KinectCamera(cam, cam_config)
    print("initialized cameras")

    robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(
        robot_client,
        camera_dict=cam_dict,
        control_rate_hz=100.0,
        save_depth_obs=True
    )
    print("initialized robot env, test obs:")

    obs = env.get_obs() # This is just to test, it won't be used in the loop
    for x in obs.keys():
        print(x, obs[x].shape)
    # --- END OF UNCHANGED SECTION ---


    # --- START: MODIFIED SECTION ---
    
    # !!! SET YOUR HDF5 FILE PATH HERE !!!
    dataset_path = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/demos/demo_abs_fixed.hdf5" 
    if not os.path.exists(dataset_path):
        cprint(f"Error: Dataset file not found at {dataset_path}", "red")
        cprint("Please update the 'dataset_path' variable in the 'evaluate' function.", "red")
        return []

    print(f"Using dataset: {dataset_path}")

    # Call the new rollout function with the dataset path
    rollout(
        env, 
        model, 
        dataset_path=dataset_path,
        action_norms=action_norms,
        horizon=rollout_horizon
    )
    # all_results.append(result) # rollout doesn't return results in this version

    # --- END: MODIFIED SECTION ---

    del env
    del model
    del optimizer
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()

    return all_results # This will be empty, but matches the original signature

@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    save_path = cfg.save_path
    result_suffix = cfg.get("result_path_suffix", "")
    result_suffix = f"_{result_suffix}" if result_suffix else result_suffix

    eval_result_dir = os.path.join(save_path, f"eval_results{result_suffix}")
    os.makedirs(eval_result_dir, exist_ok=True)


    setup(cfg)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
    fabric.launch()

    # Add target chaining flag to config if not present
    
    results = evaluate(fabric, cfg, 
                      checkpoint="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy/1027_atm_dp_absact_50_demos_1444_seed1/model_1850.ckpt")
    fabric.barrier()


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()