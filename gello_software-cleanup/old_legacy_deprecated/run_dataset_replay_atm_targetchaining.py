#!/usr/bin/env python3
"""
Script to REPLAY a dataset trajectory on the real robot.

This script:
1. Loads a 'DatasetReplayPolicy' that reads from a specified HDF5 demo file.
2. This policy generates delta actions on-the-fly using the *exact*
   target chaining logic from dp_dataloader.py.
3. The real robot rollout loop executes these ground-truth delta actions.
4. This allows for replaying a dataset trajectory using the real-world
   rollout pipeline for visualization and debugging.

**NOTE:** This version is hardcoded to *always* use target chaining.

Dry Run Mode:
- Pass --dry_run to calculate the entire trajectory without
  stepping the real robot environment.
  
Usage:
    # Real execution
    python replay_dataset_policy.py --demo_path /path/to/demo.hdf5 --T_act 16
    
    # Dry run (no robot motion)
    python replay_dataset_policy.py --demo_path /path/to/demo.hdf5 --T_act 16 --dry_run
"""

import math
import torch
import time
import datetime
import pickle
import os
import cv2
import h5py # Added for HDF5 loading
import argparse # Added to replace Hydra
import warnings # Added to suppress warnings
import gc

import sys
sys.path.insert(0, "/mnt/data2/mfm/workspace/Franka/gello_software")

import numpy as np

from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv

from atm.utils.process_utils import eef_target_pose2action, action2eef_target_pose, controller, pose2mat
# Import helpers needed for delta action conversion
from robosuite.utils.transform_utils import mat2quat, axisangle2mat, vec2axisangle, quat2mat 
from scipy.spatial.transform import Rotation as R

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

# -----------------------------------------------------------------------------
# DatasetReplayPolicy Class
# -----------------------------------------------------------------------------

class DatasetReplayPolicy:
    """
    A 'fake' policy that mimics a trained model but instead loads actions
    from a dataset file and converts them to deltas using the dataloader's logic.
    (Hardcoded to use target chaining)
    """
    def __init__(self, demo_path, T_act, extra_state_keys):
        print(f"Initializing DatasetReplayPolicy with demo: {demo_path}")
        self.T_act = T_act
        self.use_target_chaining = True # Hardcoded
        self.extra_state_keys = extra_state_keys # Store this for the real obs dict
        
        # Load HDF5 data
        self.demo_data = self.load_hdf5_data(demo_path)
        self.demo_len = len(self.demo_data["actions"])
        self.t = 0 # Current timestep in the demo
        
        print(f"  - Loaded demo of length {self.demo_len}")
        print(f"  - T_act (horizon): {self.T_act}")
        print(f"  - Using Target Chaining: True (Hardcoded)")

    def load_hdf5_data(self, file_path):
        """Loads the 'root' group from an HDF5 file."""
        f = h5py.File(file_path, 'r')
        return f["root"]

    def eval(self):
        """Dummy method to mimic a real policy."""
        pass

    def reset(self):
        """Resets the policy's internal timestep counter."""
        print("Resetting DatasetReplayPolicy timestep to 0")
        self.t = 0

    def act(self, rgb, task_emb, extra_states, **kwargs):
        """
        Returns a chunk of abs actions from the dataset.
        Ignores all observation inputs.
        """
        time_offset = self.t
        
        if time_offset >= self.demo_len:
            print("Dataset replay finished.")
            return [], None # Return empty list to stop rollout

        # Get the chunk of *absolute* actions from the dataset
        # Need T_act actions to generate T_act delta actions
        abs_actions = self.demo_data["actions"][time_offset : time_offset + self.T_act]

        # Convert absolute actions to delta actions using dataloader logic
        delta_actions_chunk = self._convert_to_delta_actions(self.demo_data, abs_actions, time_offset)
        
        # Advance the internal timestep counter by the number of actions we're returning
        self.t += len(delta_actions_chunk) 
        
        print(f"ReplayPolicy: provided {len(delta_actions_chunk)} delta actions (t={time_offset} to {self.t})")
        
        # Return delta actions and None for tracks
        return delta_actions_chunk, None

    # --- Helper functions copied directly from dp_dataloader.py ---
    
    def _get_target_pose_from_action(self, action):
        """
        Convert an action (absolute target pose) to a 4x4 pose matrix.
        """
        target_pos = action[:3]  # position (x, y, z)
        target_ori = action[3:6]  # orientation (assumed axis-angle format)
        
        if hasattr(target_pos, 'numpy'):
            target_pos = target_pos.detach().cpu().numpy()
        if hasattr(target_ori, 'numpy'):
            target_ori = target_ori.detach().cpu().numpy()
        
        axis, angle = vec2axisangle(target_ori)
        target_ori_mat = axisangle2mat(axis, angle)
        target_pose = pose2mat(target_pos, target_ori_mat)
        
        return target_pose

    def _get_current_eef_pose(self, demo, time_index):
        """
        Gets the ground-truth pose from the dataset at a specific timestep.
        NOTE: Modified to use 'extra_states' to match eval_mv_bc_real.py.
        """
        if 'ee_pos' in demo['extra_states'] and 'ee_ori' in demo['extra_states']:
            ee_pos = demo['extra_states']['ee_pos'][time_index]  # (3,)
            ee_quat = demo['extra_states']['ee_ori'][time_index]  # (4,)
            
            if hasattr(ee_pos, 'numpy'):
                ee_pos = ee_pos.detach().cpu().numpy()
            if hasattr(ee_quat, 'numpy'):
                ee_quat = ee_quat.detach().cpu().numpy()
            
            ee_ori_mat = quat2mat(ee_quat)
            current_pose = pose2mat(ee_pos, ee_ori_mat)
            return current_pose
        else:
            raise KeyError(f"Required keys 'ee_pos' and/or 'ee_ori' not found in demo['extra_states']. "
                           f"Available keys: {list(demo['extra_states'].keys())}")

    def _convert_to_delta_actions(self, demo, actions, time_offset):
        """
        Convert absolute actions to delta actions using dataloader logic.
        (Hardcoded to use target chaining)
        """
        delta_actions = []
        
        # Target chaining method: process all T_act actions
        for t in range(len(actions)):
            if t == 0:
                current_time = time_offset + t
                source_pose = self._get_current_eef_pose(demo, current_time)
            else:
                source_pose = self._get_target_pose_from_action(actions[t - 1])
            
            current_action = actions[t]
            target_pose = self._get_target_pose_from_action(current_action)
            
            gripper_action = current_action[6:7]
            if hasattr(gripper_action, 'numpy'):
                gripper_action = gripper_action.detach().cpu().numpy()
            
            delta_action = eef_target_pose2action(source_pose, target_pose, controller, gripper_action)
            delta_actions.append(delta_action)
        
        delta_actions = np.array(delta_actions, dtype=np.float32)
        
        # We must return exactly the number of actions requested,
        # even if it's the end of the demo
        return delta_actions


# -----------------------------------------------------------------------------
# Main Rollout/Evaluate Functions
# -----------------------------------------------------------------------------

@torch.no_grad()
def rollout(env, policy, horizon=None, action_norms=None, dry_run=False):
    
    if dry_run:
        print("\n*** DRY RUN MODE ENABLED. ROBOT WILL NOT MOVE. ***")
        input("continue with DRY RUN REPLAY? [enter to continue]...")
    else:
        input("continue with REAL REPLAY? (Target Chaining is ON) [enter to continue]...")

    policy.eval()

    if not dry_run:
        env._robot.reset()
    
    policy.reset()
    done = False
    step_i = 0

    confirm_first_action = False
    delta_action_queue = []

    input_output = []

    prev_quat = None
    prev_target_pose = None  # For target chaining
    
    print(f"\n{'='*60}")
    print("Using TARGET CHAINING mode (Hardcoded):")
    print("  - t=0 (of horizon): use actual robot pose")
    print("  - t>0 (of horizon): use previous target pose")
    print(f"{'='*60}\n")
    
    # Get the very *first* observation. This is the starting point.
    obs = env.get_obs()

    try:
        while not done and (horizon is None or step_i < horizon):

            # FOR REAL ROBOT: Get *new* obs at the start of each horizon for re-syncing
            # FOR DRY RUN: `obs` stays as the *initial* obs (from outside the loop).
            if not dry_run:
                obs = env.get_obs()

            # Get real-time observations for logging
            initial_ee_pos = obs["eef_pos"]
            initial_ee_ori = obs["eef_quat"]
            color_in_depth_frame = obs["agentview_color_in_depth_frame"]
            agentview_rgb = cv2.resize(obs["agentview_image"], (128,128), interpolation=cv2.INTER_LINEAR)
            wrist_rgb = cv2.resize(obs["wrist_image"], (128, 128), interpolation=cv2.INTER_LINEAR)

            # Get new delta actions if queue is empty
            if len(delta_action_queue) == 0:
                # policy.act() ignores inputs; pass None
                delta_actions, _tracks = policy.act(None, None, None)
                
                if len(delta_actions) == 0:
                    print("Policy returned no actions. Ending rollout.")
                    done = True
                    break
                    
                delta_action_queue = list(delta_actions)
                print(f"Got {len(delta_action_queue)} new delta actions from REPLAY policy")

            # Execute actions and track results
            prev_target_poses = []
            target_poses = []
            actual_poses = []
            
            for i in range(len(delta_action_queue)):
                delta_action = delta_action_queue[i]
                
                # Get source pose for computing target
                if (not dry_run and i == 0):
                    # Very first action horizon
                    curr_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                    print("i=0")
                elif (not dry_run and i > 0):
                    curr_pose = prev_target_pose
                    print("i>0")
                else:
                    # DRY RUN (after first step) OR
                    # REAL ROBOT (mid-horizon): use previous target pose
                    curr_pose = prev_target_pose

                # Compute target from source pose
                prev_target_poses.append(curr_pose)
                target_pose, gripper_action = action2eef_target_pose(curr_pose, delta_action, controller)
                target_poses.append(target_pose)
                
                # Store target pose for next iteration
                prev_target_pose = target_pose
                
                pos = target_pose[:3, 3]
                rot = target_pose[:3, :3]
                quat = mat2quat(rot)
                
                if prev_quat is not None:
                    if np.dot(quat, prev_quat) < 0.0:
                        quat = -quat
                prev_quat = quat

                abs_action = np.concatenate([pos, quat, np.array(gripper_action)])
                
                print(f"Executing action {i+1}/{len(delta_action_queue)} in horizon")
                print(f"  Delta action (from dataset): {delta_action}")
                print(f"  Target position (for robot): {pos}")

                # if not confirm_first_action and not dry_run: # Only ask for confirmation on real run
                input(f"intended action (with quat) [enter to rollout]: {abs_action}")
                    # confirm_first_action = True

                # --- Execute Step ---
                if not dry_run:
                    # Execute and save resulting pose
                    obs, act = env.step(abs_action.tolist())
                    actual_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                    
                    pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
                    print(f"  Position tracking error: {pos_error:.4f} m")
                else:
                    # DRY RUN: Don't step. "Actual" pose is just the target pose.
                    actual_pose = target_pose 
                    print("  (Dry Run) Skipping env.step()")
                
                actual_poses.append(actual_pose)
                # --- End Execute Step ---

            # Clear the queue after executing all actions
            delta_action_queue = []
            
            save_data = {
                "step_i": step_i,
                "initial_pose": pose2mat(initial_ee_pos, initial_ee_ori),
                "prev_target_poses": prev_target_poses,
                "target_poses": target_poses,
                "actual_poses": actual_poses,
                "use_target_chaining": True, # Hardcoded
                "dry_run": dry_run,
                "images": {
                    "agentview": agentview_rgb,
                    "wrist": wrist_rgb,
                    "depth_color": color_in_depth_frame
                }
            }
            input_output.append(save_data)
            
            step_i += 1
            
            # Check if policy is done
            if policy.t >= policy.demo_len:
                done = True

    except KeyboardInterrupt:
        with open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data_REPLAY.pkl", 'wb') as f:
            pickle.dump(input_output, f)

        print("data saved to recorded_data_REPLAY.pkl")
        sys.exit()

    print("Rollout finished.")
    with open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data_REPLAY.pkl", 'wb') as f:
        pickle.dump(input_output, f)
    print("data saved to recorded_data_REPLAY.pkl")
    return


def evaluate(demo_path, T_act, dry_run=False):
    
    # --- MODIFIED: Load Replay Policy instead of trained model ---
    print(f"Loading DatasetReplayPolicy from: {demo_path}")
    
    # Hardcode extra_state_keys since cfg is gone
    extra_state_keys = ["joint_states", "gripper_states"]

    policy = DatasetReplayPolicy(
        demo_path=demo_path, 
        T_act=T_act, 
        extra_state_keys=extra_state_keys
    )
    
    # Set rollout horizon to the length of the demo to ensure it plays fully
    rollout_horizon = math.ceil(policy.demo_len / T_act)
    print(f"Demo length: {policy.demo_len}, T_act: {T_act}. Setting rollout horizon to {rollout_horizon} steps.")
    # --- END MODIFICATION ---

    all_results = []

    # Action norms are not needed for replay, but pass None
    action_norms = None
    
    print("creating real world environment")

    if camera_config_dict is not None:
        for cam in camera_config_dict:
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
        control_rate_hz=20.0,
        save_depth_obs=True
    )
    print("initialized robot env, test obs:")

    obs = env.get_obs()
    for x in obs.keys():
        print(x, obs[x].shape)

    result = rollout(env, policy, horizon=rollout_horizon, action_norms=action_norms, dry_run=dry_run)

    del env
    del policy
    gc.collect()

    return all_results

def main():
    warnings.simplefilter("ignore") # Suppress warnings

    parser = argparse.ArgumentParser(description="Replay a dataset trajectory on the real robot.")
    parser.add_argument("--demo_path", type=str, required=True, 
                        help="Path to the HDF5 demo file.")
    parser.add_argument("--T_act", type=int, required=True, 
                        help="Action horizon (e.g., 16).")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Enable dry run mode (no robot motion).")
    
    args = parser.parse_args()
    
    print(f"--- Dataset Replay ---")
    print(f"  Demo file: {args.demo_path}")
    print(f"  T_act: {args.T_act}")
    print(f"  Target Chaining: True (Hardcoded)")
    print(f"  DRY RUN: {args.dry_run}")
    print(f"----------------------")
    
    evaluate(demo_path=args.demo_path,
             T_act=args.T_act,
             dry_run=args.dry_run)


if __name__ == "__main__":
    main()