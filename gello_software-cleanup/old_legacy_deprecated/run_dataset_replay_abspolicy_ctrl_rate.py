#!/usr/bin/env python3
"""
Script to REPLAY a dataset trajectory on the real robot.

This script:
1. Loads a 'DatasetReplayPolicy' that reads from a specified HDF5 demo file.
2. This policy directly returns absolute actions from the dataset.
3. The real robot rollout loop executes these ground-truth absolute actions.
4. This allows for replaying a dataset trajectory using the real-world
   rollout pipeline for visualization and debugging.
  
Usage:
    python replay_dataset_policy.py --demo_path /path/to/demo.hdf5 --T_act 16
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

from atm.utils.process_utils import controller, pose2mat
# Import helpers needed for action processing
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
    A 'fake' policy that mimics a trained model but instead loads absolute actions
    directly from a dataset file.
    """
    def __init__(self, demo_path, T_act, extra_state_keys):
        print(f"Initializing DatasetReplayPolicy with demo: {demo_path}")
        self.T_act = T_act
        self.extra_state_keys = extra_state_keys # Store this for the real obs dict
        
        # Load HDF5 data
        self.demo_data = self.load_hdf5_data(demo_path)
        self.demo_len = len(self.demo_data["actions"])
        self.t = 0 # Current timestep in the demo
        
        print(f"  - Loaded demo of length {self.demo_len}")
        print(f"  - T_act (horizon): {self.T_act}")
        print(f"  - Replaying absolute actions directly from dataset")

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

    def act(self):
        """
        Returns a chunk of absolute actions from the dataset.
        Ignores all observation inputs.
        """
        time_offset = self.t
        
        if time_offset >= self.demo_len:
            print("Dataset replay finished.")
            return [], None # Return empty list to stop rollout

        # Get the chunk of *absolute* actions from the dataset
        abs_actions = self.demo_data["actions"][time_offset : time_offset + self.T_act]
        
        # Convert to numpy array if needed
        if hasattr(abs_actions, 'numpy'):
            abs_actions = abs_actions.detach().cpu().numpy()
        abs_actions = np.array(abs_actions, dtype=np.float32)
        
        # Advance the internal timestep counter by the number of actions we're returning
        self.t += len(abs_actions) 
        
        print(f"ReplayPolicy: provided {len(abs_actions)} absolute actions (t={time_offset} to {self.t})")
        
        # Return absolute actions and None for tracks
        return abs_actions, None



# -----------------------------------------------------------------------------
# Main Rollout/Evaluate Functions
# -----------------------------------------------------------------------------

@torch.no_grad()
def rollout(env, policy, horizon=None):
    
    input("continue with REPLAY? [enter to continue]...")

    policy.eval()
    env._robot.reset()
    
    policy.reset()
    done = False
    step_i = 0

    confirm_first_action = False
    action_queue = []

    input_output = []

    prev_quat = None
    
    # Frequency tracking variables
    loop_start_times = []
    frequency_print_interval = 10  # Print frequency every N steps
    
    # Timing control for consistent execution rate
    target_frequency = 15.0  # Hz
    target_loop_time = 1.0 / target_frequency  # seconds per loop
    print(f"  - Target execution frequency: {target_frequency} Hz")
    print(f"  - Target loop time: {target_loop_time:.4f} seconds")
    
    print(f"\n{'='*60}")
    print("ABSOLUTE ACTION REPLAY mode:")
    print("  - Actions are replayed directly from dataset")
    print(f"{'='*60}\n")
    
    # Get the very *first* observation. This is the starting point.
    obs = env.get_obs()

    try:
        while not done and (horizon is None or step_i < horizon):
            
            # Record loop start time for frequency calculation
            loop_start_time = time.time()
            loop_start_times.append(loop_start_time)
            
            # Calculate and print frequency every N steps
            if len(loop_start_times) >= 2 and step_i % frequency_print_interval == 0:
                # Calculate frequency over the last N iterations
                recent_times = loop_start_times[-min(frequency_print_interval, len(loop_start_times)):]
                if len(recent_times) >= 2:
                    time_span = recent_times[-1] - recent_times[0]
                    actual_frequency = (len(recent_times) - 1) / time_span
                    print(f"\n--- EXECUTION FREQUENCY ---")
                    print(f"Step {step_i}: Actual loop frequency: {actual_frequency:.2f} Hz")
                    print(f"Target frequency: {target_frequency} Hz")
                    print(f"Frequency ratio: {actual_frequency/target_frequency:.2f}")
                    print(f"---------------------------\n")

            # Get new obs at the start of each horizon for re-syncing
            obs = env.get_obs()

            # Get real-time observations for logging
            initial_ee_pos = obs["eef_pos"]
            initial_ee_ori = obs["eef_quat"]
            color_in_depth_frame = obs["agentview_color_in_depth_frame"]
            agentview_rgb = cv2.resize(obs["agentview_image"], (128,128), interpolation=cv2.INTER_LINEAR)
            wrist_rgb = cv2.resize(obs["wrist_image"], (128, 128), interpolation=cv2.INTER_LINEAR)

            # Get new absolute actions if queue is empty
            if len(action_queue) == 0:
                # policy.act() ignores inputs; pass None
                abs_actions, _tracks = policy.act()
                
                if len(abs_actions) == 0:
                    print("Policy returned no actions. Ending rollout.")
                    done = True
                    break
                    
                action_queue = list(abs_actions)
                print(f"Got {len(action_queue)} new absolute actions from REPLAY policy")

            # Execute actions and track results
            executed_actions = []
            target_poses = []
            actual_poses = []
            
            for i in range(len(action_queue)):
                abs_action = action_queue[i]
                
                # Extract position, orientation, and gripper from absolute action
                pos = abs_action[:3]
                ori = abs_action[3:6]  # Assuming axis-angle format
                gripper_action = abs_action[6:7] if len(abs_action) > 6 else [0.0]
                
                # Convert axis-angle to quaternion for robot execution
                if hasattr(ori, 'numpy'):
                    ori = ori.detach().cpu().numpy()
                
                # Convert axis-angle to rotation matrix, then to quaternion
                axis, angle = vec2axisangle(ori)
                rot_mat = axisangle2mat(axis, angle)
                quat = mat2quat(rot_mat)
                
                # Convert absolute action to target pose for saving
                # The absolute action already contains the target pose in world coordinates
                target_pose = pose2mat(pos, rot_mat)
                target_poses.append(target_pose)
                
                # Handle quaternion sign consistency
                if prev_quat is not None:
                    if np.dot(quat, prev_quat) < 0.0:
                        quat = -quat
                prev_quat = quat

                # Create final action for robot execution
                robot_action = np.concatenate([pos, quat, np.array(gripper_action)])
                
                print(f"Executing action {i+1}/{len(action_queue)} in horizon")
                print(f"  Absolute action (from dataset): {abs_action}")
                print(f"  Robot action (pos + quat + gripper): {robot_action}")
                print(f"  Current robot position: {obs['eef_pos']}")
                print(f"  Target position: {pos}")
                print(f"  Distance to target: {np.linalg.norm(pos - obs['eef_pos'])*1000:.2f} mm")

                if not confirm_first_action:
                    input(f"intended action [enter to rollout]: {robot_action}")
                    confirm_first_action = True

                # Execute and save resulting pose
                obs_before = obs["eef_pos"].copy()
                obs, act = env.step(robot_action.tolist())
                actual_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                
                pos_error = np.linalg.norm(obs["eef_pos"] - pos)
                movement = np.linalg.norm(obs["eef_pos"] - obs_before)
                print(f"  Position tracking error: {pos_error:.4f} m")
                print(f"  Actual movement: {movement*1000:.2f} mm")
                print(f"  Final robot position: {obs['eef_pos']}")
                
                executed_actions.append(robot_action)
                actual_poses.append(actual_pose)

            # Clear the queue after executing all actions
            action_queue = []
            
            save_data = {
                "step_i": step_i,
                "initial_pose": pose2mat(initial_ee_pos, initial_ee_ori),
                "executed_actions": executed_actions,
                "target_poses": target_poses,
                "actual_poses": actual_poses,
                "absolute_action_replay": True,
                "images": {
                    "agentview": agentview_rgb,
                    "wrist": wrist_rgb,
                    "depth_color": color_in_depth_frame
                }
            }
            input_output.append(save_data)

            
            
            # Timing control to maintain target frequency
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            sleep_time = target_loop_time - loop_duration
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                actual_loop_time = target_loop_time
            else:
                actual_loop_time = loop_duration
                print(f"WARNING: Loop {step_i} took {loop_duration:.4f}s, longer than target {target_loop_time:.4f}s")
            
            # Print timing info every few steps
            if step_i % frequency_print_interval == 0:
                print(f"Step {step_i} timing: loop={loop_duration:.4f}s, sleep={max(0, sleep_time):.4f}s, total={actual_loop_time:.4f}s")
            
            step_i += 1
            
            # Check if policy is done
            if policy.t >= policy.demo_len:
                done = True

    except KeyboardInterrupt:
        # Print frequency statistics before exiting
        if len(loop_start_times) >= 2:
            total_time = loop_start_times[-1] - loop_start_times[0]
            total_iterations = len(loop_start_times) - 1
            average_frequency = total_iterations / total_time
            print(f"\n--- INTERRUPTED - FREQUENCY STATISTICS ---")
            print(f"Total iterations: {total_iterations}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average frequency: {average_frequency:.2f} Hz")
            print(f"Target frequency: {target_frequency} Hz")
            print(f"Overall frequency ratio: {average_frequency/target_frequency:.2f}")
            print(f"------------------------------------------\n")
        
        with open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data_REPLAY.pkl", 'wb') as f:
            pickle.dump(input_output, f)

        print("data saved to recorded_data_REPLAY.pkl")
        sys.exit()

    print("Rollout finished.")
    
    # Print final frequency statistics
    if len(loop_start_times) >= 2:
        total_time = loop_start_times[-1] - loop_start_times[0]
        total_iterations = len(loop_start_times) - 1
        average_frequency = total_iterations / total_time
        print(f"\n--- FINAL FREQUENCY STATISTICS ---")
        print(f"Total iterations: {total_iterations}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average frequency: {average_frequency:.2f} Hz")
        print(f"Target frequency: {target_frequency} Hz")
        print(f"Overall frequency ratio: {average_frequency/target_frequency:.2f}")
        print(f"----------------------------------\n")
    
    with open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data_REPLAY.pkl", 'wb') as f:
        pickle.dump(input_output, f)
    print("data saved to recorded_data_REPLAY.pkl")
    return


def evaluate(demo_path, T_act):
    
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

    all_results = []
    
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
        control_rate_hz=100.0,
        save_depth_obs=True
    )
    print("initialized robot env, test obs:")

    obs = env.get_obs()
    for x in obs.keys():
        print(x, obs[x].shape)

    result = rollout(env, policy, horizon=rollout_horizon)

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
    
    args = parser.parse_args()
    
    print(f"--- Dataset Replay ---")
    print(f"  Demo file: {args.demo_path}")
    print(f"  T_act: {args.T_act}")
    print(f"  Mode: Absolute Action Replay")
    print(f"----------------------")
    
    evaluate(demo_path=args.demo_path,
             T_act=args.T_act)


if __name__ == "__main__":
    main()