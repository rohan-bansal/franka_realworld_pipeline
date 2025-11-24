#!/usr/bin/env python3
"""
Script to replay delta_actions from an HDF5 demo on the real robot.

This script:
1. Loads delta_actions from a specified HDF5 demo file
2. Converts each delta_action to a target_pose using action2eef_target_pose
3. Converts target_pose to absolute action (position, quaternion, gripper)
4. Executes the action on the robot via env.step()

Usage:
    python replay_demo_on_robot.py <path_to_demo.hdf5> [--start_idx START] [--end_idx END] [--control_rate RATE]

Arguments:
    demo_path: Path to HDF5 demo file containing delta_actions
    --start_idx: Starting index in the delta_actions array (default: 0)
    --end_idx: Ending index in the delta_actions array (default: all actions)
    --control_rate: Control rate in Hz (default: 30.0)
    --confirm_first: Require confirmation before executing first action (default: True)
    --no_cameras: Run without cameras (default: False)
    --use_dataset_pose: Use dataset poses for delta action calculation instead of real robot pose (helps prevent error accumulation)
    --use_target_chaining: Use target chaining mode - first action uses robot pose, subsequent actions chain from previous targets (prevents error accumulation)

Modes for computing targets:
    DEFAULT: Use real robot pose at each timestep
        target[t] = robot_pose[t] + delta[t]
        Problem: Tracking errors accumulate and can cause overshooting
        
    --use_dataset_pose: Use recorded poses from dataset
        target[t] = dataset_pose[t] + delta[t]
        Good for exact replay, requires poses in HDF5 file
        
    --use_target_chaining: Chain targets together (RECOMMENDED)
        target[0] = robot_pose[0] + delta[0]
        target[t>0] = target[t-1] + delta[t]
        Prevents error accumulation by not feeding back tracking errors

Example:
    python replay_demo_on_robot.py /path/to/demo_0.hdf5
    python replay_demo_on_robot.py /path/to/demo_0.hdf5 --start_idx 10 --end_idx 50
    python replay_demo_on_robot.py /path/to/demo_0.hdf5 --use_dataset_pose
    python replay_demo_on_robot.py /path/to/demo_0.hdf5 --use_target_chaining
"""

import argparse
import h5py
import numpy as np
import sys
import time
import pickle

# Add gello_software to path
sys.path.insert(0, "/mnt/data2/mfm/workspace/Franka/gello_software")

from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv

# Import ATM utilities
from atm.utils.process_utils import action2eef_target_pose, controller, pose2mat
from robosuite.utils.transform_utils import mat2quat


# Camera configuration (can be disabled with --no_cameras)
CAMERA_CONFIG_DICT = {
    "agentview": {
        "sn": "001039114912",
        "type": "Kinect"
    },
    "wrist": {
        "sn": 14620168,
        "type": "Zed",
        'resize': True,
        'resize_resolution': (640, 576)
    }
}


def load_demo_data(demo_path, load_poses=False):
    """
    Load delta_actions and optionally poses from an HDF5 demo file.
    
    Args:
        demo_path: Path to HDF5 demo file
        load_poses: Whether to load original eef_pos and eef_quat
        
    Returns:
        dict: Dictionary containing delta_actions and optionally poses
    """
    data = {}
    
    with h5py.File(demo_path, 'r') as f:
        if 'delta_actions' not in f['root']:
            raise KeyError(
                f"'delta_actions' not found in {demo_path}. "
                f"Available keys: {list(f['root'].keys())}\n"
                f"Please run add_delta_actions_to_demos.py first to add delta_actions."
            )
        
        data['delta_actions'] = f['root']['delta_actions'][:]
        
        # Load original actions if available
        if 'actions' in f['root']:
            data['actions'] = f['root']['actions'][:]
            print(f"Loaded {len(data['actions'])} original actions")
        
        data["ee_ori"] = f['root']['extra_states']['ee_ori'][:]
        data["ee_pos"] = f['root']['extra_states']['ee_pos'][:]
        
    print(f"Loaded {len(data['delta_actions'])} delta actions from {demo_path}")
    print(f"Delta action shape: {data['delta_actions'].shape}")
    
    return data


def initialize_robot_env(use_cameras=True, control_rate_hz=20.0):
    """
    Initialize the robot environment with cameras.
    
    Args:
        use_cameras: Whether to initialize cameras
        control_rate_hz: Control rate in Hz
        
    Returns:
        RobotEnv: Initialized robot environment
    """
    cam_dict = {}
    
    if use_cameras:
        print("Initializing cameras...")
        for cam_name in CAMERA_CONFIG_DICT:
            cam_config = CAMERA_CONFIG_DICT[cam_name]
            
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam_name] = ZedCamera(cam_name, cam_config)
            elif cam_config["type"] == "RealSense":
                from gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam_name] = RealSenseCamera(
                    device_id=cam_config['sn'],
                    enable_depth=True
                )
            elif cam_config["type"] == "Kinect":
                from gello.cameras.kinect_camera import KinectCamera
                cam_dict[cam_name] = KinectCamera(cam_name, cam_config)
        
        print(f"Initialized {len(cam_dict)} cameras")
    else:
        print("Running without cameras")
    
    print("Initializing robot...")
    robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(
        robot_client,
        camera_dict=cam_dict,
        control_rate_hz=control_rate_hz,
        save_depth_obs=use_cameras
    )
    
    print("Robot environment initialized successfully")
    return env


def replay_delta_actions(env, delta_actions, start_idx=0, end_idx=None, confirm_first=True, 
                         dataset_eef_pos=None, dataset_eef_quat=None, use_dataset_pose=False,
                         use_target_chaining=False):
    """
    Replay delta_actions on the robot.
    
    Args:
        env: Robot environment
        delta_actions: Array of delta actions to replay
        start_idx: Starting index in delta_actions array
        end_idx: Ending index in delta_actions array (None = all remaining)
        confirm_first: Whether to require confirmation before first action
        dataset_eef_pos: Dataset poses (position) - required if use_dataset_pose=True
        dataset_eef_quat: Dataset poses (quaternion) - required if use_dataset_pose=True
        use_dataset_pose: If True, use dataset poses instead of real robot pose for delta action calculation
        use_target_chaining: If True, chain targets (t=0: use robot pose, t>0: use previous target pose)
        
    Returns:
        dict: Data collected during replay (target_poses, actual_poses, etc.)
    """
    if end_idx is None:
        end_idx = len(delta_actions)
    
    # Validate indices
    if start_idx < 0 or start_idx >= len(delta_actions):
        raise ValueError(f"start_idx {start_idx} out of range [0, {len(delta_actions)})")
    if end_idx < start_idx or end_idx > len(delta_actions):
        raise ValueError(f"end_idx {end_idx} out of range [{start_idx}, {len(delta_actions)}]")
    
    # Validate mutually exclusive options
    if use_dataset_pose and use_target_chaining:
        raise ValueError("Cannot use both use_dataset_pose and use_target_chaining simultaneously")
    
    # Validate dataset poses if using them
    if use_dataset_pose:
        if dataset_eef_pos is None or dataset_eef_quat is None:
            raise ValueError("use_dataset_pose=True but dataset_eef_pos or dataset_eef_quat is None")
        if len(dataset_eef_pos) < end_idx or len(dataset_eef_quat) < end_idx:
            raise ValueError(
                f"Dataset poses too short: need at least {end_idx} poses, "
                f"got {len(dataset_eef_pos)} eef_pos and {len(dataset_eef_quat)} eef_quat"
            )
    
    print(f"\n{'='*60}")
    print(f"Replaying actions {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
    if use_target_chaining:
        print("Using TARGET CHAINING mode:")
        print("  - t=0: use actual robot pose")
        print("  - t>0: use previous target pose (prevents error accumulation)")
    elif use_dataset_pose:
        print("Using DATASET poses for delta action calculation")
    else:
        print("Using REAL robot poses for delta action calculation")
    print(f"{'='*60}\n")
    
    # Reset robot to starting position
    input("Press Enter to start replay (robot will reset)...")
    env._robot.reset()
    time.sleep(1.0)
    
    # Get initial observation
    obs = env.get_obs()
    print(f"Initial end-effector position: {obs['eef_pos']}")
    print(f"Initial end-effector quaternion: {obs['eef_quat']}")
    
    # Track data for analysis
    replay_data = {
        "target_poses": [],
        "actual_poses": [],
        "delta_actions": [],
        "absolute_actions": [],
        "observations": []
    }
    
    prev_quat = None
    prev_target_pose = None  # For target chaining
    
    try:
        for i in range(start_idx, end_idx):
            action_num = i - start_idx + 1
            total_actions = end_idx - start_idx
            
            print(f"\n--- Action {action_num}/{total_actions} (index {i}) ---")
            
            # Get source pose for computing target (based on mode)
            if use_target_chaining:
                if action_num == 1:
                    # First action: use actual robot pose
                    curr_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                    print(f"Source pose (ROBOT, first action): {obs['eef_pos']}")
                else:
                    # Subsequent actions: use previous target pose
                    curr_pose = prev_target_pose
                    print(f"Source pose (PREV TARGET): {prev_target_pose[:3, 3]}")
            elif use_dataset_pose:
                # Use dataset pose for delta action calculation
                curr_pos = dataset_eef_pos[i]
                curr_quat = dataset_eef_quat[i]
                curr_pose = pose2mat(curr_pos, curr_quat)
                print(f"Source pose (DATASET): {curr_pos}")
            else:
                # Use real robot pose from observations
                curr_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                print(f"Source pose (ROBOT): {obs['eef_pos']}")
            
            # Get delta action
            delta_action = delta_actions[i]
            print(f"Delta action: {delta_action}")
            
            # Convert delta_action to target_pose using action2eef_target_pose
            target_pose, gripper_action = action2eef_target_pose(
                curr_pose, delta_action, controller
            )
            
            # Extract position and rotation from target pose
            pos = target_pose[:3, 3]
            rot = target_pose[:3, :3]
            quat = mat2quat(rot)
            
            # Ensure quaternion continuity (avoid sign flip)
            if prev_quat is not None:
                if np.dot(quat, prev_quat) < 0.0:
                    quat = -quat
            prev_quat = quat
            
            # Construct absolute action: [pos(3), quat(4), gripper(1)]
            abs_action = np.concatenate([pos, quat, np.array(gripper_action)])
            
            print(f"Target position: {pos}")
            print(f"Target quaternion: {quat}")
            print(f"Gripper action: {gripper_action}")
            print(f"Absolute action: {abs_action}")
            
            # Confirm first action if requested
            if confirm_first and action_num == 1:
                input(f"\nPress Enter to execute action {action_num}/{total_actions}...")
                confirm_first = False  # Only confirm first action
            
            # Execute action on robot
            obs, executed_action = env.step(abs_action.tolist())
            
            # Record actual pose after execution
            actual_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
            
            # Store target pose for next iteration (if using target chaining)
            if use_target_chaining:
                prev_target_pose = target_pose
            
            # Store data
            replay_data["target_poses"].append(target_pose)
            replay_data["actual_poses"].append(actual_pose)
            replay_data["delta_actions"].append(delta_action)
            replay_data["absolute_actions"].append(abs_action)
            replay_data["observations"].append({
                "eef_pos": obs["eef_pos"],
                "eef_quat": obs["eef_quat"],
                "gripper_position": obs.get("gripper_position", None),
                "joint_positions": obs.get("joint_positions", None)
            })
            
            # Compute and print error
            pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
            print(f"Position error: {pos_error:.4f} m")
            
            # time.sleep(0.1)  # Small delay between actions
            
    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user!")
    
    print(f"\n{'='*60}")
    print(f"Replay complete! Executed {len(replay_data['actual_poses'])} actions")
    print(f"{'='*60}\n")
    
    return replay_data


def main():
    parser = argparse.ArgumentParser(
        description="Replay delta_actions from HDF5 demo on real robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'demo_path',
        type=str,
        help='Path to HDF5 demo file containing delta_actions'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index in delta_actions array (default: 0)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        default=None,
        help='Ending index in delta_actions array (default: all actions)'
    )
    parser.add_argument(
        '--control_rate',
        type=float,
        default=20.0,
        help='Control rate in Hz (default: 20.0)'
    )
    parser.add_argument(
        '--no_confirm',
        action='store_true',
        help='Do not require confirmation before first action'
    )
    parser.add_argument(
        '--no_cameras',
        action='store_true',
        help='Run without cameras'
    )
    parser.add_argument(
        '--save_data',
        type=str,
        default=None,
        help='Path to save replay data (default: no save)'
    )
    parser.add_argument(
        '--use_dataset_pose',
        action='store_true',
        help='Use dataset poses instead of real robot pose for delta action calculation (helps prevent error accumulation)'
    )
    parser.add_argument(
        '--use_target_chaining',
        action='store_true',
        help='Use target chaining: t=0 uses robot pose, t>0 uses previous target pose (prevents error accumulation)'
    )
    
    args = parser.parse_args()
    
    # Load delta_actions and optionally poses from demo file
    print(f"Loading data from {args.demo_path}")
    demo_data = load_demo_data(args.demo_path, load_poses=args.use_dataset_pose)
    delta_actions = demo_data['delta_actions']
    
    # Get dataset poses if requested
    dataset_eef_pos = demo_data.get('ee_pos', None)
    dataset_eef_quat = demo_data.get('ee_ori', None)
    
    # Initialize robot environment
    env = initialize_robot_env(
        use_cameras=not args.no_cameras,
        control_rate_hz=args.control_rate
    )
    
    # Replay delta_actions
    replay_data = replay_delta_actions(
        env,
        delta_actions,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        confirm_first=not args.no_confirm,
        dataset_eef_pos=dataset_eef_pos,
        dataset_eef_quat=dataset_eef_quat,
        use_dataset_pose=args.use_dataset_pose,
        use_target_chaining=args.use_target_chaining
    )
    
    # Save data if requested
    if args.save_data is not None:
        print(f"\nSaving replay data to {args.save_data}")
        with open(args.save_data, 'wb') as f:
            pickle.dump(replay_data, f)
        print("Data saved successfully!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

