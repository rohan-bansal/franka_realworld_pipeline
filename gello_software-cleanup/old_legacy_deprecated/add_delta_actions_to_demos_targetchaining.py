#!/usr/bin/env python3
"""
Script to add delta_actions to HDF5 demo files (IN-PLACE modification).

This script processes a folder of HDF5 demos and adds a "delta_actions" key
that contains delta actions computed from the absolute actions.
Files are modified in-place by default. Only top-level HDF5 files are processed
(no subdirectories) to avoid processing symlinked bc_train folders.

Usage:
    python add_delta_actions_to_demos.py <demos_folder>
    
Arguments:
    demos_folder: Path to folder containing HDF5 demo files
    --recursive: Search recursively in subdirectories (default: only top-level)
    --single_file: Process a single HDF5 file instead of a folder
    --dry-run: List files that would be processed without modifying them
    --target-chaining: Use target chaining method (see below)
    
Delta Action Computation Methods:
    DEFAULT METHOD (current_state):
        Delta at timestep t = current_pose[t] -> target_pose[t+1]
        Always uses the actual robot state as the source pose.
        
    ALTERNATIVE METHOD (--target-chaining):
        Delta at timestep 0 = current_pose[0] -> target_pose[1]
        Delta at timestep t>0 = target_pose[t] -> target_pose[t+1]
        Creates a chain of target poses (only uses actual state at t=0).
    
Examples:
    # Process all top-level demos in a folder (default)
    python add_delta_actions_to_demos.py /path/to/demos/
    
    # Process demos recursively (including subdirectories)
    python add_delta_actions_to_demos.py /path/to/demos/ --recursive
    
    # Process using target chaining method
    python add_delta_actions_to_demos.py /path/to/demos/ --target-chaining
    
    # Process a single file
    python add_delta_actions_to_demos.py --single_file /path/to/demo_0.hdf5
    
    # Dry run to see what would be processed
    python add_delta_actions_to_demos.py /path/to/demos/ --dry-run
"""

import argparse
import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

# Import required functions from ATM
from atm.utils.process_utils import eef_target_pose2action, controller, pose2mat
from robosuite.utils.transform_utils import axisangle2mat, vec2axisangle, quat2mat


def get_current_eef_pose(demo, time_index):
    """
    Get the current end-effector pose at a given time index.
    
    Args:
        demo: Demo data dictionary (h5py group)
        time_index: Time index to get pose for
        
    Returns:
        np.array: 4x4 pose matrix
    """
    # Get end-effector position and quaternion from ee_data
    if 'ee_pos' in demo['extra_states'] and 'ee_ori' in demo['extra_states']:
        ee_pos = demo['extra_states']['ee_pos'][time_index]  # (3,)
        ee_quat = demo['extra_states']['ee_ori'][time_index]  # (4,)
        
        # Convert to numpy if needed
        if not isinstance(ee_pos, np.ndarray):
            ee_pos = np.array(ee_pos)
        if not isinstance(ee_quat, np.ndarray):
            ee_quat = np.array(ee_quat)
        
        # Convert quaternion to rotation matrix
        ee_ori_mat = quat2mat(ee_quat)
        
        # Construct 4x4 pose matrix
        current_pose = pose2mat(ee_pos, ee_ori_mat)
        return current_pose
    else:
        raise KeyError(f"Required keys 'ee_pos' and/or 'ee_ori' not found in extra_states. "
                      f"Available keys: {list(demo['extra_states'].keys())}")


def get_target_pose_from_action(action):
    """
    Convert an action (absolute target pose) to a 4x4 pose matrix.
    
    Args:
        action: Action array with shape (action_dim,) containing [pos, ori, gripper]
        
    Returns:
        np.array: 4x4 pose matrix
    """
    target_pos = action[:3]  # position (x, y, z)
    target_ori = action[3:6]  # orientation (assumed axis-angle format)
    
    # Convert to numpy if needed
    if not isinstance(target_pos, np.ndarray):
        target_pos = np.array(target_pos)
    if not isinstance(target_ori, np.ndarray):
        target_ori = np.array(target_ori)
    
    # Convert target orientation to rotation matrix
    axis, angle = vec2axisangle(target_ori)
    target_ori_mat = axisangle2mat(axis, angle)
    target_pose = pose2mat(target_pos, target_ori_mat)
    
    return target_pose


def convert_to_delta_actions(demo, actions):
    """
    Convert absolute actions to delta actions.
    
    CURRENT METHOD: Delta action at timestep t is computed from:
    - Current EE pose at t (actual robot state)
    - Target pose from action at t+1
    
    Args:
        demo: Demo data dictionary (h5py group)
        actions: Array of actions with shape (T, action_dim)
        
    Returns:
        np.array: Delta actions with shape (T, action_dim)
                  The last delta action is a dummy (zero) action to keep length equal to actions.
    """
    delta_actions = []
    
    for t in range(len(actions) - 1):  # Stop one step early since we need t+1
        # Get current end-effector pose
        current_pose = get_current_eef_pose(demo, t)
        
        # Parse action components from NEXT timestep
        next_action = actions[t + 1]
        target_pos = next_action[:3]  # position (x, y, z)
        target_ori = next_action[3:6]  # orientation (assumed axis-angle format)
        gripper_action = next_action[6:7]  # gripper command
        
        # Convert to numpy if needed
        if not isinstance(target_pos, np.ndarray):
            target_pos = np.array(target_pos)
        if not isinstance(target_ori, np.ndarray):
            target_ori = np.array(target_ori)
        if not isinstance(gripper_action, np.ndarray):
            gripper_action = np.array(gripper_action)
        
        # Convert target orientation to rotation matrix
        axis, angle = vec2axisangle(target_ori)
        target_ori_mat = axisangle2mat(axis, angle)
        target_pose = pose2mat(target_pos, target_ori_mat)
        
        # Convert to delta action using the function from process_utils
        delta_action = eef_target_pose2action(current_pose, target_pose, controller, gripper_action)
        delta_actions.append(delta_action)
    
    # Convert to numpy array with float32 dtype
    delta_actions = np.array(delta_actions, dtype=np.float32)
    
    # Add a dummy action at the end to keep length equal to actions
    if len(delta_actions) > 0:
        dummy_action = np.zeros_like(delta_actions[0])
        last_gripper = delta_actions[-1][6:7]
        dummy_action[6:7] = last_gripper
        delta_actions = np.vstack([delta_actions, dummy_action])
    
    return delta_actions


def convert_to_delta_actions_target_chaining(demo, actions):
    """
    Convert absolute actions to delta actions using TARGET CHAINING method.
    
    ALTERNATIVE METHOD: Delta action at timestep t is computed from:
    - t=0: Current EE pose at 0 -> Target pose at 1
    - t>0: Target pose at t -> Target pose at t+1
    
    This creates a chain of target poses rather than always using the actual robot state.
    
    Args:
        demo: Demo data dictionary (h5py group)
        actions: Array of actions with shape (T, action_dim)
        
    Returns:
        np.array: Delta actions with shape (T, action_dim)
                  The last delta action is a dummy (zero) action to keep length equal to actions.
    """
    delta_actions = []
    
    for t in range(len(actions) - 1):  # Stop one step early since we need t+1
        if t == 0:
            # First timestep: use current EE pose
            source_pose = get_current_eef_pose(demo, t)
        else:
            # Subsequent timesteps: use previous target pose
            source_pose = get_target_pose_from_action(actions[t])
        
        # Get target pose from next action
        next_action = actions[t + 1]
        target_pose = get_target_pose_from_action(next_action)
        
        # Extract gripper action
        gripper_action = next_action[6:7]
        if not isinstance(gripper_action, np.ndarray):
            gripper_action = np.array(gripper_action)
        
        # Convert to delta action using the function from process_utils
        delta_action = eef_target_pose2action(source_pose, target_pose, controller, gripper_action)
        delta_actions.append(delta_action)
    
    # Convert to numpy array with float32 dtype
    delta_actions = np.array(delta_actions, dtype=np.float32)
    
    # Add a dummy action at the end to keep length equal to actions
    if len(delta_actions) > 0:
        dummy_action = np.zeros_like(delta_actions[0])
        last_gripper = delta_actions[-1][6:7]
        dummy_action[6:7] = last_gripper
        delta_actions = np.vstack([delta_actions, dummy_action])
    
    return delta_actions


def process_demo_file(input_path, in_place=True, output_path=None, use_target_chaining=False):
    """
    Process a single HDF5 demo file and add delta_actions.
    
    Args:
        input_path: Path to input HDF5 file
        in_place: If True, modify the file in-place. If False, create a copy.
        output_path: Path to output HDF5 file (only used if in_place=False)
        use_target_chaining: If True, use target chaining method. If False, use current method.
    """
    # Check if delta_actions already exists and delete it if present
    with h5py.File(input_path, 'a') as f_check:
        if 'delta_actions' in f_check['root']:
            print(f"Warning: delta_actions already exists in {input_path}, deleting and reprocessing...")
            del f_check['root']['delta_actions']
    
    # Read the demo file to compute delta actions
    with h5py.File(input_path, 'r') as f_check:
        # Get actions from the demo
        actions = f_check['root']['actions'][:]
        
        # Compute delta actions using selected method
        try:
            if use_target_chaining:
                delta_actions = convert_to_delta_actions_target_chaining(f_check['root'], actions)
                method_name = "target_chaining"
            else:
                delta_actions = convert_to_delta_actions(f_check['root'], actions)
                method_name = "current_state"
            print(f"Computed {len(delta_actions)} delta actions from {len(actions)} absolute actions using {method_name} method")
        except Exception as e:
            print(f"Error computing delta actions for {input_path}: {e}")
            raise
    
    if in_place:
        # Modify file in-place
        with h5py.File(input_path, 'a') as f:
            f['root'].create_dataset('delta_actions', data=delta_actions, compression='gzip')
            print(f"Added delta_actions with shape {delta_actions.shape} to {input_path}")
    else:
        # Create a copy with delta actions
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_delta{ext}"
        
        with h5py.File(input_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                # Copy all existing data
                for key in f_in.keys():
                    f_in.copy(key, f_out)
                
                # Add delta_actions to the 'data' group
                f_out['root'].create_dataset('delta_actions', data=delta_actions, compression='gzip')
                
                print(f"Added delta_actions with shape {delta_actions.shape} to {output_path}")


def process_folder(demos_folder, recursive=False, in_place=True, use_target_chaining=False):
    """
    Process all HDF5 files in a folder.
    
    Args:
        demos_folder: Path to folder containing HDF5 demos
        recursive: If True, search for HDF5 files recursively (default: False)
        in_place: If True, modify files in-place (default: True)
        use_target_chaining: If True, use target chaining method (default: False)
    """
    # Find all HDF5 files
    if recursive:
        hdf5_files = glob.glob(os.path.join(demos_folder, '**', '*.hdf5'), recursive=True)
    else:
        hdf5_files = glob.glob(os.path.join(demos_folder, '*.hdf5'))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {demos_folder}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process each file
    for input_file in tqdm(hdf5_files, desc="Processing demos"):
        try:
            process_demo_file(input_file, in_place=in_place, use_target_chaining=use_target_chaining)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Add delta_actions to HDF5 demo files (modifies files in-place by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'demos_folder',
        type=str,
        help='Path to folder containing HDF5 demo files'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively in subdirectories (default: only process top-level HDF5 files)'
    )
    parser.add_argument(
        '--single_file',
        type=str,
        default=None,
        help='Process a single HDF5 file instead of a folder'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be processed without actually modifying them'
    )
    parser.add_argument(
        '--target-chaining',
        action='store_true',
        help='Use target chaining method: compute delta from target[t] to target[t+1] (instead of current_pose[t] to target[t+1])'
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process a single file
        print(f"Processing single file: {args.single_file}")
        if args.dry_run:
            print(f"Would process: {args.single_file}")
        else:
            process_demo_file(args.single_file, in_place=True, use_target_chaining=args.target_chaining)
    else:
        # Process folder
        if not os.path.isdir(args.demos_folder):
            print(f"Error: {args.demos_folder} is not a valid directory")
            return
        
        if args.dry_run:
            # Just list files without processing
            if args.recursive:
                hdf5_files = glob.glob(os.path.join(args.demos_folder, '**', '*.hdf5'), recursive=True)
            else:
                hdf5_files = glob.glob(os.path.join(args.demos_folder, '*.hdf5'))
            
            print(f"Would process {len(hdf5_files)} files:")
            for f in hdf5_files:
                print(f"  {f}")
        else:
            process_folder(args.demos_folder, recursive=args.recursive, in_place=True, use_target_chaining=args.target_chaining)
    
    print("Done!")


if __name__ == "__main__":
    main()

