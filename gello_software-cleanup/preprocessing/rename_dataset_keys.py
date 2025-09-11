#!/usr/bin/env python3
"""
Script to copy an HDF5 dataset with renamed keys.
Takes a merged HDF5 file and creates a copy with specific key transformations:
- Remove "actions" and "delta_actions"
- Rename "absolute_actions" to "actions" 
- Rename keys beginning with "wrist" to begin with "hand_in_eye"
- Rename "gripper_position" to "gripper_states"
- Rename "joint_positions" to "joint_states"
- Rename "eef_pos" to "ee_pos"
- Rename "eef_quat" to "ee_ori"
- Rename "eef_pose" to "ee_states"
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_new_key_name(original_key: str) -> str:
    """
    Transform key names according to the specified rules.
    
    Args:
        original_key: Original key name
        
    Returns:
        New key name, or None if key should be removed
    """
    # Keys to remove
    if original_key in ['actions', 'delta_actions']:
        return None
    
    # Rename absolute_actions to actions
    if original_key == 'absolute_actions':
        return 'actions'
    
    # Rename keys beginning with "wrist" to begin with "hand_in_eye"
    if original_key.startswith('wrist'):
        return original_key.replace('wrist', 'hand_in_eye', 1)
    
    # Specific renames
    rename_map = {
        'gripper_position': 'gripper_states',
        'joint_positions': 'joint_states',
        'eef_pos': 'ee_pos',
        'eef_quat': 'ee_ori',
        'eef_pose': 'ee_states'
    }
    
    return rename_map.get(original_key, original_key)


def copy_group_with_renamed_keys(source_group, target_group):
    """
    Recursively copy HDF5 group contents with key renaming.
    
    Args:
        source_group: Source HDF5 group
        target_group: Target HDF5 group
    """
    for key in source_group.keys():
        new_key = get_new_key_name(key)
        
        # Skip keys that should be removed
        if new_key is None:
            logger.info(f"Removing key: {key}")
            continue
            
        if new_key != key:
            logger.info(f"Renaming key: {key} -> {new_key}")
        
        item = source_group[key]
        
        if isinstance(item, h5py.Group):
            # Create new group and recursively copy contents
            new_group = target_group.create_group(new_key)
            copy_group_with_renamed_keys(item, new_group)
        else:
            # Copy dataset
            target_group.create_dataset(new_key, data=item[:])


def copy_dataset_with_renamed_keys(input_file: str, output_file: str):
    """
    Copy HDF5 dataset with renamed keys.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Path to output HDF5 file with renamed keys
    """
    logger.info(f"Processing input file: {input_file}")
    
    with h5py.File(input_file, 'r') as input_f:
        # Check if data group exists
        if 'data' not in input_f:
            raise ValueError(f"No 'data' group found in {input_file}")
        
        data_group = input_f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        
        if not demo_keys:
            raise ValueError(f"No demo groups found in {input_file}")
        
        logger.info(f"Found {len(demo_keys)} demos to process")
        
        # Create output file
        with h5py.File(output_file, 'w') as output_f:
            output_data_group = output_f.create_group('data')
            
            for demo_key in demo_keys:
                logger.info(f"Processing {demo_key}")
                
                source_demo = data_group[demo_key]
                target_demo = output_data_group.create_group(demo_key)
                
                # Copy demo contents with renamed keys
                copy_group_with_renamed_keys(source_demo, target_demo)
    
    logger.info(f"Successfully created renamed dataset: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Copy HDF5 dataset with renamed keys according to specified transformation rules'
    )
    parser.add_argument('input_file', help='Path to input HDF5 file')
    parser.add_argument('output_file', help='Path to output HDF5 file with renamed keys')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Copy dataset with renamed keys
    copy_dataset_with_renamed_keys(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
