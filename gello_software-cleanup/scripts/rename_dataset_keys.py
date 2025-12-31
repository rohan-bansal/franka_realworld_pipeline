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
import json
import h5py
import numpy as np
import argparse
from pathlib import Path
import logging

import torch
from transformers import AutoTokenizer, AutoModel

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


def copy_group_with_renamed_keys(source_group, target_group, new_task_emb=None):
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
            copy_group_with_renamed_keys(item, new_group, new_task_emb=new_task_emb)
        else:
            data = item[:]

            # Optionally override obs/task_emb with a newly computed task embedding
            if (
                new_task_emb is not None
                and new_key == "task_emb"
                and source_group.name.endswith("/obs")
            ):
                logger.info(
                    f"Replacing task_emb at group {source_group.name} "
                    f"with newly computed embedding of shape {new_task_emb.shape}"
                )
                # Match original shape: either [D] or [T, D]
                if data.ndim == 1:
                    out_data = new_task_emb.astype(data.dtype)
                elif data.ndim == 2:
                    T = data.shape[0]
                    out_data = np.tile(new_task_emb.astype(data.dtype)[None, :], (T, 1))
                else:
                    logger.warning(
                        f"Unexpected task_emb shape {data.shape}; keeping original."
                    )
                    out_data = data

                target_group.create_dataset(new_key, data=out_data)
            else:
                # Copy dataset as-is
                target_group.create_dataset(new_key, data=data)


def copy_dataset_with_renamed_keys(input_file: str, output_file: str, new_task_emb=None):
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

                # Copy demo contents with renamed keys and optional new task embedding
                copy_group_with_renamed_keys(source_demo, target_demo, new_task_emb=new_task_emb)
    
    logger.info(f"Successfully created renamed dataset: {output_file}")


def get_task_embs(descriptions):
    """
    Compute BERT embeddings for task descriptions.
    Mirrors the logic in preprocess_realworld_cotrack.get_task_embs.
    """
    tz = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")

    tokens = tz(
        text=descriptions,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(tokens["input_ids"], tokens["attention_mask"])
    task_embs = outputs["pooler_output"].detach().cpu().numpy()
    return task_embs


def compute_task_emb_map(task_name_map):
    """
    Given a mapping {file_name: task_name_string}, compute a
    mapping {file_name: embedding_vector}.
    """
    # Unique list of task description strings
    descriptions = sorted(set(task_name_map.values()))
    if not descriptions:
        return {}

    logger.info(f"Computing BERT embeddings for {len(descriptions)} unique task names")
    embs = get_task_embs(descriptions)
    desc_to_emb = {descriptions[i]: embs[i] for i in range(len(descriptions))}

    file_to_emb = {}
    for fname, task_name in task_name_map.items():
        if task_name not in desc_to_emb:
            logger.warning(f"No embedding found for task name: {task_name}")
            continue
        file_to_emb[fname] = desc_to_emb[task_name]

    return file_to_emb


def normalize_task_name_map_keys(task_name_map: dict) -> dict:
    """
    Normalize task_name_map keys to support both:
      - keys without extension (e.g. "place_blue_cup_top_of_drawer_demo")
      - keys with ".hdf5" extension (e.g. "place_blue_cup_top_of_drawer_demo.hdf5")

    Returns a new dict containing the original entries plus extension-augmented
    versions (when missing).
    """
    out = dict(task_name_map)
    for k, v in task_name_map.items():
        if isinstance(k, str) and not k.endswith(".hdf5"):
            out[f"{k}.hdf5"] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy HDF5 dataset with renamed keys according to specified rules.\n"
            "Supports single-file mode (input_file, output_file) and batch mode over a directory."
        )
    )
    parser.add_argument('input_file', nargs='?', help='Path to input HDF5 file (single-file mode)')
    parser.add_argument('output_file', nargs='?', help='Path to output HDF5 file with renamed keys (single-file mode)')

    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Directory containing merged HDF5 files to process in batch mode'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for renamed HDF5 files in batch mode'
    )
    parser.add_argument(
        '--task_map',
        type=str,
        default=None,
        help='Path to JSON mapping {file_name: task_name_string} for computing new task embeddings'
    )

    args = parser.parse_args()

    # Batch mode: process all files in base_dir
    if args.base_dir is not None:
        if args.output_dir is None:
            logger.error("When using --base_dir, you must also provide --output_dir")
            return
        if args.task_map is None:
            logger.error("When using --base_dir, you must provide --task_map JSON with file_name→task_name mapping")
            return

        if not os.path.exists(args.base_dir):
            logger.error(f"base_dir does not exist: {args.base_dir}")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # Load filename → task_name mapping
        with open(args.task_map, 'r') as f:
            task_name_map = json.load(f)
        task_name_map = normalize_task_name_map_keys(task_name_map)

        # Compute embeddings for each file
        file_to_emb = compute_task_emb_map(task_name_map)

        # Iterate over .hdf5 files in base_dir
        all_files = [fn for fn in os.listdir(args.base_dir) if fn.endswith(".hdf5")]
        if not all_files:
            logger.error(f"No .hdf5 files found in base_dir: {args.base_dir}")
            return

        logger.info(f"Found {len(all_files)} .hdf5 files in {args.base_dir}")

        for fname in all_files:
            input_path = os.path.join(args.base_dir, fname)
            output_path = os.path.join(args.output_dir, fname)

            if os.path.exists(output_path):
                logger.info(f"Skipping (already exists): {output_path}")
                continue

            new_task_emb = None
            if fname in file_to_emb:
                new_task_emb = file_to_emb[fname]
                logger.info(
                    f"Using new task embedding for {fname} with shape {new_task_emb.shape}"
                )
            else:
                logger.warning(
                    f"No task embedding found for {fname} in task_map; keeping existing obs/task_emb if present."
                )

            # Copy dataset with renamed keys and optional new task embedding
            copy_dataset_with_renamed_keys(input_path, output_path, new_task_emb=new_task_emb)

        logger.info(f"Batch processing complete. Outputs saved to {args.output_dir}")
        return

    # Single-file mode (original behavior)
    if not args.input_file or not args.output_file:
        logger.error("In single-file mode you must provide input_file and output_file")
        return

    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(args.output_file):
        logger.info(f"Skipping (already exists): {args.output_file}")
        return

    # Single-file copy without changing task_emb (unless you extend to pass one)
    copy_dataset_with_renamed_keys(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
