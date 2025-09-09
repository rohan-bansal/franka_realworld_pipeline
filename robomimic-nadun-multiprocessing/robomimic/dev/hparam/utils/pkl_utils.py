"""
Utility functions for loading pickle files

This does not assume any specific structure of the pickle file.

Author: Wonsuhk Jung
"""

import os
import glob
import pickle
import numpy as np
from typing import Callable, List, Dict

def get_result_pkl_files(result_dir: str, filter: str = None) -> list:
    """Get all pkl result files from a directory.
    
    Args:
        result_dir: Path to the directory containing result files
        filter: Optional string to filter filenames that contain this string
        
    Returns:
        List of paths to pkl files
    """
    
    # Get all pkl files in directory and subdirectories
    pkl_pattern = os.path.join(result_dir, "**/*.pkl")
    pkl_files = glob.glob(pkl_pattern, recursive=True)
    
    # Filter files if filter string provided
    if filter is not None:
        pkl_files = [f for f in pkl_files if filter in os.path.basename(f)]
    
    # Sort files by modification time (newest first)
    pkl_files.sort(key=os.path.getmtime, reverse=True)
    
    return pkl_files

def load_result_pkl_file(pkl_file: str) -> dict:
    """Load a pkl file into a dictionary.
    
    Args:
        pkl_file: Path to the pkl file
        
    Returns:
        Dictionary containing the data from the pkl file
    """
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)
    
def load_demo_from_pkl(pkl_file: str, demo_idx: int) -> dict:
    """Load a demo from a pkl file.
    
    Args:
        pkl_file: Path to the pkl file
        
    Returns:
        Dictionary containing the demo data
    """
    data = load_result_pkl_file(pkl_file)
    return data["rollouts"][f"demo_{demo_idx}"]
    
def print_nested_structure(d, indent=0):
    """Recursively print nested dictionary structure with shapes.
    
    Args:
        d: Dictionary or nested structure to print
        indent: Current indentation level
    """
    prefix = "    " * indent
    
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_nested_structure(value, indent + 1)
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}{key}: (length: {len(value)})")
                if len(value) > 0:
                    # Print structure of first item as example
                    print(f"{prefix}    First item:")
                    print_nested_structure(value[0], indent + 2)
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: shape {value.shape}")
            else:
                print(f"{prefix}{key}: {type(value)}")
    elif isinstance(d, np.ndarray):
        print(f"{prefix}shape: {d.shape}")
    else:
        print(f"{prefix}type: {type(d)}")