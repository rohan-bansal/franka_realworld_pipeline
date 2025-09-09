"""
Split the dataset to motion segments and skill segments
"""

import h5py
import numpy as np
import os


def split_dataset(dataset_path, save_dir, split_rule):
    """
    Split the dataset to motion segments and skill segments
    """
    if split_rule=="half":
        motion_segments = 0.5
        skill_segments = 0.5
    else:
        raise ValueError(f"Invalid split rule: {split_rule}")
    
    # Get filename without extension from dataset path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Load the dataset
    f = h5py.File(dataset_path, "r")
    
    demos = list(f["data"].keys())
    
    # Create new HDF5 files for motion and skill data
    f_motion = h5py.File(os.path.join(save_dir, f"{dataset_name}_motion.hdf5"), "w")
    f_skill = h5py.File(os.path.join(save_dir, f"{dataset_name}_skill.hdf5"), "w")
    
    # Create the main groups
    demos_data_motion = f_motion.create_group("data")
    demos_data_skill = f_skill.create_group("data")
    
    # Copy attributes if any
    for attr_name, attr_value in f["data"].attrs.items():
        demos_data_motion.attrs[attr_name] = attr_value
        demos_data_skill.attrs[attr_name] = attr_value
    
    # Copy mask data
    mask_motion = f_motion.create_group("mask")
    mask_skill = f_skill.create_group("mask")
    
    # Copy all datasets and attributes from the mask group
    for name, dataset in f["mask"].items():
        mask_motion.create_dataset(name, data=dataset[:])
        mask_skill.create_dataset(name, data=dataset[:])
        
    # Copy any attributes from the mask group
    for attr_name, attr_value in f["mask"].attrs.items():
        mask_motion.attrs[attr_name] = attr_value 
        mask_skill.attrs[attr_name] = attr_value

    ## update the demo data with different stage rules
    for demo in demos:
        ### Fields
        actions = f[f"data/{demo}/actions"][:] # array
        obs = f[f"data/{demo}/obs"].copy() # dict
        next_obs = f[f"data/{demo}/next_obs"].copy() # dict
        states = f[f"data/{demo}/states"][:] # array
        done = f[f"data/{demo}/done"][:] # array

        demo_len, act_dim = actions.shape
        skill_len = int(demo_len * skill_segments)
        motion_len = demo_len - skill_len

        skill_actions = actions[:motion_len]
        motion_actions = actions[motion_len:]
        skill_states = states[:motion_len]
        motion_states = states[motion_len:]
        skill_done = done[:motion_len]
        motion_done = done[motion_len:]

        skill_obs = {}
        motion_obs = {}
        for obs_key in obs.keys():
            skill_obs[obs_key] = obs[obs_key][:motion_len]
            motion_obs[obs_key] = obs[obs_key][motion_len:]

        skill_next_obs = {}
        motion_next_obs = {}
        for obs_key in next_obs.keys():
            skill_next_obs[obs_key] = next_obs[obs_key][:motion_len]
            motion_next_obs[obs_key] = next_obs[obs_key][motion_len:]

        ### Update the motion and skill datasets
        demos_data_motion[demo] = {
            "obs": motion_obs,
            "next_obs": motion_next_obs,
            "actions": motion_actions,
            "states": motion_states,
            "done": motion_done
        }

        demos_data_skill[demo] = {
            "obs": skill_obs,
            "next_obs": skill_next_obs,
            "actions": skill_actions,
            "states": skill_states,
            "done": skill_done
        }

    f_motion.close()
    f_skill.close()
    f.close()

    # ## Mask
    # skill_mask = f["mask"]
    # motion_mask = f["mask"]

    # ## Meta data
    # env_meta = f["data"].attrs["env_args"]
    # total = f["data"].attrs["total"]
    # shape_meta = f["data"].attrs["shape_args"]
    
if __name__ == "__main__":
    dataset_path = "/coc/flash7/zhenyang/data/robomimic-sim/square_low_dim.hdf5"
    save_dir = "/coc/flash7/zhenyang/data/robomimic-sim"
    split_rule = "half"
    split_dataset(dataset_path, save_dir, split_rule)