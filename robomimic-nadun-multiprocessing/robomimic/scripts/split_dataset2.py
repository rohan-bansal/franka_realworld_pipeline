"""
Split the dataset to motion segments and skill segments
"""

import h5py
import numpy as np
import os


def split_dataset(dataset_path, save_dir):
    """
    Split the dataset to motion segments and skill segments
    """

    # Get filename without extension from dataset path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Create a copy of dataset with skill appendix
    skill_dataset_name = f"{dataset_name}_skill"
    skill_dataset_path = os.path.join(save_dir, f"{skill_dataset_name}.hdf5")
    
    # Copy the original dataset to the new location
    if os.path.exists(skill_dataset_path):
        os.remove(skill_dataset_path)
    os.system(f"cp {dataset_path} {skill_dataset_path}")
    
    # Load the dataset
    f = h5py.File(skill_dataset_path, "r+")
    
    demos = list(f["data"].keys())

    ## update the demo data with different stage rules
    for demo in demos:
        demo_len = f[f"data/{demo}"].attrs["num_samples"]
        skill_len = int(demo_len * 0.5) # TODO: tune for different tasks
        motion_len = demo_len - skill_len
        
        # Update the attributes first
        f[f"data/{demo}"].attrs["num_samples"] = skill_len

        # Handle arrays with proper HDF5 operations
        for field in ["actions", "states", "dones"]:
            data = f[f"data/{demo}/{field}"][:]
            new_data = data[motion_len:]
            del f[f"data/{demo}/{field}"]  # Delete old dataset
            f[f"data/{demo}"].create_dataset(field, data=new_data)  # Create new dataset

        # Handle nested observation dictionaries
        for obs_type in ["obs", "next_obs"]:
            obs_group = f[f"data/{demo}/{obs_type}"]
            for obs_key in obs_group.keys():
                data = obs_group[obs_key][:]
                new_data = data[motion_len:]
                del obs_group[obs_key]  # Delete old dataset
                obs_group.create_dataset(obs_key, data=new_data)  # Create new dataset

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
    split_dataset(dataset_path, save_dir)