import h5py
import os
import numpy as np
from pathlib import Path

def modify_gripper_states(hdf5_path):
    """
    Modify gripper_states in an HDF5 file from shape (x,) to (x, 2)
    where the second column is the negative of the first column.
    """
    try:
        with h5py.File(hdf5_path, 'r+') as f:
            # Check if the required path exists
            if 'root' in f and 'extra_states' in f['root'] and 'gripper_states' in f['root']['extra_states']:
                gripper_states = f['root']['extra_states']['gripper_states']
                
                # Get original data
                original_data = gripper_states[:]
                print(f"Original gripper_states shape: {original_data.shape}")
                # print(f"Original data: {original_data}")
                
                # Create new 2D array
                new_shape = (original_data.shape[0], 2)
                new_data = np.zeros(new_shape)
                new_data[:, 0] = original_data  # First column is original data
                new_data[:, 1] = -original_data  # Second column is negative of original data
                
                print(f"New gripper_states shape: {new_data.shape}")
                # print(f"New data:\n{new_data}")
                
                # Delete the old dataset and create new one
                del f['root']['extra_states']['gripper_states']
                f['root']['extra_states'].create_dataset('gripper_states', data=new_data)
                
                print(f"Successfully modified {hdf5_path}")
                return True
            else:
                print(f"Required path 'root/extra_states/gripper_states' not found in {hdf5_path}")
                return False
                
    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")
        return False

def process_folder(folder_path):
    """
    Process all HDF5 files in a folder.
    """
    folder_path = Path(folder_path)
    hdf5_files = list(folder_path.glob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {folder_path}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files in {folder_path}")
    
    success_count = 0
    for hdf5_file in hdf5_files:
        print(f"\nProcessing: {hdf5_file}")
        if modify_gripper_states(hdf5_file):
            success_count += 1
    
    print(f"\nProcessing complete! Successfully modified {success_count}/{len(hdf5_files)} files.")

if __name__ == "__main__":
    # Example usage - modify this path to your HDF5 folder
    hdf5_folder = "/srv/rl2-lab/flash8/rbansal66/ATM/data/pickup_block_atm_cotrack_128/pickup_block/WORKSPACE_SCENE2_pick_up_the_block_and_put_it_down_demo"  # Adjust this path as needed
    
    if os.path.exists(hdf5_folder):
        process_folder(hdf5_folder)
    else:
        print(f"Folder {hdf5_folder} does not exist. Please update the path in the script.")
