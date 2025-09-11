import robomimic.utils.transform_utils as trans_mimic
import numpy as np
import h5py
import os
import glob

def normalize_euler_vector(matrix):
    matrix = np.array(matrix)
    flip_occurred = False
    
    for i in range(1, len(matrix)):
        euler_vec_prev = matrix[i - 1, 3:6]
        euler_vec_curr = matrix[i, 3:6]

        # Get axis, angle from euler vectors
        axis_curr, angle_curr = trans_mimic.vec2axisangle(euler_vec_curr)

        # Check if axes are nearly opposite
        dot_product = np.dot(euler_vec_prev, euler_vec_curr)
        if dot_product < -0.99:
            # Flip the current axis and adjust the angle
            new_axis_curr = -axis_curr
            angle_curr = 2 * np.pi - angle_curr

            euler_vec_curr = trans_mimic.axisangle2vec(new_axis_curr, angle_curr)
            matrix[i, 3:6] = euler_vec_curr
            flip_occurred = True

    return matrix, flip_occurred

def process_hdf5_folder(folder_path):
    """Process all HDF5 files in the given folder."""
    hdf5_files = glob.glob(os.path.join(folder_path, "*.hdf5"))
    
    for file_path in hdf5_files:
        print(f"Processing: {file_path}")
        
        with h5py.File(file_path, 'r+') as f:
            if 'root' in f and 'actions' in f['root']:
                actions = f['root']['actions'][:]
                normalized_actions, flip_occurred = normalize_euler_vector(actions)
                
                # Update the dataset
                f['root']['actions'][:] = normalized_actions
                
                if flip_occurred:
                    print(f"  -> Flip occurred in trajectory: {os.path.basename(file_path)}")
            else:
                print(f"  -> Warning: 'root/actions' not found in {file_path}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing HDF5 files: ")
    process_hdf5_folder(folder_path)

