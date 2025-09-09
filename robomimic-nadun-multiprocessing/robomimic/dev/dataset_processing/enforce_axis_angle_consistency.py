import h5py
import numpy as np
import argparse
from robomimic.utils import transform_utils as T


def normalize_euler_vector(matrix):

    matrix = np.array(matrix)
    print(f"check the sign of the first one {matrix[0, 3:6]}")
    for i in range(1, len(matrix)):
        euler_vec_prev = matrix[i - 1, 3:6]
        euler_vec_curr = matrix[i, 3:6]

        # Get axis, angle from euler vectors
        axis_curr, angle_curr = T.vec2axisangle(euler_vec_curr)

        # Check if axes are nearly opposite
        dot_product = np.dot(euler_vec_prev, euler_vec_curr)
        if dot_product < -0.99:
            # Flip the current axis and adjust the angle
            new_axis_curr = -axis_curr
            angle_curr = 2 * np.pi - angle_curr

            euler_vec_curr = T.axisangle2vec(new_axis_curr, angle_curr)
            matrix[i, 3:6] = euler_vec_curr

    return matrix

def process_demo_file(demo_path, action_keys):
    """
    Process demonstration file and normalize action vectors.
    
    Args:
        demo_path (str): Path to demonstration HDF5 file
    """
    demo_file = h5py.File(demo_path, 'a')
    data = demo_file['data']
    counter = 0

    for demo_num in data:
        print(f"Processing demo: {counter}")
        counter += 1

        demo = data[demo_num]

        absolute_actions = demo[action_keys][:]
        consistent_absolute_actions = normalize_euler_vector(absolute_actions)

        if f"consistent_{action_keys}" in demo:
            del demo[f"consistent_{action_keys}"]

        demo.create_dataset(f"consistent_{action_keys}", data=np.array(consistent_absolute_actions))

    demo_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                       default="/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5",
                       help="Path to demonstration HDF5 file")
    
    parser.add_argument('--action_keys', type=str, 
                       default="absolute_actions", # could be obs/eef_pose
                       help="Path to demonstration HDF5 file")
    args = parser.parse_args()
    
    process_demo_file(args.dataset, args.action_keys)

