import h5py
import numpy as np
import argparse
from tqdm import tqdm

def main(dataset, waypoints_key, absolute_actions_key, output_key):
    f = h5py.File(dataset, "r+")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for i in tqdm(range(len(demos))):
        ep = demos[i]

        waypoints = f[f"data/{ep}/{waypoints_key}"][()]
        absolute_actions = f[f"data/{ep}/{absolute_actions_key}"][()]

        absolute_actions_filled = np.zeros_like(absolute_actions)
        
        # element 0 -> waypoint 0
        if len(waypoints) > 0:
            absolute_actions_filled[:waypoints[0]] = absolute_actions[waypoints[0]]
        
        # between waypoints
        for j in range(len(waypoints) - 1):
            start_idx = waypoints[j]
            end_idx = waypoints[j + 1]
            absolute_actions_filled[start_idx:end_idx] = absolute_actions[start_idx]
            
        # last waypoint to end
        if len(waypoints) > 0:
            last_idx = waypoints[-1]
            absolute_actions_filled[last_idx:] = absolute_actions[last_idx]

        # Store the filled actions back in the dataset
        if f"data/{ep}/{output_key}" in f:
            del f[f"data/{ep}/{output_key}"]
        f[f"data/{ep}/{output_key}"] = absolute_actions_filled

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    parser.add_argument(
        "--waypoints_key",
        type=str,
        default="waypoints_dp",
        help="key for waypoints in hdf5 dataset",
    )

    parser.add_argument(
        "--absolute_actions_key",
        type=str,
        default="absolute_actions",
        help="key for absolute actions in hdf5 dataset",
    )

    parser.add_argument(
        "--output_key",
        type=str,
        default="awe_actions",
        help="key for output actions in hdf5 dataset",
    )

    args = parser.parse_args()

    main(args.dataset, args.waypoints_key, args.absolute_actions_key, args.output_key)