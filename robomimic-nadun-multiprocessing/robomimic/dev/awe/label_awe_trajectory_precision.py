import h5py
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def main(dataset, method, eps, min_samples, abs_action_key="absolute_actions"):
    f = h5py.File(dataset, "r+")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for i in tqdm(range(len(demos))):
        ep = demos[i]
    
        waypoints = f[f"data/{ep}/waypoints_dp"][()]
        absolute_actions = f[f"data/{ep}/{abs_action_key}"][()]

        # get x,y,z positions and orientations of waypoints
        waypoints_pos = absolute_actions[waypoints][:, :3]
        waypoints_ori = absolute_actions[waypoints][:, 3:6]


        if method == "CLUSTERING":

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(waypoints_pos)
            waypoint_labels = clustering.labels_

            precisions = np.zeros(len(absolute_actions))

            # Set precision values at waypoint indices
            precisions[waypoints] = np.array([1 if label >= 0 else 0 for label in waypoint_labels])

            # Fill gaps
            for i in range(len(waypoint_labels)-1):
                if waypoint_labels[i] >= 0 and waypoint_labels[i+1] >= 0:
                    start_idx = waypoints[i]
                    end_idx = waypoints[i+1]
                    precisions[start_idx:end_idx+1] = 1

            # print(demos[i], precisions)

        elif method == "INV_PROP_DIST":
            precisions = np.zeros(len(absolute_actions))
            distances = np.linalg.norm(np.diff(waypoints_pos, axis=0), axis=1)

            # calculate precision values inversely proportional to distances
            waypoint_precisions = 1 / (distances + 1e-8)

            # normalize
            waypoint_precisions = (waypoint_precisions - waypoint_precisions.min()) / (waypoint_precisions.max() - waypoint_precisions.min())

            # Fill gaps
            for i in range(len(waypoint_precisions)):
                start_idx = waypoints[i]
                end_idx = waypoints[i+1] if i < len(waypoints)-1 else waypoints[i]
                precisions[start_idx:end_idx+1] = waypoint_precisions[i]

            print(demos[i], precisions)
            
        abs_actions_with_precision = np.hstack((absolute_actions, precisions[:, None]))
        # print(abs_actions_with_precision)
        if "absolute_actions_with_precision" in f[f"data/{ep}"].keys():
            del f[f"data/{ep}/absolute_actions_with_precision"]
        f[f"data/{ep}/absolute_actions_with_precision"] = abs_actions_with_precision


        # if "commanded_absolute_actions_with_precision" in f[f"data/{ep}"].keys():
        #     del f[f"data/{ep}/commanded_absolute_actions_with_precision"]
        # f[f"data/{ep}/commanded_absolute_actions_with_precision"] = abs_actions_with_precision

        if "precisions" in f[f"data/{ep}"].keys():
            del f[f"data/{ep}/precisions"]
        f.create_dataset(f"data/{ep}/precisions", data=precisions)



if __name__ == '__main__':

    # dataset = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/can/ph/all_obs_v141.hdf5"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    args = parser.parse_args()

    # can be CLUSTERING (int 0 or 1) or INV_PROP_DIST (float in range [0,1])
    method = "CLUSTERING"

    # CLUSTERING ONLY
    eps = 0.025
    min_samples = 3


    main(args.dataset, method, eps, min_samples)