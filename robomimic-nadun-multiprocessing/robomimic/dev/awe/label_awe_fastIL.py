import h5py
import numpy as np
from sklearn.cluster import DBSCAN
import argparse

def main(dataset, method, eps, min_samples, abs_action_key="absolute_actions", save_action_key="absolute_actions_with_precision"):
    f = h5py.File(dataset, "r+")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for i in range(len(demos)):
        ep = demos[i]

        # fix for dataset with no precisions
        # absolute_reached_actions_with_precision = f[f"data/{ep}/absolute_commanded_actions_with_precision"][()]
        # precisions = absolute_reached_actions_with_precision[:, -1]

        absolute_actions = f[f"data/{ep}/{abs_action_key}"][()]
        actions_bkp = f[f"data/{ep}/absolute_actions"][()]
        gripper_actions = actions_bkp[:, -1]
    
        waypoints = f[f"data/{ep}/waypoints_dp"][()]
        waypoints_pos = absolute_actions[waypoints][:, :3]
        waypoints_ori = absolute_actions[waypoints][:, 3:6]

        if method == "CLUSTERING":

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(waypoints_pos)
            waypoint_labels = clustering.labels_

            precisions = np.zeros(len(absolute_actions))

            # Set precision values at waypoint indices
            precisions[waypoints] = np.array([1 if label >= 0 else 0 for label in waypoint_labels])

            # Fill gaps
            for j in range(len(waypoint_labels)-1):
                if waypoint_labels[j] >= 0 and waypoint_labels[j+1] >= 0:
                    start_idx = waypoints[j]
                    end_idx = waypoints[j+1]
                    precisions[start_idx:end_idx+1] = 1

            print(demos[i], precisions)

        elif method == "INV_PROP_DIST":
            precisions = np.zeros(len(absolute_actions))
            distances = np.linalg.norm(np.diff(waypoints_pos, axis=0), axis=1)

            # calculate precision values inversely proportional to distances
            waypoint_precisions = 1 / (distances + 1e-8)

            # normalize
            waypoint_precisions = (waypoint_precisions - waypoint_precisions.min()) / (waypoint_precisions.max() - waypoint_precisions.min())

            # Fill gaps
            for j in range(len(waypoint_precisions)):
                start_idx = waypoints[j]
                end_idx = waypoints[j+1] if j < len(waypoints)-1 else waypoints[j]
                precisions[start_idx:end_idx+1] = waypoint_precisions[j]

            # print(demos[i], precisions)
            
        if abs_action_key == "obs/eef_pose" or abs_action_key == "obs/joint_positions":
            # append the gripper action if we are not inputting absolute actions
            abs_reached_pose_with_precision = np.hstack((absolute_actions, gripper_actions[:, None], precisions[:, None]))
            # Important: perform time shift, because this is reached actions
            abs_actions_with_precision = abs_reached_pose_with_precision[1:, :] # time shift t+1
            abs_actions_with_precision = np.concatenate((abs_reached_pose_with_precision, abs_reached_pose_with_precision[-1:, :]), axis=0) # repeat last position

        elif abs_action_key == "absolute_actions":
            abs_actions_with_precision = np.hstack((absolute_actions, precisions[:, None]))
        else:
            raise ValueError("Check abs action key")

        print(f"shape of abs actions with precision {abs_actions_with_precision.shape}")
        if save_action_key in f[f"data/{ep}"].keys():
            del f[f"data/{ep}/{save_action_key}"]
        f[f"data/{ep}/{save_action_key}"] = abs_actions_with_precision

        if "precisions" in f[f"data/{ep}"].keys():
            del f[f"data/{ep}/precisions"]
        f.create_dataset(f"data/{ep}/precisions", data=precisions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=str,
        required=True,
        # default="/home/mbronars/zhenyang/demos/speed_stacking_agentview_0116/with_agentview_demo.hdf5", #"/home/mbronars/zhenyang/demos/oven_bowl_demo_0115/_demo.hdf5", #"/home/mbronars/zhenyang/demos/stacking_cup_demo_0112/demo.hdf5",
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--reached_act",
        action='store_true',
        help="Using reached act as action label. By default we use absolute actions",
    )
    args = parser.parse_args()

    # can be CLUSTERING (int 0 or 1) or INV_PROP_DIST (float in range [0,1])
    method = "CLUSTERING"

    # CLUSTERING ONLY for default
    eps = 0.025
    min_samples = 3

    eps = 0.025
    min_samples = 8

    eps = 0.04
    min_samples = 3

    # main(args.demo, method, eps, min_samples, abs_action_key="obs/joint_positions", save_action_key="abs_joint_reached_actions_with_precision") 
    if args.reached_act == True:
        main(args.demo, method, eps, min_samples, abs_action_key="obs/eef_pose", save_action_key="absolute_reached_actions_with_precision")
    else:
        main(args.demo, method, eps, min_samples, abs_action_key="absolute_actions", save_action_key="absolute_commanded_actions_with_precision")