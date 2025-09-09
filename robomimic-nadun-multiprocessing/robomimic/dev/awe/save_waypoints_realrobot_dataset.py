import h5py
import argparse
import numpy as np
from tqdm import tqdm

from waypoint_extraction.extract_waypoints import (
    dp_waypoint_selection,
)
from robosuite.utils.transform_utils import quat2axisangle

num_waypoints = []
num_frames = []


def main(args):

    # load the dataset
    f = h5py.File(args.dataset, "r+")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    print("# Demos: ", len(demos))

    for idx in tqdm(range(0, len(demos)), desc="Saving Waypoints"):
        ep = demos[idx]

        traj_len = f[f"data/{ep}/actions"][()].shape[0]

        eef_pos = f[f"data/{ep}/obs/eef_pos"][()]
        eef_pose = f[f"data/{ep}/obs/eef_pose"][()]
        eef_quat = f[f"data/{ep}/obs/eef_quat"][()]
        joint_pos = f[f"data/{ep}/obs/joint_positions"][()]
        gt_states = []
        for i in range(traj_len):
            gt_states.append(
                dict(
                    robot0_eef_pos=eef_pos[i],
                    robot0_eef_quat=eef_quat[i],
                    robot0_joint_pos=joint_pos[i],
                )
            )


        actions = f[f"data/{ep}/obs/eef_pose"][()]

        # overwrite given axis angles because they seem to be faulty for now
        if actions.shape[1] == 6:
            actions = np.hstack((eef_pos, eef_quat))

        # Convert quaternion data in actions to axis angles and create a new dataset
        axis_angles = np.array([quat2axisangle(quat) for quat in actions[:, 3:]])
        # fill gripper value with dummy 0s
        actions = np.hstack((actions[:, :3], axis_angles, np.zeros((actions.shape[0], 1))))

        waypoints = dp_waypoint_selection(
            actions=actions,
            gt_states=gt_states,
            err_threshold=args.err_threshold,
        )

        num_waypoints.append(len(waypoints))
        num_frames.append(traj_len)

        # save waypoints
        try:
            f[f"data/{ep}/waypoints_dp"] = waypoints
        except:
            del f[f"data/{ep}/waypoints_dp"]
            f[f"data/{ep}/waypoints_dp"] = waypoints

    f.close()
    print(
        f"Average number of waypoints: {np.mean(num_waypoints)}, average number of frames: {np.mean(num_frames)}, average waypoint ratio: {np.mean(num_frames) / np.mean(num_waypoints)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="robomimic/datasets/lift/ph/low_dim.hdf5",
        help="path to hdf5 dataset",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.01,
        help="(optional) error threshold for reconstructing the trajectory",
    )


    args = parser.parse_args()
    main(args)
