import h5py
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

from waypoint_extraction.extract_waypoints import (
    dp_waypoint_selection,
)
from multiprocessing import Pool
from robosuite.utils.transform_utils import quat2axisangle


def process_demo(demo_idx, dataset_path, err_threshold):
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    ep = demos[demo_idx]

    traj_len = f[f"data/{ep}/absolute_actions"][()].shape[0]

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
        err_threshold=err_threshold,
    )

    f.close()
    return ep, waypoints, len(waypoints), traj_len

def main(args):
    f = h5py.File(args.dataset, "r+")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    demos = [demos[i] for i in range(args.num_workers)] # TODO: for testing only
    
    f.close()
    print("# Demos: ", len(demos))

    pool = Pool(processes=args.num_workers)

    process_demo_partial = partial(process_demo, dataset_path=args.dataset, err_threshold=args.err_threshold)

    results = list(tqdm(
        pool.imap(process_demo_partial, range(len(demos))),
        total=len(demos),
        desc="Saving Waypoints"
    ))

    pool.close()
    pool.join()

    num_waypoints = []
    num_frames = []

    f = h5py.File(args.dataset, "r+")

    for ep, waypoints, n_waypoints, n_frames in results:
        num_waypoints.append(n_waypoints)
        num_frames.append(n_frames)

        # save waypoints
        try:
            f[f"data/{ep}/waypoints_dp{args.err_threshold}"] = waypoints
        except:
            del f[f"data/{ep}/waypoints_dp{args.err_threshold}"]
            f[f"data/{ep}/waypoints_dp{args.err_threshold}"] = waypoints

    f.close()
    print(
        f"Average number of waypoints: {np.mean(num_waypoints)}, average number of frames: {np.mean(num_frames)}, average waypoint ratio: {np.mean(num_frames) / np.mean(num_waypoints)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/mbronars/zhenyang/demos/speed_stacking_agentview_0116/with_agentview_demo.hdf5", #"/home/mbronars/zhenyang/demos/oven_bowl_demo_0115/_demo.hdf5", #"/home/mbronars/zhenyang/demos/stacking_cup_demo_0112/demo.hdf5",
        help="path to hdf5 dataset",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.01,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # number of worker processes
    parser.add_argument(
        "--num_workers",
        type=int,
        default=9,
        help="number of worker processes for parallel processing",
    )

    args = parser.parse_args()
    main(args)
