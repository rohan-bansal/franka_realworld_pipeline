import h5py
import numpy as np
# from sklearn.cluster import DBSCAN
import argparse

def main(dataset, abs_action_key="absolute_actions", save_action_key="absolute_actions_with_precision", precision=False):
    # If precision is true, we will use the precision of the absolute action as the gripper action
    
    f = h5py.File(dataset, "r+")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for i in range(len(demos)):
        ep = demos[i]
    
        absolute_actions = f[f"data/{ep}/{abs_action_key}"][()]
        actions_bkp = f[f"data/{ep}/absolute_actions"][()]
        gripper_actions = actions_bkp[:, -1]

        if precision:
            abs_actions_with_precision = f[f"data/{ep}/absolute_commanded_actions_with_precision"][()]
            precision_label = abs_actions_with_precision[:, -1:]

        # append the gripper action if we are not inputting absolute actions
        if precision:
            abs_reached_pose = np.hstack((absolute_actions, gripper_actions[:, None], precision_label))
        else:
            abs_reached_pose = np.hstack((absolute_actions, gripper_actions[:, None]))
        # Important: perform time shift, because this is reached actions

        abs_actions = abs_reached_pose[1:, :] # time shift t+1
        abs_actions = np.concatenate((abs_actions, abs_reached_pose[-1:, :]), axis=0) # repeat last position

        if save_action_key in f[f"data/{ep}"].keys():
            del f[f"data/{ep}/{save_action_key}"]
        f[f"data/{ep}/{save_action_key}"] = abs_actions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=str,
        required=True,
        # default="/home/mbronars/zhenyang/demos/speed_stacking_agentview_0116/with_agentview_demo.hdf5", #"/home/mbronars/zhenyang/demos/oven_bowl_demo_0115/_demo.hdf5", #"/home/mbronars/zhenyang/demos/stacking_cup_demo_0112/demo.hdf5",
        help="path to hdf5 dataset",
    )
    args = parser.parse_args()

    # whether add precision at the same time
    # can also be joint_pos
    # main(args.demo, abs_action_key="obs/eef_pose", save_action_key="absolute_reached_actions_eefpose_shift_precision", precision=True)
    main(args.demo, abs_action_key="obs/joint_positions", save_action_key="abs_reached_actions_joint_pos_shift_precision", precision=True)