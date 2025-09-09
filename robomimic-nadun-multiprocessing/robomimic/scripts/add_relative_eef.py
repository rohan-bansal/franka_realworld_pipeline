import robosuite.utils.transform_utils as transform_utils
import robosuite as suite
import numpy as np
import h5py
import robomimic.utils.transform_utils as T
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import argparse

def add_relative_eef_transforms_to_dataset(demo_fn):

    demo_file = h5py.File(demo_fn, 'a')

    # create env to get base pose
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    env = EnvUtils.create_env_from_metadata(env_meta,
                                                render=False,
                                                use_image_obs=False)
    env = env.env

    # get base pose
    robot = env.robots[0]
    base_pos_in_world = robot.base_pos
    base_quat_in_world = robot.base_ori

    for demo in demo_file['data']:
        demo_data = demo_file[f'data/{demo}']
        obs = demo_data['obs']
        absolute_actions = demo_data['absolute_actions'][:]
        print(f"processing demo {demo}")

        # Retrieve the eef pose in world frame
        obs_eef_pos = obs['robot0_eef_pos'][:]
        obs_eef_quat = obs['robot0_eef_quat'][:]

        rel_pos = []
        rel_quat = []
        rel_axis_angle = []

        for pos, quat in zip(obs_eef_pos, obs_eef_quat):

            eef_pos_in_world = pos
            eef_quat_in_world = quat

            eef_pose_in_world = transform_utils.pose2mat((eef_pos_in_world, eef_quat_in_world))

            base_pose_in_world = transform_utils.pose2mat((base_pos_in_world, base_quat_in_world))
            world_pose_in_base = transform_utils.pose_inv(base_pose_in_world)

            eef_pose_in_base = transform_utils.pose_in_A_to_pose_in_B(eef_pose_in_world, world_pose_in_base)

            res_pos, res_quat = transform_utils.mat2pose(eef_pose_in_base)
            rel_pos.append(res_pos)
            rel_quat.append(res_quat)

            # Get relative rotation in axis angle format
            axis, angle = T.quat2axisangle(res_quat)
            axis_angle = T.axisangle2vec(axis, angle)
            rel_axis_angle.append(axis_angle)

        ### Cleanup everything first
        if "robot0_eef_pos_relative" in obs:
            del obs["robot0_eef_pos_relative"]
        if "robot0_eef_quat_relative" in obs:
            del obs["robot0_eef_quat_relative"]
        if "robot0_eef_axis_angle_relative" in obs:
            del obs["robot0_eef_axis_angle_relative"]

        obs.create_dataset("robot0_eef_pos_relative", data=np.array(rel_pos))
        obs.create_dataset("robot0_eef_quat_relative", data=np.array(rel_quat))
        obs.create_dataset("robot0_eef_axis_angle_relative", data=np.array(rel_axis_angle))

    demo_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to dataset",
    )
    args = parser.parse_args()

    add_relative_eef_transforms_to_dataset(args.dataset)
