import h5py
import numpy as np
import nexusformat.nexus as nx


"""
For using reached joint positions in training, we need to add the joint positions to the dataset.
1. Action label: left shift the joint positions, so that using t+1 as action. Because position is current observation.
2. Extract gripper action from "actions". Use as a new separate action label (also left shifted).
"""

# demo_fn = "/srv/rl2-lab/flash7/zhenyang/data/pick_cube_demo_1108_keypoint.hdf5"
demo_fn = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo.hdf5"

demo_file = h5py.File(demo_fn, 'a')

dataset_grp = demo_file['data']

for demo in dataset_grp:
    print(f"processing demo: {demo}")
    demo = dataset_grp[demo]
    # del demo['shifted_joint_positions_as_action']
    # del demo['gripper_action']

    gripper_action = demo['actions'][1:, -1:] # keep dim as BxD
    gripper_action = np.concatenate((gripper_action, gripper_action[-1:]), axis=0) # repeat last position
    # print(f"action shape: {demo['actions'].shape}")

    shifted_joint_pos = demo['obs/joint_positions'][1:, :] # time shift t+1
    shifted_joint_pos = np.concatenate((shifted_joint_pos, shifted_joint_pos[-1:, :]), axis=0) # repeat last position
    # print(f"shifted joint pos shape: {shifted_joint_pos.shape}")
    
    demo.create_dataset('gripper_action', data=np.array(gripper_action))
    demo.create_dataset('shifted_joint_positions_as_action', data=np.array(shifted_joint_pos))

demo_file.close()

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)