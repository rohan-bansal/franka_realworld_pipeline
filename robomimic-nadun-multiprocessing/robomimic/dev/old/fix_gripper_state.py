import h5py
import numpy as np
import nexusformat.nexus as nx

# demo_fn = "/srv/rl2-lab/flash7/zhenyang/data/pick_cube_demo_1108_keypoint.hdf5"
# demo_fn = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo.hdf5"
demo_fn = "/home/mbronars/zhenyang/demos/wiping_board_1227/2024-12-27_demo_128x128.hdf5"

demo_file = h5py.File(demo_fn, 'a')

dataset_grp = demo_file['data']

for demo in dataset_grp:
    print(f"processing demo: {demo}")
    demo = dataset_grp[demo]
    gripper_state = demo['obs/gripper_position'][:]
    gripper_pos = gripper_state[..., np.newaxis]
    if "gripper_position" in demo['obs']:
        del demo['obs/gripper_position']
    demo.create_dataset('obs/gripper_position', data=np.array(gripper_pos))

demo_file.close()

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)