import nexusformat.nexus as nx
import h5py
import numpy as np

demo_fn = "/home/mbronars/zhenyang/demos/wiping_board_1227/2024-12-27_demo.hdf5"


demo_file = h5py.File(demo_fn, 'a')

data = demo_file['data']

for demo_num in data:
    demo = data[demo_num]
    all_obs = demo['obs']
    for obs_mod in all_obs:
        obs = all_obs[obs_mod][:]
        if obs.ndim == 3:
            obs = np.squeeze(obs, axis=1)
        del demo[f'obs/{obs_mod}']
        demo.create_dataset(f"obs/{obs_mod}", data=obs)


demo_file = nx.nxload(demo_fn)
print(demo_file.tree)