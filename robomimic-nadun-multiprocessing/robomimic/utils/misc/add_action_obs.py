import h5py

dataset_path = "/home/mbronars/zhenyang/robomimic-nadun/datasets/lift/ph/all_obs_v141.hdf5"


with h5py.File(dataset_path, 'r+') as f:
    for demo in f['data'].keys():
        actions = f['data/{}/actions'.format(demo)][()]

        # add actions to obs
        f['data/{}/obs/actions'.format(demo)] = actions

