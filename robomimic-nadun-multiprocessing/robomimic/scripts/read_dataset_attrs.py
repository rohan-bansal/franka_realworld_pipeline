import h5py

dataset_path = "/coc/flash7/zhenyang/data/robomimic-sim/square_low_dim.hdf5"

f = h5py.File(dataset_path, "r")


print(f["data"].attrs["env_args"])
print(f["data"].attrs["total"])

# print(f["data"].keys())
# print(f["data/square_osc_low_dim/actions"].keys())


