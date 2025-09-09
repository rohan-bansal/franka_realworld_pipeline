
import h5py
import numpy as np
import matplotlib.pyplot as plt

path = "/coc/flash7/zhenyang/data/robomimic-sim/square_low_dim_skill.hdf5"

def plot_initial_low_dim(path):
    pose_data = []
    orientation_data = []
    sim_states_data = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        # print(data.keys())
        # for all the demos
        for key in data.keys():
            eef_pos = data[key]["obs"]["robot0_eef_pos"][:]
            eef_quat = data[key]["obs"]["robot0_eef_quat"][:]
            sim_states = data[key]["states"][:]
            # print(f"eef_pos shape is {eef_pos.shape}")
            # print(f"eef_quat shape is {eef_quat.shape}")
            pose_data.append(eef_pos[0])
            orientation_data.append(eef_quat[0])
            sim_states_data.append(sim_states[0])

    pose_data = np.array(pose_data)
    orientation_data = np.array(orientation_data)
    sim_states_data = np.array(sim_states_data)
    np.save("result/pose_data.npy", pose_data)
    np.save("result/orientation_data.npy", orientation_data)
    np.save("result/sim_states_data.npy", sim_states_data)
    print(f"shape is {pose_data.shape}")
    print(f"shape is {orientation_data.shape}")
    print(f"shape is {sim_states_data.shape}")

    # plot the data
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pose_data[:, 0], pose_data[:, 1], pose_data[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Low Dimensional Data')
    plt.savefig("result/initial_low_dim.png")


if __name__ == "__main__":
    plot_initial_low_dim(path)