import h5py
import matplotlib.pyplot as plt
import nexusformat.nexus as nx
import robomimic.dev.dev_utils as dev_utils


POINT_TIME = 0.05
PLOT_JOINT = 0
PLOT_CARTESIAN_POSE_FEATURE = 2 # which part of cartesian pose

CARTESIAN_FEATURE_TO_LABEL = {
    0: "x position",
    1: "y position",
    2: "z position"
}


def plot_joint_position_actions(demo):

    demo_obs = demo['obs']
    demo_length = demo.attrs['num_samples']
    joint_position_actions = demo["joint_position_actions"][:]
    X = []
    y = []

    for i in range(joint_position_actions.shape[0]):
        X.append(i)
        y.append(joint_position_actions[i, PLOT_JOINT])


    plt.plot(X, y)


def plot_ee_pose_from_joint_position_actions(demo):
    fk_solver = dev_utils.FK_Solver()

    demo_length = demo.attrs['num_samples']
    joint_position_actions = demo["joint_position_actions"][:]
    X = []
    y = []

    for i in range(joint_position_actions.shape[0]):
        X.append(i)
        q = joint_position_actions[i][:-1]
        ee_pose = fk_solver.joints_to_ee_pose(q)
        y.append(ee_pose[PLOT_CARTESIAN_POSE_FEATURE])

    plt.plot(X, y, label=f"{CARTESIAN_FEATURE_TO_LABEL[PLOT_CARTESIAN_POSE_FEATURE]} from fk on joint action")


def plot_ee_pose(demo):
    ee_pose = demo['obs/ee_pose'][:]
    demo_length = demo.attrs['num_samples']

    X = []
    y = []
    for i in range(demo_length):
        X.append(i)
        y.append(ee_pose[i, PLOT_CARTESIAN_POSE_FEATURE])

    plt.plot(X, y, label=f"{CARTESIAN_FEATURE_TO_LABEL[PLOT_CARTESIAN_POSE_FEATURE]} of recorded ee pose")


### Setup the demo
demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_10_30.hdf5"
demo_file = h5py.File(demo_fn)['data']
demo = demo_file['demo_0']
plot_ee_pose_from_joint_position_actions(demo)
plot_ee_pose(demo)
plt.legend()
plt.show()

