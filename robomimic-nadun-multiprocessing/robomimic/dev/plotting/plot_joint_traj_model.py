from os import replace

import nexusformat.nexus as nx
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pickle
from bisect import bisect_left

import robomimic.dev.dev_utils as dev_utils
from robomimic.utils.ros_utils import ros_time_to_float
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from networkx.algorithms.bipartite import color

# CONSTANTS
PLOT_JOINT = 0
PLOT_CARTESIAN_FEATURE = 2 # what part of the ee pose to plot
JOINT_TO_NAME = {
    0: "elbow_joint",
    1: "shoulder_lift_joint",
    2: "shoulder_pan_joint",
    3: "wrist_1_joint",
    4: "wrist_2_joint",
    5: "wrist_3_joint"
}

ANGLE_MULTIPLIER = 180/np.pi
PLOT_SAMPLES = 5
# Y_LIM = (-1.90*ANGLE_MULTIPLIER, -1.10*ANGLE_MULTIPLIER)
Y_LIM = (0.20, 0.70)


# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_single.hdf5"
# demo_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_joint_position_trajectory_with_demo_obs.hdf5"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/real_robot/logs/rollout_temporal_ensemble_1x.pkl"
log_fn = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32_run_10_2x_new_temporal_ensemble.pkl"
log_fn = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32_run_10_1x_all_fixes.pkl"

with open(log_fn, "rb") as f:
    log = pickle.load(f)

demo = log['demo_0']

pt_time = demo['traj_point_time']
obs_ee_pose = demo["obs_ee_pose"]
run_actions = demo['run_actions'] # how many actions were executed per prediction
all_preds_list = demo['all_preds_list']


# Setup model for plotting with obs:
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32/20241108134533/models/model_epoch_800.pth"
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_30/absolute_joint_control_no_framestack_all_obs_seq_32/20241030202541/models/model_epoch_1000.pth"

# device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# restore policy
_, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
model, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

def plot_preds_with_saved_obs(demo, model):

    model.start_episode()

    kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "control_mode": "Joint_Position_Trajectory", "delta_model": False,
              "temporal_ensemble": True, "spline": True,
              "diffusion_sample_n": 1,  "run_actions" : 10}

    obs = demo['obs']
    start_time = obs[0]['joint_state_time']

    obs_X_list = []
    obs_y_list = []
    pred_X_lists = []
    pred_y_lists = []

    for i in range(len(obs)):
        o = obs[i]
        pred = model(o, **kwargs)

        obs_time = obs[i]['joint_state_time'] - start_time
        obs_X_list.append(obs_time)
        obs_y_list.append(obs[i]['joint_positions'][PLOT_JOINT] * ANGLE_MULTIPLIER)

        pred_X = []
        pred_y = []
        for j in range(pred.shape[0]):
            pred_X.append(obs_time + (j + 1) * pt_time)
            pred_y.append(pred[j][PLOT_JOINT] * ANGLE_MULTIPLIER)

        pred_X_lists.append(pred_X)
        pred_y_lists.append(pred_y)

    for i, X in enumerate(obs_X_list):
        plt.plot(obs_X_list[i], obs_y_list[i], label="joint state obs given to model", ls='None', marker=f'${str(i)}$',
                 color='red', markersize=18)
        plt.plot(pred_X_lists[i], pred_y_lists[i], label="model preds on saved obs", ls='None', marker=f'${str(i)}$',
                 color='orange', markersize=12)



def plot_joint_predictions(demo, plot_actual_actions=True):
    # Setup for plotting
    actions = demo['actions']
    preds = demo['preds']
    obs_joint_pos = demo['obs_joint_pos']
    drop_actions = demo['drop_action_list']
    obs_acquisition_times = demo['obs_acquisition_times']
    pred_times = demo['inf_start_times']

    predictions_times = []
    predicted_actions_list = []

    actual_actions_list = [] # The actual actions that were sent to the controller after smoothing, blending, etc.
    actual_actions_times = []

    obs_X_list = []
    obs_y_list = []

    sideview_left_time_list = []

    start_time = ros_time_to_float(demo['traj_start_times'][0])
    # start_time = 0

    for i, time in enumerate(demo['traj_start_times']):
        time = ros_time_to_float(time) - start_time
        pred_time = ros_time_to_float(pred_times[i]) - start_time
        joint_obs_time = obs_acquisition_times[i]['joint_state_time'] - start_time
        sideview_left_time = obs_acquisition_times[i]['sideview_left_camera_rgb_time'] - start_time
        actual_times = []
        actual_actions = []
        pred_y = []
        pred_x = []

        obs_X_list.append(joint_obs_time)
        obs_y_list.append(obs_joint_pos[i][PLOT_JOINT]* ANGLE_MULTIPLIER)
        sideview_left_time_list.append(sideview_left_time)
        for j in range(preds[i].shape[0]):
            pred_x.append(pred_time + (j + 1) * pt_time)
            pred_y.append(preds[i][j, PLOT_JOINT] * ANGLE_MULTIPLIER)
        for k in range(actions[i].shape[0]):
            actual_times.append(time + (k + 1 + drop_actions[i]) * pt_time)
            actual_actions.append(actions[i][k, PLOT_JOINT] * ANGLE_MULTIPLIER)
        predicted_actions_list.append(pred_y)
        predictions_times.append(pred_x)

        actual_actions_times.append(actual_times)
        actual_actions_list.append(actual_actions)

    for i, X in enumerate(predictions_times):
        plt.plot(obs_X_list[i], obs_y_list[i], label="joint state obs given to model", ls='None', marker=f'${str(i)}$',
                 color='red', markersize=18)
        plt.plot(X, predicted_actions_list[i], label="predictions", ls='None', marker=f'${str(i)}$', color="orange", markersize=14)

    if plot_actual_actions:
        for i, X in enumerate(actual_actions_times):
            plt.plot(X, actual_actions_list[i], label="processed actions (temporal ensemble + spline)", ls="None", marker=f'${str(i)}$', markersize=10, color="green")

def match_obs_joint_state_to_all(demo):
    joint_msg_times = []
    for msg in demo['all_joint_msg']:
        joint_msg_times.append(ros_time_to_float(msg.header.stamp))

    obs_acquisition_times = demo['obs_acquisition_times']
    inds = []
    for time in obs_acquisition_times:
        time = time[0]
        ind = joint_msg_times.index(time)
        inds.append(ind)

    print()


def plot_actual_joint_states(demo):

    # First, find out when the trajectory started to filter out the joint messaeges
    joint_msg_times = []
    for msg in demo['all_joint_msg']:
        joint_msg_times.append(ros_time_to_float(msg.header.stamp))

    traj_publish_start = demo['traj_publish_times'][0]
    traj_publish_start = ros_time_to_float(traj_publish_start)
    traj_publish_end = ros_time_to_float(demo['traj_publish_times'][-1])

    start_time = ros_time_to_float(demo['traj_start_times'][0])
    # start_time = 0

    j_msg_start_ind = max(bisect_left(joint_msg_times, traj_publish_start) - 100, 0)
    j_msg_end_ind = bisect_left(joint_msg_times, traj_publish_end) + 100


    joint_msgs = demo["all_joint_msg"][j_msg_start_ind:j_msg_end_ind]
    j_msg_X = []
    j_msg_y = []

    for msg in joint_msgs:
        joint_name = JOINT_TO_NAME[PLOT_JOINT]
        ind = msg.name.index(joint_name)
        j_msg_X.append(ros_time_to_float(msg.header.stamp) - start_time)
        j_msg_y.append(msg.position[ind] * ANGLE_MULTIPLIER)

    ## PLOT THE ACTUAL JOINT STATE
    plt.plot(j_msg_X, j_msg_y, color="blue", label="actual joint positions")


def plot_published_traj_messages(demo):

    ### Plot the actual published traj msgs
    traj_msgs = demo['published_traj_msgs']
    start_time = ros_time_to_float(demo['traj_start_times'][0])
    msg_X_list = []
    msg_y_list = []

    for msg in traj_msgs:
        msg_X = []
        msg_y = []
        for point in msg.points:
            x = ros_time_to_float(msg.header.stamp) + ros_time_to_float(point.time_from_start) - start_time
            msg_X.append(x)
            msg_y.append(point.positions[PLOT_JOINT]* ANGLE_MULTIPLIER)
        msg_X_list.append(msg_X)
        msg_y_list.append(msg_y)

    ## PLOT THE ACTUAL MSGS
    for i, X in enumerate(msg_X_list):
        plt.plot(X, msg_y_list[i], color="purple", label="published traj msgs",  marker=f'${str(i)}$',
                 markersize=8, ls="None")


### Plot predicted ee pose by doing fk on joint position predictions
def plot_ee_pose_from_joint_positions(demo):

    # Setup for plotting
    fk_solver = dev_utils.FK_Solver()


    preds = demo['preds']
    run_actions = demo['run_actions']

    X_lists = []
    y_lists = []

    obs_x_list = []
    obs_y_list = []

    for i, pred in enumerate(preds):
        start_timestep = i*run_actions# each pred is a sequence
        X = []
        y = []
        obs_x_list.append(start_timestep)
        obs_y_list.append(obs_ee_pose[i][PLOT_CARTESIAN_FEATURE])
        for j in range(pred.shape[0]):
            X.append(start_timestep + j)
            q = pred[j][:-1]
            ee_pose = fk_solver.joints_to_ee_pose(q)
            y.append(ee_pose[PLOT_CARTESIAN_FEATURE])
        X_lists.append(X)
        y_lists.append(y)

    for idx, X in enumerate(X_lists):
        # plt.plot(X, y_lists[idx], ls='None', marker=f'${str(idx)}$',
        #          color='orange', markersize=12, label="z position from predicted joint action")
        plt.plot(X, y_lists[idx], label="z position from predicted joint action")
        plt.plot(obs_x_list[idx], obs_y_list[idx], ls='None', marker=f'${str(idx)}$',
                 color='red', markersize=16, label="obs z position of robot")
#
# plot_joint_predictions(demo)
# plot_actual_joint_states(demo)
# plot_published_traj_messages(demo)
plot_ee_pose_from_joint_positions(demo)
# match_obs_joint_state_to_all(demo)
# plot_preds_with_saved_obs(demo, model)
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
# plt.title("ee pose for rollout at 3x")
plt.title("Joint state and predictions for rollout at 3x")
plt.legend(by_label.values(), by_label.keys())
# plt.ylim(Y_LIM)
plt.show()
print()

