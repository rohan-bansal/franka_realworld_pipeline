import h5py
import numpy as np
import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.dev.dev_utils import demo_obs_to_obs_dict, postprocess_obs, FK_Solver
import matplotlib.pyplot as plt



kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "control_mode": "Joint_Position_Trajectory", "delta_model": False,
              "temporal_ensemble": True, "spline": False, "inpaint_first_action": False,
              "diffusion_sample_n": 10, "return_all_pred": True}

POINT_TIME = 0.05
PLOT_JOINT = 0
PLOT_CARTESIAN_POSE_FEATURE = 2 # which part of the predicted and actual cartesian pose to plot
RUN_ACTIONS = 10
OBSERVATION_HORIZON = 1

ANGLE_MULTIPLIER = 1
ANGLE_MULTIPLIER = 180/np.pi
Y_LIM = (-1.90*ANGLE_MULTIPLIER, -1.10*ANGLE_MULTIPLIER)


### SETUP THE MODEL
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_24_filtered/absolute_joint_control_framestack_2_image_only/20241024231728/models/model_epoch_800.pth"
# ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_24_filtered/absolute_joint_control_framestack_2_all_obs/20241024233437/models/model_epoch_800.pth"
# ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_22/absolute_joint_control_framestack_2_image_only/20241022221216/models/model_epoch_800.pth"
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_10_30/absolute_joint_control_no_framestack_all_obs/20241030202123/models/model_epoch_1000.pth"
ckpt_path = "/home/robot-aiml/ac_learning_repos/robomimic-nadun/bc_trained_models/real_robot/pick_cube_11_7/absolute_joint_control_no_framestack_all_obs_seq_32/20241108134533/models/model_epoch_800.pth"

# device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# restore policy
_, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)


### SETUP THE DEMO:


demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_10_24_filtered.hdf5"
demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_10_30_holdout_filtered.hdf5"
demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_11_7_holdout_night_filtered.hdf5"
# demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/demo_pick_cube_11_7_filtered.hdf5"
# demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/pick_cube/demo_pick_cube_10_22_holdout.hdf5"

demo_file = h5py.File(demo_fn)['data']

demo = demo_file['demo_0']

def plot_actual_joint_preds(demo, policy):
    demo_obs = demo['obs']
    demo_length = demo.attrs['num_samples']

    joint_acts = demo['joint_position_actions'][:]

    # Start 'rollout'
    policy.start_episode()

    ### DATA COLLECTION STUFF

    pred_actions = []
    drop_action_list = []

    obs_x_list = []
    obs_y_list = []
    pred_x_list = []
    pred_y_list = []

    actual_joint_act = []

    for i in range(0, demo_length, RUN_ACTIONS):
        obs = demo_obs_to_obs_dict(demo_obs, i, OBSERVATION_HORIZON)
        obs = postprocess_obs(obs)
        act, all_preds = policy(ob=obs, **kwargs)


        pred_actions.append(act)

        # Generating stuff for plotting
        obs_x_list.append([i*POINT_TIME])
        if OBSERVATION_HORIZON > 1:
            obs_y_list.append([obs['joint_positions'][-1, PLOT_JOINT]])
        else:
            obs_y_list.append([obs['joint_positions'][PLOT_JOINT]*ANGLE_MULTIPLIER])
        actual_joint_act.append(joint_acts[i, PLOT_JOINT] * ANGLE_MULTIPLIER)

        pred_x = []
        pred_y = []

        for j in range(act.shape[0]):
            pred_x.append((i*POINT_TIME) + (j + 1)*POINT_TIME)
            pred_y.append(act[j, PLOT_JOINT] * ANGLE_MULTIPLIER)

        pred_x_list.append(pred_x)
        pred_y_list.append(pred_y)

    for i, X in enumerate(pred_x_list):
        plt.plot(obs_x_list[i], obs_y_list[i], label="joint state at time of prediction", ls='None', marker=f'${str(i)}$',
                 color='red', markersize=18)
        plt.plot(obs_x_list[i], actual_joint_act[i], label="actual joint action", ls='None', marker=f'${str(i)}$',
                 color='purple', markersize=14)
        plt.plot(X, pred_y_list[i], label="predictions", ls='None', marker=f'${str(i)}$', color="orange", markersize=12)


    plt.ylabel("Joint Position")
    plt.xlabel("Timestep")


def plot_cartesian_pose_from_preds(demo, policy, plot_pred_sequence=True):

    # Start FK solver
    fk_solver = FK_Solver()


    demo_obs = demo['obs']
    demo_length = demo.attrs['num_samples']

    # The action representations
    cartesian_acts = demo['absolute_axis_angle_actions'][:]
    joint_acts = demo['joint_position_actions'][:]

    # Start 'rollout'
    policy.start_episode()

    ### DATA COLLECTION STUFF

    pred_actions = []
    prev_actions_completed = []
    drop_action_list = []


    obs_x_list = []
    obs_y_list = []
    pred_x_list = []
    pred_y_list = []
    actual_cartesian_acts = []
    actual_joint_acts = []

    for i in range(0, demo_length, RUN_ACTIONS):
        obs = demo_obs_to_obs_dict(demo_obs, i, OBSERVATION_HORIZON)
        obs = postprocess_obs(obs)
        act, all_preds = policy(ob=obs, **kwargs)

        # First save the actual actions
        actual_cartesian_acts.append(cartesian_acts[i][PLOT_CARTESIAN_POSE_FEATURE])
        actual_joint_act = joint_acts[i]
        calc_cartesian_act_from_joint = fk_solver.joints_to_ee_pose(actual_joint_act[:-1]) # joint act to cartesian
        actual_joint_acts.append(calc_cartesian_act_from_joint[PLOT_CARTESIAN_POSE_FEATURE])

        # When the observation was taken
        obs_x_list.append([i * POINT_TIME])


        pred_x = []
        pred_y = []
        if plot_pred_sequence:
            for j in range(act.shape[0]):
                pred_x.append((i * POINT_TIME) + (j + 1) * POINT_TIME)
                pred = act[j]
                calc_ee_pose = fk_solver.joints_to_ee_pose(pred[:-1]) # predicted joint action to cartesian pose
                pred_y.append(calc_ee_pose[PLOT_CARTESIAN_POSE_FEATURE])
        else:
            # Plot only the first prediction in sequence
            for j in range(1):
                pred_x.append((i * POINT_TIME) + (j + 1) * POINT_TIME)
                pred = act[j]
                calc_ee_pose = fk_solver.joints_to_ee_pose(pred[:-1]) # predicted joint action to cartesian pose
                pred_y.append(calc_ee_pose[PLOT_CARTESIAN_POSE_FEATURE])

        pred_x_list.append(pred_x)
        pred_y_list.append(pred_y)

    for i, X in enumerate(pred_x_list):
        # plt.plot(obs_x_list[i], obs_y_list[i], label="x position at time of prediction", ls='None', marker=f'${str(i)}$',
        #          color='red', markersize=18)
        plt.plot(obs_x_list[i], actual_cartesian_acts[i], label="commanded z position in demo", ls='None', marker=f'${str(i)}$',
                 color='red', markersize=18)
        plt.plot(obs_x_list[i], actual_joint_acts[i], label="z position from actual joint action", ls='None', marker=f'${str(i)}$',
                 color='purple', markersize=14)
        if plot_pred_sequence:
            plt.plot(X, pred_y_list[i], label="z position from predicted joint action")
        else:
            plt.plot(X, pred_y_list[i], label="z position from predicted joint action", ls='None', marker=f'${str(i)}$', color="orange", markersize=12)




# plot_cartesian_pose_from_preds(demo, policy, True)
plot_actual_joint_preds(demo, policy)

### Plotting stuff

plt.ylabel("Z Position of robot")
plt.xlabel("Timestep")


plt.title("Demo inference on OOD demo")
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
plt.legend(by_label.values(), by_label.keys())
# plt.ylim(Y_LIM)
plt.show()



print()