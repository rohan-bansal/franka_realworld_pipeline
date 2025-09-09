from cProfile import label

import h5py

import torch
import nexusformat.nexus as nx
import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from dev_utils import demo_obs_to_obs_dict, postprocess_obs
import matplotlib.pyplot as plt


### CONSTANTS

kwargs = {"return_action_sequence": True, "step_action_sequence": True,
              "diffusion_sample_n": 1, "return_all_pred": True}


POINT_TIME = 0.05
PLOT_FEATURE = 0
RUN_ACTIONS = 8
OBSERVATION_HORIZON = 2

# device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

def plot_demo_preds_single_line(ckpt_path, demo_fn, label=None):
    # restore policy
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

    ### SETUP THE DEMO:

    demo_file = nx.nxload(demo_fn)
    print(demo_file.tree)
    demo_file = h5py.File(demo_fn)['data']

    demo = demo_file['demo_155']

    demo_obs = demo['obs']
    demo_actions = demo['absolute_actions']

    demo_length = demo.attrs['num_samples']

    # Start 'rollout'
    policy.start_episode()

    pred_actions = []
    obs_ee_pos = []
    prev_actions_completed = []
    drop_action_list = []
    all_preds_list = []

    obs_x_list = []
    obs_y_list = []
    pred_x_list = []
    pred_y_list = []
    demo_acts = []

    for i in range(0, demo_length, RUN_ACTIONS):
        obs = demo_obs_to_obs_dict(demo_obs, i, OBSERVATION_HORIZON)
        obs = postprocess_obs(obs)
        act, all_preds = policy(ob=obs, **kwargs)


        pred_actions.append(act)
        all_preds_list.append(all_preds)




        pred_x = []
        pred_y = []
        for j in range(act.shape[0]):
            pred_x.append((i * POINT_TIME) + (j + 1) * POINT_TIME)
            pred_y.append(act[j, PLOT_FEATURE])
            demo_acts.append(demo_actions[i][PLOT_FEATURE])
            obs_x_list.append([i * POINT_TIME])
            obs_y_list.append([obs['robot0_eef_pos'][-1, PLOT_FEATURE]])
            obs_ee_pos.append(obs['robot0_eef_pos'][-1, PLOT_FEATURE])

        pred_x_list.extend(pred_x)
        pred_y_list.extend(pred_y)


    plt.plot(obs_x_list, demo_acts, label="commanded x position in demo", color="red")
    plt.plot(pred_x_list, pred_y_list, label=label)

    plt.ylabel("X Position of robot")
    plt.xlabel("Timestep")

def plot_demo_preds_sequences(ckpt_path, demo_fn):

    # restore policy
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)


    ### SETUP THE DEMO:


    # demo_fn = "/home/robot-aiml/ac_learning_repos/Task_Demos/merged/move_to_skynet/demo_pick_cube_realsense.hdf5"

    demo_file = nx.nxload(demo_fn)
    print(demo_file.tree)
    demo_file = h5py.File(demo_fn)['data']

    demo = demo_file['demo_155']

    demo_obs = demo['obs']
    demo_actions = demo['absolute_actions']

    demo_length = demo.attrs['num_samples']

    # Start 'rollout'
    policy.start_episode()



    pred_actions = []
    obs_ee_pos = []
    prev_actions_completed = []
    drop_action_list = []
    all_preds_list = []

    obs_x_list = []
    obs_y_list = []
    pred_x_list = []
    pred_y_list = []
    demo_acts = []

    for i in range(0, demo_length, RUN_ACTIONS):
        obs = demo_obs_to_obs_dict(demo_obs, i, OBSERVATION_HORIZON)
        obs = postprocess_obs(obs)
        act, all_preds = policy(ob=obs, **kwargs)

        obs_ee_pos.append(obs['robot0_eef_pos'][-1, PLOT_FEATURE])
        pred_actions.append(act)
        all_preds_list.append(all_preds)

        # Generating stuff for plotting
        obs_x_list.append([i*POINT_TIME])
        obs_y_list.append([obs['robot0_eef_pos'][-1, PLOT_FEATURE]])

        demo_acts.append(demo_actions[i][PLOT_FEATURE])

        pred_x = []
        pred_y = []
        for j in range(act.shape[0]):
            pred_x.append((i*POINT_TIME) + (j + 1)*POINT_TIME)
            pred_y.append(act[j, PLOT_FEATURE])

        pred_x_list.append(pred_x)
        pred_y_list.append(pred_y)

    for i, X in enumerate(pred_x_list):
        # plt.plot(obs_x_list[i], obs_y_list[i], label="x position at time of prediction", ls='None', marker=f'${str(i)}$',
        #          color='red', markersize=18)
        plt.plot(obs_x_list[i], demo_acts[i], label="commanded x position in demo", ls='None', marker=f'${str(i)}$',
                 color='red', markersize=18)
        plt.plot(X, pred_y_list[i], label="predictions", ls='None', marker=f'${str(i)}$', color="orange", markersize=14)


    plt.ylabel("X Position of robot")
    plt.xlabel("Timestep")


demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"
# ckpt_path = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image_20_demos/20241020144021/models/model_epoch_500.pth"
# plot_demo_preds_single_line(ckpt_path, demo_fn, label="20 demo model")

ckpt_path = "/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image_50_demos/20241020144409/models/model_epoch_500.pth"
plot_demo_preds_single_line(ckpt_path, demo_fn, label="50 demo model")

# ckpt_path = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image_100_demos/20241020144449/models/model_epoch_500.pth"
# plot_demo_preds_single_line(ckpt_path, demo_fn, label="100 demo model")

ckpt_path = "/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image_150_demos/20241020144547/models/model_epoch_500.pth"
plot_demo_preds_single_line(ckpt_path, demo_fn, label="150 demo model")


plt.title("Rollout with Demo Obs")
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
plt.legend(by_label.values(), by_label.keys())
plt.show()
