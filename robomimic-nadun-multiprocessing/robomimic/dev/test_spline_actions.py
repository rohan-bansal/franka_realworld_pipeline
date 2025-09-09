import h5py
import numpy as np
from copy import  deepcopy
import json

from numpy import poly1d
import imageio
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly
from robomimic.dev.dev_utils import complete_setup_for_replay
import robomimic.utils.file_utils as FileUtils

ORIGINAL_ACTION_KEY = "joint_position_actions"
SPLINE_ACTION_KEY = "joint_position_actions_cubic_spline_32"
SPLINE_LENGTH = 32
SEGMENT_LENGTH = 4
KNOTS = 8
ACT_DIM = 8
NUM_COEFFS = 4
PLOT_FEATURE = 2

PLOT_FEATURE_TO_LABEL = {
    0 : "x-axis position",
    1 : "y-axis position",
    2 : "z-axis position",
    3 : "angle component 1",
    4 : "angle component 2",
    5 : "angle component 3"
}

PLOT_FEATURE_TO_LABEL_JOINT_POSITIONS = {
    0 : "base joint",
    1 : "shoulder 1 joint",
    2 : "shoulder 2 joint",
    3 : "elbow 1 joint",
    4 : "elbow 2 joint",
    5 : "wrist 1 joint",
    6 : "wrist 2 joint",
    7 : "gripper joint"
}

Y_LABEL = PLOT_FEATURE_TO_LABEL_JOINT_POSITIONS[PLOT_FEATURE]
# Y_LABEL = PLOT_FEATURE_TO_LABEL[PLOT_FEATURE]

demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"

def replay_osc_spline_actions(demo_fn, limit, video_fn=None):
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    abs_env_meta = deepcopy(env_meta)
    abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    env, demo_file = complete_setup_for_replay(demo_fn, env_meta=abs_env_meta)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    reward = 0
    num_actual_actions = 0
    num_splines_played = 0

    for ep in demo_file["data"]:

        counter += 1
        if counter > limit:
            break

        if counter % 20 == 0 :
            print(f"Replaying demo : {counter}")

        demo = demo_file[f'data/{ep}']
        spline_actions = demo[SPLINE_ACTION_KEY][:]
        original_actions = demo[ORIGINAL_ACTION_KEY][:]
        act_dim = original_actions.shape[1]
        all_smooth_actions = []
        all_timesteps = []
        all_actual_actions = []

        timestep = 0


        # Reset env
        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        n = original_actions.shape[0]

        x = [j for j in range(SPLINE_LENGTH)]

        for i in range(0, n, SPLINE_LENGTH):
            per_act_dim_coeefs = spline_actions[i].reshape(ACT_DIM, NUM_COEFFS, KNOTS, order='C')
            knots = [x for x in range(0, SPLINE_LENGTH, SEGMENT_LENGTH)]
            knots.append(SPLINE_LENGTH - 1)
            smooth_act = []
            for d in range(act_dim):
                coeef = per_act_dim_coeefs[d]
                spline = PPoly(coeef, knots)  # init the spline
                sample = spline(x)
                smooth_act.append(sample)
            smooth_act = np.array(smooth_act).transpose()
            # sampled_act = spline(x) # sample from the spline
            actual_act = original_actions[i:i + SPLINE_LENGTH]

            for s in range(smooth_act.shape[0]):
                act = smooth_act[s]
                all_smooth_actions.append(act[PLOT_FEATURE])
                # all_actual_actions.append(original_actions[timestep, PLOT_FEATURE])
                all_timesteps.append(timestep)
                next_obs, _, _, _ = env.step(act)
                timestep += 1

                if video_fn is not None:
                    video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                    video_writer.append_data(video_img)
        plt.plot(all_timesteps, all_smooth_actions, label="spline generated actions")

        # Plotting the actual actions
        original_timesteps = range(original_actions.shape[0])
        plt.plot(original_timesteps, original_actions[:, PLOT_FEATURE], label="actual actions")
        plt.title("Original vs splined actions")
        plt.ylabel(Y_LABEL)
        plt.xlabel("timestep")
        plt.legend()
        plt.show()

def replay_osc_polyfit_actions(demo_fn, limit=10, video_fn=None):
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    abs_env_meta = deepcopy(env_meta)
    abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    env, demo_file = complete_setup_for_replay(demo_fn, env_meta=abs_env_meta)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    reward = 0
    num_actual_actions = 0
    num_splines_played = 0

    for ep in demo_file["data"]:

        counter += 1
        if counter > limit:
            break

        if counter % 20 == 0:
            print(f"Replaying demo : {counter}")

        demo = demo_file[f'data/{ep}']
        spline_actions = demo[SPLINE_ACTION_KEY][:]
        original_actions = demo[ORIGINAL_ACTION_KEY][:]
        act_dim = original_actions.shape[1]

        # Reset env
        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        n = spline_actions.shape[0]

        x = [j for j in range(SPLINE_LENGTH)]

        for i in range(0, n, SPLINE_LENGTH):
            per_act_dim_coeefs = spline_actions[i]
            smooth_act = []
            for d in range(act_dim):
                coeef = per_act_dim_coeefs[d]
                cubic_poly = poly1d(coeef)  # init the spline
                sample = cubic_poly(x)
                smooth_act.append(sample)
            smooth_act = np.array(smooth_act).transpose()
            # sampled_act = spline(x) # sample from the spline
            actual_act = original_actions[i:i + SPLINE_LENGTH]

            for s in range(smooth_act.shape[0]):
                act = smooth_act[s]
                next_obs, _, _, _ = env.step(act)

                if video_fn is not None:
                    video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                    video_writer.append_data(video_img)


def replay_joint_spline_actions(demo_fn, limit, video_fn=None):
    env, demo_file = complete_setup_for_replay(demo_fn)

    ### Init env
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    from robomimic.robosuite_configs.paths import joint_position_nadun as jp_path
    joint_controller_fp = jp_path()
    controller_configs = json.load(open(joint_controller_fp))
    env_meta["env_kwargs"]["controller_configs"] = controller_configs



    env, demo_file = complete_setup_for_replay(demo_fn, env_meta=env_meta)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    reward = 0
    num_actual_actions = 0
    num_splines_played = 0

    for ep in demo_file["data"]:

        counter += 1
        if counter > limit:
            break

        if counter % 20 == 0 :
            print(f"Replaying demo : {counter}")

        demo = demo_file[f'data/{ep}']
        spline_actions = demo[SPLINE_ACTION_KEY][:]
        original_actions = demo[ORIGINAL_ACTION_KEY][:]
        act_dim = original_actions.shape[1]
        all_smooth_actions = []
        all_timesteps = []

        timestep = 0


        # Reset env
        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        n = original_actions.shape[0]


        x = [j for j in range(SPLINE_LENGTH)]

        for i in range(0, n, SPLINE_LENGTH):
            per_act_dim_coeefs = spline_actions[i].reshape(ACT_DIM, NUM_COEFFS, KNOTS, order='C')
            knots = [x for x in range(0, SPLINE_LENGTH, SEGMENT_LENGTH)]
            knots.append(SPLINE_LENGTH - 1)
            smooth_act = []
            for d in range(act_dim):
                coeef = per_act_dim_coeefs[d]
                spline = PPoly(coeef, knots)  # init the spline
                sample = spline(x)
                smooth_act.append(sample)
            smooth_act = np.array(smooth_act).transpose()
            # sampled_act = spline(x) # sample from the spline
            actual_act = original_actions[i:i+SPLINE_LENGTH]

            for s in range(smooth_act.shape[0]):
                act = smooth_act[s]
                all_smooth_actions.append(act[PLOT_FEATURE])

                # Convert absolute joint position action to delta
                obs = env.get_observation()
                joint_pos = obs["robot0_joint_pos"]
                act[:-1] = act[:-1] - joint_pos

                all_timesteps.append(timestep)
                next_obs, _, _, _ = env.step(act)
                timestep += 1

                if video_fn is not None:
                    video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
                    video_writer.append_data(video_img)
        plt.plot(all_timesteps, all_smooth_actions, label="spline generated actions")

        # Plotting the actual actions
        original_timesteps = range(original_actions.shape[0])
        plt.plot(original_timesteps, original_actions[:, PLOT_FEATURE], label="actual actions")
        plt.title("Original vs splined actions")
        plt.ylabel(Y_LABEL)
        plt.xlabel("timestep")
        plt.legend()
        # plt.show()

video_fn = "/media/nadun/Data/phd_project/robomimic/videos/spline_actions/demo_replay/can_10_replay_joint_spline_flattened.mp4"
# video_fn = None
# replay_osc_spline_actions(demo_fn, limit=10, video_fn=video_fn)
replay_joint_spline_actions(demo_fn, limit=10, video_fn=video_fn)
# replay_polyfit_actions(demo_fn, limit=2, video_fn=video_fn)
