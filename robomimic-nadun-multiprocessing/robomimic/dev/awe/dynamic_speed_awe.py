import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.image


import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.transform_utils as TransformUtils
from robosuite.utils.motion_profile import MotionProfile, MotionProfileSteps
from robomimic.envs.env_base import EnvBase
from robomimic.dev.dev_utils import aggregate_delta_actions, in_same_direction, \
    aggregate_delta_actions_with_gripper_check
from robomimic.dev.dev_utils import DELTA_ACTION_MAGNITUDE_LIMIT, SCALE_ACTION_LIMIT, REPEAT_LAST_ACTION_TIMES
from robomimic.dev.dev_utils import complete_setup_for_replay

import nexusformat.nexus as nx

def dynamic_ctrl_freq_demo(demo_fn, limit, video_fn, mode):

    env_meta = FileUtils.get_env_metadata_from_dataset(demo_fn)
    abs_env_meta = deepcopy(env_meta)
    abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    abs_env_meta["env_kwargs"]['has_offscreen_renderer'] = True
    abs_env_meta['env_kwargs']["use_camera_obs"] = True
    env, demo_file = complete_setup_for_replay(demo_fn, env_meta=abs_env_meta)

    if video_fn is not None:
        video_writer = imageio.get_writer(video_fn, fps=20)

    counter = 0
    num_actions = 0
    num_precise_actions = 0

    # MODIFY TO CHANGE "SPEED" OF SIMULATION
    BASE_CONTROL_FREQ = 20
    RENDER_INTERVAL_SEC = 0.1

    num_success = 0
    sim_time_success = 0
    
    for ep in demo_file["data"]:
        counter += 1
        if counter > limit:
            break

        print(f"processing demo: {counter}")

        demo = demo_file[f'data/{ep}']

        states = demo_file["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo_file["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        env.reset_to(initial_state)

        actions = demo["absolute_actions_with_precision"]
        precisions = actions[:, -1]  # Extract the last column of actions
        print(precisions)
        num_precise_actions += np.sum(precisions)
        actions = actions[:, :7]
        waypoints = demo["waypoints_dp"][:]


        if video_fn is not None:
            video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1]
            blank_img = np.zeros((512, 512, 3), dtype=np.uint8)
            combined_img = cv2.hconcat([blank_img, video_img])
            video_writer.append_data(combined_img)

        n = actions.shape[0]
        num_actions += n

        motion_profile = MotionProfileSteps(default_control_freq=20, integers_only=False, default_easing=None)

        min_freq = 20
        max_freq = 40


        if mode == "INV_PROP_DIST":
            # Handle start to first waypoint
            start_freq = max_freq - (max_freq - min_freq) * precisions[0]
            first_wp_freq = max_freq - (max_freq - min_freq) * precisions[waypoints[0]]
            motion_profile.add_interval(
                0,
                waypoints[0],
                start_freq,
                first_wp_freq
            )

            # Handle between waypoints
            for i in range(1, len(waypoints)):
                start_idx = waypoints[i-1]
                end_idx = waypoints[i]
                
                prev_freq = max_freq - (max_freq - min_freq) * precisions[start_idx]
                freq = max_freq - (max_freq - min_freq) * precisions[end_idx]

                motion_profile.add_interval(
                    start_idx,
                    end_idx,
                    prev_freq,
                    freq
                )

            # Handle last waypoint to end
            last_wp_freq = max_freq - (max_freq - min_freq) * precisions[waypoints[-1]]
            end_freq = max_freq - (max_freq - min_freq) * precisions[-1]
            motion_profile.add_interval(
                waypoints[-1],
                len(precisions) - 1,
                last_wp_freq,
                end_freq
            )
        elif mode == "CLUSTERING":
            for i in range(len(precisions)):
                freq = max_freq - (max_freq - min_freq) * precisions[i]

                motion_profile.add_interval(
                    i,
                    i + 1,
                    freq,
                )

        fig = motion_profile.generate_graph(n)
        ax = fig.gca()
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        graph_img = cv2.resize(graph_img, (512, 512))
        plt.close(fig)


        for i in range(n):

            control_frequency = motion_profile.get_control_frequency(i)
            ax.plot(i, control_frequency, 'ro')
            if i in waypoints:
                if precisions[i] == 1 and mode == "CLUSTERING":
                    ax.plot(i, control_frequency, 'yo', markersize=10)  # Yellow circle cluster
                else:
                    ax.plot(i, control_frequency, 'bo', markersize=10)
            fig.canvas.draw()
            graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
            graph_img = cv2.resize(graph_img, (512, 512))

            act = np.copy(actions[i])
            # print(f"Stepping action: {i}, control_freq: {control_frequency}")

            next_obs, _, _, _ = env.step(act, control_freq=control_frequency)

            sim_steps, sim_time_elapsed = env.getSimTimeInfo()

            render_interval = RENDER_INTERVAL_SEC
            if video_fn is not None and (sim_time_elapsed % render_interval < (1 / control_frequency)):
                video_img = env.env.sim.render(height=512, width=512, camera_name="agentview")[::-1].astype('uint8')
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(video_img, f"hz: {control_frequency:.2f} | stp: {sim_steps} | {sim_time_elapsed:.2f}s", (10,30), font, 0.9, (255, 255, 255), 2)

                combined_img = cv2.hconcat([graph_img, video_img])
                video_writer.append_data(combined_img)

        success = env.get_reward()

        if success:
            num_success += 1
            sim_time_success += env.getSimTimeInfo()[1]

    print(f"Total played: {counter}. Num success: {num_success}. Sim time success: {sim_time_success} \n"
          f"Total num actions: {num_actions}. Num precise: {num_precise_actions}")


    if video_fn is not None:
        video_writer.close()



if __name__ == "__main__":
    demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"
    video_fn = "/media/nadun/Data/phd_project/robomimic/videos/adaptive_speed/can.mp4"

    # can be CLUSTERING or INV_PROP_DIST
    mode = "CLUSTERING"

    dynamic_ctrl_freq_demo(demo_fn, 100, video_fn=video_fn, mode=mode)
