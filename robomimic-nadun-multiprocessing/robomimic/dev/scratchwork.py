import os
import nexusformat.nexus as nx
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
#

log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/square_image_action_horizon_16/20241211173329/logs/normal_eval_slow2025-01-05 22:42:33.114050.pkl"

demo_fn = "/home/mbronars/zhenyang/demos/wiping_board_0105_singlestart_singleline_demo/2025-01-05_demo.hdf5"

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

absolute_actions = demo['absolute_actions'][:, :3]
commanded_absolute_actions = demo['commanded_absolute_actions'][:, :3]
eef_pose = demo['obs/robot0_eef_pos'][:]

diff = absolute_actions - eef_pose
diff2 = commanded_absolute_actions - eef_pose
print()

# log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/absolute_osc/stack_three_image_action_horizon_16_speed_baseline_absolute_osc/20250122162954/logs/eval_normal_time_uncapped_small_test2025-01-23 18:44:07.834905.pkl"
# video_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/absolute_osc/stack_three_image_action_horizon_16_speed_baseline_absolute_osc/20250122162954/videos/model_eye_in_hand.mp4"
#
# writer = imageio.get_writer(video_fn, fps=20)
#
# with open(log_fn, "rb") as f:
#     data = pickle.load(f)
#
# rollouts = data['rollouts']
# for demo in rollouts:
#     demo_data = rollouts[demo]
#     obs_list = demo_data['obs']
#     for obs in obs_list:
#         agentview_image = obs['robot0_eye_in_hand_image'][-1]
#         writer.append_data(agentview_image)
#
# writer.close()
# print()



# env_args['env_name'] = 'StackThree_D0'
# env_args['env_version'] = '1.4.1'
#
# data.attrs['env_args'] = json.dumps(env_args, indent=4)


