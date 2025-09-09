import numpy as np
import h5py
from copy import deepcopy
from robomimic.dev.dev_utils import in_same_direction, create_aggregated_delta_actions_with_gripper_check
import nexusformat.nexus as nx


demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"
processed_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141_with_aggregated_actions.hdf5"

demo_file = nx.nxload(demo_fn)
print(demo_file.tree)

demo_file = h5py.File(demo_fn)
out_file  = h5py.File(processed_fn, 'w')

for obj in demo_file.keys():
    demo_file.copy(obj, out_file)

out_file_data = out_file['data']

delta_action_magnitudes = [1.0, 2.0, 3.0, 4.0]

for mag in delta_action_magnitudes:
    kw_args = {'delta_action_magnitude_limit': mag}
    for ep in demo_file["data"]:
        demo = demo_file[f'data/{ep}']
        delta_actions = deepcopy(demo["actions"][:])
        obs = demo['obs']
        print(f"processing demo {ep}")

        agg_actions = create_aggregated_delta_actions_with_gripper_check(delta_actions, obs, **kw_args)

        # Write the dataset with new actions
        out_file_ep_grp = out_file_data[f'{ep}']
        # del out_file_ep_grp['actions']

        out_file_ep_grp.create_dataset(f"agg_actions_magnitude_{int(mag)}", data=agg_actions)
        out_file_ep_grp.attrs['num_samples'] = len(agg_actions)

out_file.close()

