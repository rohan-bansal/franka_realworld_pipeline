import os
import json


Ta_LIST = [i for i in range(8, 65, 8)]

def modify_seq_lengths_and_save_dp(config_fp):
    with open(config_fp, 'rb') as f:
        config = json.load(f)
    exp_name = config['experiment']['name']
    for length in Ta_LIST:
        config['experiment']['name'] = f"{exp_name}_action_horizon_{length}"
        config['train']['seq_length'] = length*2
        config['algo']['horizon']['action_horizon'] = length
        config['algo']['horizon']['prediction_horizon'] = length*2

        new_config_filepath = config_fp.replace(".json", f"_action_horizon_{length}.json")
        with open(new_config_filepath, 'w') as f:
            json.dump(config, f, indent=4)

config_fp = "/media/nadun/Data/phd_project/robomimic/skynet/configs/diffusion-policy/sim/action_horizon_experiments/joint_position/can_image.json"

config_fp = "/media/nadun/Data/phd_project/robomimic/skynet/configs/diffusion-policy/sim/action_horizon_experiments/absolute_osc/can_image.json"
modify_seq_lengths_and_save_dp(config_fp)