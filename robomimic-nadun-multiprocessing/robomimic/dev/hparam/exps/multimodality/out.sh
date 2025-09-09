#!/bin/bash

# conditional model
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_no_guide.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_inp_true_nres_0.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_inp_true_nres_3.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_inp_true_nres_10.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_inp_true_nres_50.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_weight_0.3.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_weight_0.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_weight_1.json
python /home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_fixed_obs.py --guide_config /home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/exps/multimodality/baseline_config_square_cfg_true_weight_2.json

# unconditional model