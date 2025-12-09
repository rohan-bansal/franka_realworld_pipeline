import os
import argparse

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", required=False, default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy_new/1208_atm_dp_spatracker_mfm_baseline_abs_20_demos_1455_seed1")
args = parser.parse_args()

# evaluation configs
train_gpu_ids = [0]
env_gpu_ids = [0]


exp_dir = args.exp_dir
command = (f'python experiments/run_trained_policy_atm_absolute_3.py --config-dir={exp_dir} --config-name=config hydra.run.dir=/tmp '
            f'+save_path={exp_dir} '
            f'train_gpus="{train_gpu_ids}" '
            f'env_cfg.env_name="realworld" env_cfg.task_name="realworld" '
            f'env_cfg.render_gpu_ids="{env_gpu_ids}" env_cfg.vec_env_num=10 ')

os.system(command)