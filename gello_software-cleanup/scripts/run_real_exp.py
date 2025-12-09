import os
import argparse

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# input parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy-dir",
    required=False,
    default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy_new/1208_atm_dp_spatracker_mfm_baseline_abs_20_demos_1455_seed1",
)
parser.add_argument(
    "--checkpoint-name",
    required=False,
    default="model_2000.ckpt",
)
parser.add_argument(
    "--action-norms-path",
    required=False,
    default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/action_norms/action_normalization_stats_full.pkl",
)
args = parser.parse_args()

# evaluation configs
train_gpu_ids = [0]
env_gpu_ids = [0]


exp_dir = args.policy_dir
checkpoint_path = os.path.join(exp_dir, args.checkpoint_name)
command = (
    f'python experiments/run_trained_policy_atm_absolute_3.py '
    f'--config-dir={exp_dir} --config-name=config hydra.run.dir=/tmp '
    f'+save_path={exp_dir} '
    f'+checkpoint_path={checkpoint_path} '
    f'+action_norms_path={args.action_norms_path} '
    f'train_gpus="{train_gpu_ids}" '
    f'env_cfg.env_name="realworld" env_cfg.task_name="realworld" '
    f'env_cfg.render_gpu_ids="{env_gpu_ids}" env_cfg.vec_env_num=10 '
)

os.system(command)