import os
import argparse

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# input parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy-dir",
    required=False,
    default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy_official/1230_atm_dp_16_tasks_20_demos_1648_seed1",
)
parser.add_argument(
    "--policy-checkpoint",
    required=False,
    default="model_2000.ckpt",
)
parser.add_argument(
    "--track-policy-dir",
    required=False,
    default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/track_transformer_official/1230_realworld_track_transformer_mfm_16_realworld_tasks_ep3001_0258"
)
parser.add_argument(
    "--action-norms-path",
    required=False,
    default="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy_official/action_norm/action_normalization_stats_full.pkl",
)
args = parser.parse_args()

# evaluation configs
train_gpu_ids = [0]
env_gpu_ids = [0]


exp_dir = args.policy_dir
checkpoint_path = os.path.join(exp_dir, args.policy_checkpoint)
command = (
    f'python experiments/run_trained_policy_atm_absolute_lang.py '
    f'--config-dir={exp_dir} --config-name=config hydra.run.dir=/tmp '
    f'+save_path={exp_dir} '
    f'+checkpoint_path={checkpoint_path} '
    f'+action_norms_path={args.action_norms_path} '
    f'train_gpus="{train_gpu_ids}" '
    f'env_cfg.env_name="realworld" env_cfg.task_name="realworld" '
    f'env_cfg.render_gpu_ids="{env_gpu_ids}" env_cfg.vec_env_num=10 '
    f'model_cfg.track_cfg.track_fn={args.track_policy_dir}'
)

os.system(command)
