import robomimic
import argparse
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

import pickle
from tqdm import tqdm


from robomimic.dev.dev_utils import complete_setup_for_replay


def sample_init_states(demo_fn, save_fn, num_samples=500):
    env, demo_file = complete_setup_for_replay(demo_fn)

    init_state_samples = []

    for i in tqdm(range(num_samples)):
        env.reset()
        state_dict = env.get_state()
        init_state_samples.append(state_dict)

    with open(save_fn, "wb") as f:
        pickle.dump(init_state_samples, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=None,
        help="path to dataset",
    )

    parser.add_argument(
        "--save_fn",
        type=str,
        required=False,
        default=None,
        help="path to dataset",
    )

    args = parser.parse_args()

    sample_init_states(args.dataset, args.save_fn)


