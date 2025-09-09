import robomimic.scripts.train as rmtrain
from argparse import Namespace

if __name__ == "__main__":
    args = Namespace(
        config="robomimic/dev/cfg/base_config_square.json",
        algo="diffusion_policy",
        name="debug_diffusion_policy",
        dataset="/home/wjung85/Repo/projects/FastIL/datasets/square/ph/all_obs_new.hdf5",
        gpu=0,
        debug=False,
        output=None,
        auto_remove_exp = None,
    )
    rmtrain.main(args)