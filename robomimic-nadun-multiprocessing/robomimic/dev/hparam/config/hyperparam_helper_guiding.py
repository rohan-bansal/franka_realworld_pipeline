import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils
import argparse
import tyro


def make_generator_cfg(config_file, script_file):
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )
    
    cfg_enabled = False

    generator.add_param(
        key="guiding.cfg.enabled",
        name="cfg" if cfg_enabled else "",
        group=0,
        values=[
            cfg_enabled,
        ],
        value_names=[
            "t" if cfg_enabled else "f",
        ],
    )

    generator.add_param(
        key = "guiding.ddim_eta",
        name = "eta",
        group = 1,
        values = [0, 1],
        value_names = ["0", "1"],
    )

    if cfg_enabled:
        generator.add_param(
            key = "guiding.cfg.weight",
            name = "cfg_weight",
            group = 2,
            values = [0.0, 0.3, 0.5, 1.0, 2.0, 4.0],
            value_names = ["0.0", "0.3", "0.5", "1.0", "2.0", "4.0"],
        )
    
    else:
        generator.add_param(
            key = "guiding.inpainting.enabled",
            name = "inp",
            group = 2,
            values = [True],
            value_names = ["t"],
        )

        generator.add_param(
            key = "guiding.inpainting.n_resample",
            name = "nres",
            group = 3,
            values = [0, 3, 10],
            value_names = ["0", "3", "10"],
        )

    return generator

def make_generator(config_file, script_file):
    """
    Implement this function to setup your own hyperparameter scan!
    """
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file, script_file=script_file
    )

    use_inpainting = True

    generator.add_param(
        key="guiding.timestep_end",
        name="",
        group=-1,
        values=[0],
    )
    
    generator.add_param(
        key="guiding.inpainting.enabled",
        name="inp" if use_inpainting else "",
        group=0,
        values=[
            use_inpainting,
        ],
        value_names=[
            "t" if use_inpainting else "f",
        ],
    )

    generator.add_param(
        key="guiding.consistency_loss.enabled",
        name="cls" if not use_inpainting else "",
        group=0,
        values=[
            (not use_inpainting),
        ],
        value_names=[
            "f" if not use_inpainting else "t",
        ],
    )

    generator.add_param(
        key="guiding.timestep_start",
        name="ts",
        group=1,
        values=[20, 60, 100],
    )

    generator.add_param(
        key="guiding.ddim_eta",
        name="eta",
        group=2,
        values=[0, 1], 
    )

    if use_inpainting:
        generator.add_param(
            key="guiding.inpainting.n_resample",
            name="nres",
            group=1002,
            values=[0, 1, 5, 10],
        )

    else:
        generator.add_param(
            key="guiding.consistency_loss.n_resample",
            name="nres",
            group=1002,
            values=[0, 1, 5],
        )

        generator.add_param(
            key="guiding.consistency_loss.loss_type",
            name="loss",
            group=1003,
            values=["mse", "weighted_mse"],
        )

        generator.add_param(
            key="guiding.consistency_loss.weight",
            name="wg",
            group=1004,
            values=[10, 100, 1e3],
        )

        generator.add_param(
             key = "guiding.consistency_loss.N_sample_monte_carlo",
             name="nsmc",
             group=1005,
             values=[1, 4, 4, 16]
        )

        generator.add_param(
             key = "guiding.consistency_loss.std_monte_carlo",
             name="stdmc",
             group=1005,
             values=[0, 0.5, 1, 1]
        )


    return generator

def main(args):

    # make config generator
    generator = make_generator_cfg(
      config_file=args.config, # base config file from step 1
      script_file=args.script  # explained later in step 4
    )

    # generate jsons and script
    generator.generate_custom(args.process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to base json config - will override any defaults.
    parser.add_argument(
        "--config",
        type=str,
        default = "/home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/square_cfg/base_config_square_cfg.json",
        help="path to base config json that will be modified to generate jsons. The jsons will\
            be generated in the same folder as this file.",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--script",
        type=str,
        help="path to output script that contains commands to run the generated training runs",
        default = "/home/wjung85/Repo/projects/FastIL/robomimic/dev/hparam/config/square_cfg/out_inp.sh",
    )

    # Script name to generate - will override any defaults
    parser.add_argument(
        "--process",
        type=str,
        help="Path to python script that is registered to the bash script",
        default = "/home/wjung85/Repo/projects/FastIL/robomimic/dev/sim/run_trained_agent_rh_hparam_noprecision.py",
    )

    args = parser.parse_args()
    main(args)