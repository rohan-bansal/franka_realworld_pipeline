import os


Ta_LIST = [i for i in range(8, 65, 8)]


fp = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/sbatch_scripts/diffusion_policy/sim/action_horizon_experiments/joint_position/can_image.sh"

for length in Ta_LIST:
    with open(fp, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):

        # Replace job id and stuff
        line = line.replace("action_horizon_length", f"action_horizon_{length}")
        lines[i] = line

        # Replace config
        line = line.replace("file.json", f"can_image_action_horizon_{length}.json")
        lines[i] = line
    new_fp = fp.replace("can_image.sh", f"can_image_{length}.sh")

    with open(new_fp, 'w') as file:
        file.writelines(lines)