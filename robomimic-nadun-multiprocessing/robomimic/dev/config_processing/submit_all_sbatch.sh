#!/bin/bash

# Set the directory containing the sbatch scripts
DIR="/nethome/nkra3/flash7/phd_project/robomimic-nadun/skynet/sbatch_scripts/diffusion_policy/sim/action_horizon_experiments/joint_position"

# Loop through all .sh files and submit each with sbatch
for script in "$DIR"/*.sh; do
    sbatch "$script"
done
