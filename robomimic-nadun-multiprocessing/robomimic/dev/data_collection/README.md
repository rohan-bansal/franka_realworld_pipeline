# Environment Generation + Dynamic Frequency Testing

## Prerequisites

### General Env Prereqs

1. Create and activate a conda environment
2. Use this version of robosuite v1.4.1: https://github.com/rohan-bansal/robosuite/tree/v1.4.1
3. Check out this branch of robomimic-nadun (dynamic-ctrl-freq): https://github.com/nadunRanawaka1/robomimic-nadun/tree/dynamic-ctrl-freq
   1. If you see this README you are good! (in case branch merged down the line)

### Dynamic Frequency Prereqs

1. `dynamic_speed.py` currently utilizes absolute actions for testing, so ensure input dataset has these added.

## Environment Setup for Demo Collection

Integrated procedural environment generation from [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

1. See `collect_bddl_demo.py`. Working command I have tested:
   1. `python collect_demo.py --controller OSC_POSE --camera agentview --robots Panda --num-demonstration 50 --bddl-file <file.bddl> --device keyboard`
2. If you don't want to use this script, see lines 264-309 in above file for how environment is created. You will still need the BDDL environment config file as input.
3. Config file I created for picking up multiple objects and placing in single bin is `bddl_files/pick_place_all_objects.bddl`
4. To see examples of additional possible BDDL files and configurations, look [here](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files).


## Dynamic Frequency Testing

Generates side-by-side video of motion profile graph and robot sim.

1. In `dynamic_speed.py`
   2. Set `demo_fn`, `video_fn` file paths
   3. Modify the motion profile to your liking (will change HZ at different "checkpoints" in action set)
      1. Can view full documentation on how to use in `robosuite/utils/motion_profile.py`
2. Run with `python dynamic_speed.py`

## Troubleshooting

1. If you receive errors about missing packages, I probably just installed them. You can compare the output of `conda list` and `pip freeze` with my outputs in the `package-lists` folder.

## Questions

Please reach out to Rohan Bansal on RL2 Slack (rohanbansal@gatech.edu)