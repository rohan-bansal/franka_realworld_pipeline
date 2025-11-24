import hydra
import math
from glob import glob
import pandas as pd
import torch
import time

import datetime
import torch
torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import torch.distributed as dist
import lightning
from lightning.fabric import Fabric
import gc
import pickle
import os
import cv2

from einops import rearrange
from omegaconf import DictConfig

import sys
sys.path.insert(0, "/mnt/data2/mfm/workspace/Franka/gello_software")

import numpy as np

from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv

from atm.policy import *
from atm.utils.train_utils import setup_optimizer
from atm.utils.process_utils import eef_target_pose2action, action2eef_target_pose, controller, pose2mat
from robosuite.utils.transform_utils import mat2quat
#from atm.utils.env_utils import build_env
from scipy.spatial.transform import Rotation as R

from collections import defaultdict
import pickle
torch.set_float32_matmul_precision('medium')

obs_key_mapping = {
    "gripper_states": "gripper_position",
    "joint_states": "joint_positions",
    "ee_pos": "eef_pos",
    "ee_ori": "eef_quat"
}

cam_dict = {}

camera_config_dict = {"agentview": {"sn" : "001039114912", "type": "Kinect"},
                    "wrist": {"sn": 14620168, "type": "Zed", 'resize': True, 'resize_resolution': (640, 576)}}

@torch.no_grad()
def rollout(env, policy, horizon=None, action_norms=None):
    
    input("continue with rollout? enter to continue...")

    policy.eval()

    env._robot.reset()
    policy.reset()
    done = False
    step_i = 0

    obs = env.get_obs()

    confirm_first_action = False
    delta_action_queue = []

    input_output = []


    prev_quat = None

    try:
        while not done and (horizon is None or step_i < horizon):

            predict_delta_actions = []
            predict_poses = []
            executed_actions = []

            agentview_rgb = cv2.resize(obs["agentview_image"], (128,128), interpolation=cv2.INTER_LINEAR)
            wrist_rgb = cv2.resize(obs["wrist_image"], (128, 128), interpolation=cv2.INTER_LINEAR)
            views = np.stack([agentview_rgb, wrist_rgb], axis=0) # v,h,w,c
            rgb = np.expand_dims(views, axis=0) # b,v,h,w,c
            task_emb = obs.get("task_emb", None)[np.newaxis, ...]
            extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
            
            for k in extra_states:
                extra_states[k] = extra_states[k][np.newaxis, ...]

            
            # Get new delta actions if queue is empty
            if len(delta_action_queue) == 0:
                delta_actions, _tracks = policy.act(rgb, task_emb, extra_states)
                delta_action_queue = list(delta_actions)
                predict_delta_actions = delta_actions
                print(f"Got {len(delta_action_queue)} new delta actions from policy")

            # Execute actions one at a time, converting each delta to absolute using current pose
            for i in range(len(delta_action_queue)):
                # Get current pose from current observation
                ee_pos = obs["eef_pos"]
                ee_ori = obs["eef_quat"]
                color_in_depth_frame = obs["agentview_color_in_depth_frame"]
                curr_pose = pose2mat(ee_pos, ee_ori)

                # Pop next delta action from queue
                delta_action = delta_action_queue[i]

                # Convert delta to absolute action using CURRENT actual pose
                target_pose, gripper_action = action2eef_target_pose(curr_pose, delta_action, controller)
                predict_poses.append(target_pose)
                pos = target_pose[:3, 3]
                rot = target_pose[:3, :3]

                quat = mat2quat(rot)
                if prev_quat is not None:
                    if np.dot(quat, prev_quat) < 0.0:
                        quat = -quat
                prev_quat = quat

                abs_action = np.concatenate([pos, quat, np.array(gripper_action)])
                executed_actions.append(abs_action)

                print(f"Executing action {i+1}/{len(delta_action_queue)} in horizon")

                if not confirm_first_action:
                    input(f"intended action (with quat) [enter to rollout]: {abs_action}")
                    confirm_first_action = True

                # Execute action and get NEW observation
                obs, act = env.step(abs_action.tolist())

            # Clear the queue after executing all actions
            delta_action_queue = []

            save_data = {
                "step_i": step_i,
                "input" : {
                    "agentview_rgb": agentview_rgb,
                    "wrist_rgb": wrist_rgb,
                    "agentview_wrist_concat": rgb,
                    "task_emb": task_emb,
                    "extra_states": extra_states,
                    "ee_ori": ee_ori,
                    "ee_pos": ee_pos,
                    "pc_color": color_in_depth_frame
                },
                "output" : {
                    "predicted_target_poses": predict_poses,
                    "predicted_delta_actions": predict_delta_actions,
                    "predicted_absolute_actions": executed_actions
                }
            }
            input_output.append(save_data)
            
            step_i += 1

            time.sleep(0.1)
    except KeyboardInterrupt:
        with open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data.pkl", 'wb') as f:
            pickle.dump(input_output, f)

        print("data saved")

        sys.exit()

    return



def evaluate(fabric, cfg, checkpoint):
    cfg.model_cfg.load_path = checkpoint
    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)

    cfg.optimizer_cfg.params.lr = 0.
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)

    model, optimizer = fabric.setup(model, optimizer)
    if hasattr(model, "setup_ema"):
        model.setup_ema()
    # model.mark_forward_method('act')

    rollout_horizon = cfg.env_cfg.get("horizon", None)

    all_results = []

    action_norms = pickle.load(open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/action_norms/action_normalization_stats_sep20.pkl", "rb"))["actions"]
    
    print("creating real world environment")

    if camera_config_dict is not None:
        for cam in camera_config_dict:
            # cam_config_dict is {"camera_name" : {"sn": int or str, type: "Zed", "RealSense" or "Kinect" +
            #                                                   camera-specific configs}
            cam_config = camera_config_dict[cam]
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam] = ZedCamera(cam, cam_config)
            elif cam_config["type"] == "RealSense":
                from  gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'], enable_depth=True)
            elif cam_config["type"] == "Kinect":
                from gello.cameras.kinect_camera import KinectCamera
                cam_dict[cam] = KinectCamera(cam, cam_config)
    print("initialized cameras")

    robot_client = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(
        robot_client,
        camera_dict=cam_dict,
        control_rate_hz=30.0,
        save_depth_obs=True
    )
    print("initialized robot env, test obs:")

    obs = env.get_obs()
    for x in obs.keys():
        print(x, obs[x].shape)

    # exit()

    result = rollout(env, model, horizon=rollout_horizon, action_norms=action_norms)

    del env
    del model
    del optimizer
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()

    return all_results

@hydra.main(version_base="1.3")
def main(cfg: DictConfig):
    save_path = cfg.save_path
    result_suffix = cfg.get("result_path_suffix", "")
    result_suffix = f"_{result_suffix}" if result_suffix else result_suffix

    eval_result_dir = os.path.join(save_path, f"eval_results{result_suffix}")
    os.makedirs(eval_result_dir, exist_ok=True)


    setup(cfg)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
    fabric.launch()

    results = evaluate(fabric, cfg, checkpoint="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy/1001_atm_dp_cotracker_delta_actions_50_demos_0222_seed1/model_1600.ckpt")
    fabric.barrier()


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()