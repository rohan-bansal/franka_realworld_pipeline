#!/usr/bin/env python3


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


# from einops import rearrange
from omegaconf import DictConfig

import sys
sys.path.insert(0, "/mnt/data2/mfm/workspace/Franka/gello_software")

import numpy as np

from gello.robots.panda_deoxys_simple import PandaRobot
from gello.rl2_env import RobotEnv

from atm.policy import *
from atm.utils.train_utils import setup_optimizer
# from atm.utils.process_utils import eef_target_pose2action, action2eef_target_pose, controller, pose2mat
# from robosuite.utils.transform_utils import mat2quat
# from scipy.spatial.transform import Rotation as R
from termcolor import cprint

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

class Rate:
    def __init__(self, rate: float, name: str=None, log_warning=False):
        self.last = time.perf_counter()
        self.rate = rate
        self.dt   = 1.0 / self.rate
        self.name = name
        self.log_warning = log_warning

    def sleep(self) -> None:
        update_rate = 1.0 / (time.perf_counter() - self.last)
        # if self.name=="RL2 Robot Env":
        #     print(f"update rate is: {update_rate}")
        if update_rate < self.rate and self.log_warning:
            cprint(f"Warning: {self.name} update rate is {update_rate}Hz, lower than {self.rate}Hz", "red")
        while (self.last + 1.0 / self.rate) > time.perf_counter():
            time.sleep(0.001)
        self.last = time.perf_counter()


def axisangle2quat(vec):
    angle = np.linalg.norm(vec)

    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q

def obs_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Performs a center crop on the image to make it square.
    """
    h, w = image.shape[:2]
    if h == w:
        return image  # Already square

    short_l = min(h, w)
    
    if h > w:  # Taller image (crop height)
        start_y = (h - short_l) // 2
        end_y = start_y + short_l
        return image[start_y:end_y, :, :]
    else:  # Wider image (crop width)
        start_x = (w - short_l) // 2
        end_x = start_x + short_l
        return image[:, start_x:end_x, :]

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
    action_queue = []

    input_output = []

    exec_horizon_step = 0

    rate = Rate(10.0, name="rollout_rate", log_warning=True)

    # --- ADDED: Flag to track if first frame is saved ---
    first_frame_saved = False

    try:
        while not done and (horizon is None or step_i < horizon):

            # Save initial pose before policy runs
            initial_ee_pos = obs["eef_pos"]
            initial_ee_ori = obs["eef_quat"]
            color_in_depth_frame = obs["agentview_color_in_depth_frame"]

            # output_image_path = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/debug_policy_input_2.png"
            
            agentview_image_cropped = obs_preprocess(obs["agentview_image"])
            wrist_image_cropped = obs_preprocess(obs["wrist_image"])

            agentview_rgb = cv2.resize(agentview_image_cropped, (128,128), interpolation=cv2.INTER_LINEAR)
            wrist_rgb = cv2.resize(wrist_image_cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
            
            # --- ADDED START: Save First Frame ---
            '''
            if not first_frame_saved:
                try:
                    # Using a different name to avoid overwriting your other script's output
                    save_path = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/first_frame_visualization_real_obs.png"
                    
                    # Convert from RGB (from env) to BGR (for cv2)
                    agent_img_bgr = cv2.cvtColor(agentview_rgb, cv2.COLOR_RGB2BGR)
                    wrist_img_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Stack horizontally (agent | wrist)
                    combined_frame = np.hstack((agent_img_bgr, wrist_img_bgr))
                    
                    cv2.imwrite(save_path, combined_frame)
                    print(f"Saved first frame (real obs) visualization to {save_path}")
                    first_frame_saved = True # Set flag so we don't save again
                except Exception as e:
                    cprint(f"Warning: Failed to save first frame. Error: {e}", "red")
            # --- ADDED END: Save First Frame ---
            '''
            views = np.stack([agentview_rgb, wrist_rgb], axis=0) # v,h,w,c
            rgb = np.expand_dims(views, axis=0) # b,v,h,w,c
            task_emb = obs.get("task_emb", None)[np.newaxis, ...]
            extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
            
            for k in extra_states:
                extra_states[k] = extra_states[k][np.newaxis, ...]

            # Get new actions if queue is empty
            if len(action_queue) == 0:

                combined_frame = np.hstack((agentview_rgb, wrist_rgb))
                cv2.imwrite(output_image_path, views)
                
                actions, _tracks = policy.act(rgb, task_emb, extra_states)
                action_queue = list(actions)
                exec_horizon_step = 0
                print(f"Got {len(action_queue)} new abs actions from policy")

            
            abs_action = action_queue[exec_horizon_step]
            abs_action = (action_norms["scale"] * abs_action) + action_norms["offset"]
            abs_action = abs_action.squeeze()
            
            pos = abs_action[:3]
            rot = abs_action[3:6]
            gripper_action = abs_action[6]
            quat = axisangle2quat(rot)

            abs_action = np.concatenate([pos, quat, np.array([gripper_action])])
            
            print(f"Executing action {exec_horizon_step+1}/{len(action_queue)} in horizon")
            print(f"Abs action: {abs_action}")
            print(f"Target position: {pos}")

            if not confirm_first_action:
                input(f"intended action (with quat) [enter to rollout]: {abs_action}")
                confirm_first_action = True

            # Execute and save resulting pose
            obs, act = env.step(abs_action.tolist())
                # actual_pose = pose2mat(obs["eef_pos"], obs["eef_quat"])
                # actual_poses.append(actual_pose)
                
                # Compute and print tracking error
                # pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
                # print(f"Position tracking error: {pos_error:.4f} m")

            exec_horizon_step += 1
            if exec_horizon_step >= len(action_queue):
                action_queue = []
                exec_horizon_step = 0
            
            # save_data = {
            #     "step_i": step_i,
            #     "initial_pose": pose2mat(initial_ee_pos, initial_ee_ori),  # Starting pose for this horizon
            #     "prev_target_poses": prev_target_poses,
            #     "target_poses": target_poses,    # What we commanded (list of 4x4 matrices)
            #     "actual_poses": actual_poses,    # Where we ended up (list of 4x4 matrices)
            #     "use_target_chaining": use_target_chaining,  # Record which method was used
            #     "images": {
            #         "agentview": agentview_rgb,
            #         "wrist": wrist_rgb,
            #         "depth_color": color_in_depth_frame
            #     }
            # }
            # input_output.append(save_data)
            
            step_i += 1

            rate.sleep()

            # time.sleep(0.1)
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

    action_norms = pickle.load(open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/action_norms/action_normalization_stats_full.pkl", "rb"))["actions"]
    
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
        control_rate_hz=100.0,
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

    # Add target chaining flag to config if not present
    
    results = evaluate(fabric, cfg, 
                      checkpoint="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy_new/1106_atm_dp_cotracker_abs_20_demos_2111_seed1/model_2625.ckpt")
    fabric.barrier()


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()