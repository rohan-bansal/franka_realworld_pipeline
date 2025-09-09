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
#from atm.utils.env_utils import build_env
from scipy.spatial.transform import Rotation as R



torch.set_float32_matmul_precision('medium')

obs_key_mapping = {
    "gripper_states": "gripper_position",
    "joint_states": "joint_positions",
}

cam_dict = {}
camera_config_dict = {"agentview": {"sn" : "241222076871", "type": "RealSense"},
                    "wrist": {"sn": 14620168, "type": "Zed"}}

def axis_angle_to_quat(axis_angle):
    """
    Convert an axis-angle (rotation vector) to a quaternion [w, x, y, z].
    Parameters:
        axis_angle (numpy array): A 3D vector representing axis-angle (rotation vector).
    Returns:
        numpy array: Quaternion in [w, x, y, z] format.
    """
    rot = R.from_rotvec(axis_angle)
    q_scipy = rot.as_quat()  # [x, y, z, w]
    # return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]) # [w, x, y, z]
    return q_scipy


def action_axis_angle_to_quat(action):
    pos = action[0:3]
    rot = action[3:6]
    rot = axis_angle_to_quat(rot)
    gripper = action[6:len(action)]
    return np.concatenate((pos, rot, gripper), axis=0)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    if hasattr(math, "isclose"):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def norm_action_axisangle(action):
    new_action = action.copy()
    rot = new_action[3:6]
    angle = np.linalg.norm(rot)
    if isclose(angle, 0.):
        return np.array([1., 0., 0.]), 0.

    axis = rot / angle
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    rot = axis * angle
    new_action[3:6] = rot
    return new_action

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

    while not done and (horizon is None or step_i < horizon):

        # for x in obs.keys():
        #     print(x, obs[x].shape)

        # print(policy.extra_state_keys)
        
        # TODO print all values and stack agentview/wrist
        agentview_rgb = cv2.resize(obs["agentview_image"], (128,128))
        wrist_rgb = cv2.resize(obs["wrist_image"], (128, 128))
        views = np.stack([agentview_rgb, wrist_rgb], axis=0) # v,h,w,c
        rgb = np.expand_dims(views, axis=0) # b,v,h,w,c
        task_emb = obs.get("task_emb", None)[np.newaxis, ...]
        extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
        for k in extra_states:
            extra_states[k] = extra_states[k][np.newaxis, ...]

        print("rgb shape", rgb.shape)
        print("task emb shape", task_emb.shape)
        print("extra states shape", {k: v.shape for k, v in extra_states.items()})


        a, _tracks = policy.act(rgb, task_emb, extra_states)
        
        print(f"unnormalized action: {a[0]}")
        scaled_action  = np.squeeze((action_norms["scale"] * a[0]) + action_norms["offset"])
        print(f"scaled action: {scaled_action}")
        scaled_action = action_axis_angle_to_quat(scaled_action)

        if not confirm_first_action:
            input(f"intended action (with quat) [enter to rollout]: {scaled_action}")
            confirm_first_action = True

        # scaled_action_with_quat[0] = min(max(0.30, scaled_action_with_quat[0]), 0.75)
        # scaled_action_with_quat[1] = min(max(-0.2, scaled_action_with_quat[1]), 0.30)
        # scaled_action_with_quat[2] = min(max(0.20, scaled_action_with_quat[2]), 0.4)

        obs, act = env.step(scaled_action.tolist())



        step_i += 1

        time.sleep(0.1)

    return



def evaluate(fabric, cfg, checkpoint):
    cfg.model_cfg.load_path = checkpoint
    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)

    cfg.optimizer_cfg.params.lr = 0.
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)

    model, optimizer = fabric.setup(model, optimizer)
    # model.mark_forward_method('act')

    rollout_horizon = cfg.env_cfg.get("horizon", None)

    all_results = []

    action_norms = pickle.load(open("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/action_normalization_stats.pkl", "rb"))["actions"]
    
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
        save_depth_obs=False
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

    results = evaluate(fabric, cfg, checkpoint="/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/checkpoints/policy/0907_atm-policy_pickplace_demo_40_1954_seed1/model_740.ckpt")
    fabric.barrier()


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()