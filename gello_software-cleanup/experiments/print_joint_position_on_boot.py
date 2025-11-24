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


cam_dict = {}
# camera_config_dict = {"agentview": {"sn" : "241222076871", "type": "RealSense"},
#                     "wrist": {"sn": 14620168, "type": "Zed"}}
camera_config_dict = {"agentview": {"sn" : "001039114912", "type": "Kinect", 'resize': True, 'resize_resolution': (640, 480)},
# cam_config_dict = {"agentview": {"sn" : "241222076871", "type": "RealSense"},
                    "wrist": {"sn": 14620168, "type": "Zed", 'resize': True, 'resize_resolution': (640, 480)}}



def evaluate():
   
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

    while True:
        obs = env.get_obs()
        print(obs.keys())
        action = obs["current_action"]
        print(action)

    # exit()




if __name__ == "__main__":
    evaluate()