import h5py
import time
import numpy as np
import argparse
from gello.rl2_env import RobotEnv
from gello.robots.panda_deoxys_simple import PandaRobot
from deoxys.utils import transform_utils
from termcolor import cprint

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

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def load_action_sequence(hdf5_path):
    print(f"Loading demo from: {hdf5_path}")
    actions = []

    with h5py.File(hdf5_path, "r") as f:
        demo_group = f["data/demo_0"]
        chunks = sorted([k for k in demo_group.keys() if k.startswith("chunk_")])

        for chunk_name in chunks:
            chunk = demo_group[chunk_name]
            abs_actions = chunk["action_absolute"][()]
            actions.append(abs_actions)
        # actions = demo_group["actions"][()]

    actions = np.concatenate(actions, axis=0)
    print(f"Loaded {len(actions)} actions")
    return actions


def main(hdf5_path, delay=0.05, go_home=True):
    # Load actions from file
    actions = load_action_sequence(hdf5_path)
    rate = Rate(10.0, name="rollout_rate", log_warning=True)

    cam_dict = {}
    cam_config_dict = {"agentview": {"sn" : "001039114912", "type": "Kinect", 'resize': True, 'resize_resolution': (640, 480)},
# cam_config_dict = {"agentview": {"sn" : "241222076871", "type": "RealSense"},
                    "wrist": {"sn": 14620168, "type": "Zed", 'resize': True, 'resize_resolution': (640, 480)}}

    if cam_config_dict is not None:
        for cam in cam_config_dict:
            # cam_config_dict is {"camera_name" : {"sn": int or str, type: "Zed", "RealSense" or "Kinect" +
            #                                                   camera-specific configs}
            cam_config = cam_config_dict[cam]
            if cam_config["type"] == "Zed":
                from gello.cameras.zed_camera import ZedCamera
                cam_dict[cam] = ZedCamera(cam, cam_config)
            elif cam_config["type"] == "RealSense":
                from  gello.cameras.realsense_camera import RealSenseCamera
                cam_dict[cam] = RealSenseCamera(device_id=cam_config['sn'])
            elif cam_config["type"] == "Kinect":
                from gello.cameras.kinect_camera import KinectCamera
                cam_dict[cam] = KinectCamera(cam, cam_config)
    # Initialize robot
    print("Initializing robot...")
    robot = PandaRobot("OSC_POSE", gripper_type="robotiq")
    env = RobotEnv(robot, camera_dict=cam_dict)

    # Move to initial position
    if go_home:
        print("Moving to home position...")
        robot.reset()
        time.sleep(2)

    print("Replaying trajectory...")
    obs = env.get_obs()
    for i, action in enumerate(actions):

        input_pos = action[0:3]
        input_aa = np.array(action[3:-1])
        print(action)
        input_quat = transform_utils.axisangle2quat(input_aa)
        gripper_act = action[-1]
        action = np.concatenate([input_pos, input_quat, [gripper_act]])
        # input(f"next act? {action}")
        # print(action)
        obs, _ = env.step(action)
        # time.sleep(delay)
        rate.sleep()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str, help="Path to HDF5 demo file")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (in seconds)")
    parser.add_argument("--no_home", action="store_true", help="Skip home reset before replay")

    args = parser.parse_args()
    main(args.hdf5_path, delay=args.delay, go_home=not args.no_home)
