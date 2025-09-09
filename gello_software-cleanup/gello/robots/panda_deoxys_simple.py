import time
from typing import Dict
import os
import sys
import pathlib
from collections import namedtuple

import numpy as np
# import pybullet as p


from gello.robots.robot import Robot
from gello.pymodbus_robotiq.robotiq_2f_gripper import Robotiq2FingerGripper

import deoxys
from deoxys.utils import YamlConfig, transform_utils
from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj

# FILE_PATH = pathlib.Path(__file__).parent.absolute()

MAX_OPEN = 0.09


class PandaRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
            self, 
            controller_type,
            gripper_type="franka",
            # robot_ip: str = "100.97.47.74",
        ):

        from deoxys.franka_interface import FrankaInterface

        # self.robot = RobotInterface(
        #     ip_address=robot_ip,
        # )
        # self.gripper = GripperInterface(
        #     ip_address="localhost",
        # )

        print(f"Using config at : {os.path.join(config_root)} charmander.yml")
        self.robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, "charmander.yml"), # TODO check
            control_freq=20,
            state_freq=100,
            control_timeout=1.0,
            has_gripper=True,
            use_visualizer=False,
        )

        self.gripper_type = gripper_type

        if self.gripper_type == "franka":
            self.gripper_interface = None
        elif self.gripper_type == "robotiq":
            self.gripper_interface = Robotiq2FingerGripper()

        self.controller_type = controller_type
        if controller_type == "OSC_POSE":
            # self.controller_cfg_file = os.path.join(config_root, "osc-pose-controller.yml") # TODO check
            self.controller_cfg_file = os.path.join(config_root, "osc-pose-controller-absolute.yml")  # TODO check
        elif controller_type == "JOINT_IMPEDANCE":
            self.controller_cfg_file = os.path.join(config_root, "joint-impedance-controller.yml") # TODO check
        else:
            raise NotImplementedError
        self.controller_cfg = YamlConfig(self.controller_cfg_file).as_easydict()

        self.delta_control = self.controller_cfg["is_delta"]
        
        # Using joint position control to reset robot
        self.reset_controller_type = "JOINT_POSITION"
        self.reset_controller_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()

        # Golden resetting joints
        self.reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        # Resetting positions in Kitchen
        self.reset_joint_positions = [
            -0.0097146,
            -0.53030559,
            0.04198769,
            -2.29382955,
            0.03028904,
            1.73564748,
            0.04139311
        ]


        self.reset() # self.robot.go_home()

        # self.robot.start_joint_impedance()
        # self.gripper.goto(width=MAX_OPEN, speed=255, force=255)
        # time.sleep(1)


    
    def reset(self):

        #TODO move this down below later

        if self.gripper_type == "robotiq":
            self.gripper_interface.grasp(-1)
        self.robot_interface.reset()
        time.sleep(1.0)
        print("restarting the robot interface now")

        action = self.reset_joint_positions + [-1.0]

        assert self.reset_controller_type == "JOINT_POSITION", self.reset_controller_type

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                # logger.info(f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
                # logger.info(f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")
                if (
                    np.max(
                        np.abs(
                            np.array(self.robot_interface._state_buffer[-1].q)
                            - np.array(self.reset_joint_positions)
                        )
                    )
                    < 5e-3
                ):
                    break
            self.robot_interface.control(
                controller_type=self.reset_controller_type,
                action=action,
                controller_cfg=self.reset_controller_cfg,
            )
        # if self.gripper_type == "robotiq":
        #     self.gripper_interface.grasp(-1)

        # We added this sleep here to give the C++ controller time to reset from joint control mode to no control mode
        # to prevent some issues.
        time.sleep(1.0)
        print("RESET DONE")

    def num_dofs(self) -> int:
        """Get the number of joints of `the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """

        joint_positions = self.robot_interface._state_buffer[-1].q
        joint_velocities = self.robot_interface._state_buffer[-1].dq

        # if self.gripper_type == "franka":
        #     gripper_width = self.robot_interface._gripper_state_buffer[-1].width
        # elif self.gripper_type == "robotiq":
        #     gripper_width = self.gripper_interface.get_gripper_act()

        # jointpos = np.append(robot_joints, gripper_width)

        joint_state = {
            "joint_positions": np.array(joint_positions),
            "joint_velocities": np.array(joint_velocities)
        }

        return joint_state

    def get_gripper_state(self):
        if self.gripper_type == "franka":
            gripper_pos = self.robot_interface._gripper_state_buffer[-1].width
        elif self.gripper_type == "robotiq":
            # TODO: 0 is fully open, 1 is fully closed. This is to conform to the DROID convention
            # TODO: maybe change to different variables, one for gripper position and 1 to show absolute width
            gripper_pos = self.gripper_interface.get_gripper_act()

        return np.array(gripper_pos)

    # def get_robot_state(self):


    def step(self, action: np.ndarray) -> tuple:
        """
        Step in the environment with an action

        Args:
            Action (np.ndarray): The action to take in the environment.
        """
        # import torch

        # self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
        # self.gripper.goto(width=(MAX_OPEN * (1 - joint_state[-1])), speed=1, force=1)

        if self.controller_type == "JOINT_IMPEDANCE":
            # assert self.controller_type == "JOINT_IMPEDANCE", self.controller_type

            last_q = np.array(self.robot_interface.last_q)
            joint_traj = joint_interpolation_traj(start_q=last_q, end_q=action[:-1])

            print("command joint state:", action)
            # while True:
            #     if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(joint_state[:-1]))) < 8e-3:
            #         break
            #     self.robot_interface.control(
            #         controller_type=self.reset_controller_type,
            #         action=list(joint_state),
            #         controller_cfg=self.reset_controller_cfg,
            #     )
            # for joint in joint_traj:
            #     action = joint.tolist() + [joint_state[-1]]
            #     self.robot_interface.control(
            #         controller_type=self.controller_type,
            #         action=action,
            #         controller_cfg=self.controller_cfg
            #     )
            if action[-1] < 0.01: # ~0
                action = np.array(list(action[:-1])+[-1.])
            else:
                action = np.array(list(action[:-1])+[0.])
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg
            )
            return (action, )
        elif self.controller_type == "OSC_POSE":
            ### The teleop action will always be an absolute action
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)

            # Assume action is always an absolute pose
            target_pos = action[0:3]
            target_quat = np.array(action[3:-1])
            gripper_act = action[-1]

            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff).tolist()
            action_pos = ((np.array(target_pos) - np.array(current_pos)) * 10).tolist()

            delta_action = action_pos + axis_angle_diff + [gripper_act]

            if self.delta_control:

                action = delta_action

            else:
                # If we use absolute control, we just need to change the rotation action from quat to axis angle
                target_pos = action[0:3]
                target_quat = np.array(action[3:-1])
                target_axis_angle = transform_utils.quat2axisangle(target_quat).tolist()
                gripper_act = action[-1]
                # action = target_pos + target_axis_angle + [gripper_act]
                action = np.concatenate([target_pos, target_axis_angle, [gripper_act]])

            # Call interface to command robot
            if self.gripper_type == "franka":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
            elif self.gripper_type == "robotiq":
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg
                )
                robotiq_grasp_act = 2 * gripper_act + 1  # [-1,1]
                self.gripper_interface.grasp(robotiq_grasp_act)
            else:
                raise NotImplementedError

            return (delta_action, action)

        print("DONE COMMAND")
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """
        This robot will not implement this method
        Args:
            joint_state:

        Returns:

        """
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()

        current_pose = self.robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]

        eef_pos = np.squeeze(np.array(current_pos))
        eef_quat = np.squeeze(np.array(transform_utils.mat2quat(current_rot))[:, None])

        eef_angle = transform_utils.quat2axisangle(np.copy(eef_quat))
        gripper_pos = self.get_gripper_state()
        return {
            "joint_positions": joint_state["joint_positions"],
            "joint_velocities": joint_state["joint_velocities"], # unused?
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_axis_angle": eef_angle,
            "eef_pose": np.array(eef_pos.tolist() + eef_quat.tolist()),
            "gripper_position": gripper_pos,
        }


def main():
    robot = PandaRobot("OSC_POSE", gripper_type="robotiq")
    while True:
        current_joints = robot.get_joint_state()
        # robot.reset()
        # robot.gripper_interface.grasp(1)
        # print(f"Current joints are : {current_joints}")
        obs = robot.get_observations()
        print(obs)
        # time.sleep(3)
        # robot.gripper_interface.grasp(-1)
        #
        # time.sleep(1)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
