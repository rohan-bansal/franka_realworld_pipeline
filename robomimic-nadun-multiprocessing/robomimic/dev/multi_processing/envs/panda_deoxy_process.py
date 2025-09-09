# ... existing imports ...
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import matplotlib.pyplot as plt
from nexusformat.nexus.plot import label

from robomimic.dev.multi_processing.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robomimic.dev.multi_processing.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from robomimic.dev.multi_processing.pose_traj_interpolator import PoseTrajectoryInterpolator
from robomimic.utils.time_utils import precise_wait
from robomimic.dev.multi_processing.common import ObsAccumulator
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.transform_utils as TransUtils

from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys import config_root

import enum
import numpy as np
import time
import os

# Dictionary storing different reset joint positions for the Panda robot
RESET_JOINT_POSITIONS = {
    "golden": [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.30396583422025, 0.8480939705504309],
    "kitchen": [-0.0097146, -0.53030559, 0.04198769, -2.29382955, 0.03028904, 1.73564748, 0.04139311],
    "pick_and_place": [0.0007925123372213324, -0.16362003257456625, 0.14518421694487627, -2.5930469890684105, 0.02467854399933535, 2.4951139314723902, 0.21047918180756753]
}

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET = 3

class PandaRobot(mp.Process):
    """
    This class contains the Panda robot interface and the controller.
    """
    def __init__(self,
            shm_manager: SharedMemoryManager,
            controller_type,
            control_rate=20.0,
            receive_latency=3.75e-3,
            get_max_k=128,
            reset_position="pick_and_place",
            verbose=False):
        
        super().__init__(name="PandaRobotController")
        
        # Store init params
        self.controller_type = controller_type
        self.control_rate = control_rate
        self.receive_latency = receive_latency
        self.verbose = verbose
        
        # Setup controller configs
        if controller_type == "OSC_POSE":
            self.controller_cfg_file = os.path.join(config_root, "osc-pose-controller.yml")
            self.action_dim = 12 # with vel tracking
        elif controller_type == "JOINT_IMPEDANCE":
            raise NotImplementedError("Interpolator for joint impedance controller not implemented")
            self.controller_cfg_file = os.path.join(config_root, "joint-impedance-controller.yml")
            self.action_dim = 7
        else:
            raise NotImplementedError
        self.controller_cfg = YamlConfig(self.controller_cfg_file).as_easydict()
        print(f"controller cfg {self.controller_cfg}")

        # Using joint position control to reset robot
        self.reset_controller_type = "JOINT_POSITION"
        self.reset_controller_cfg = YamlConfig(os.path.join(config_root, "joint-position-controller.yml")).as_easydict()
        if reset_position not in RESET_JOINT_POSITIONS:
            raise ValueError(f"Reset position {reset_position} not found in RESET_JOINT_POSITIONS")
        self.reset_position = RESET_JOINT_POSITIONS[reset_position]
        
        # Build input queue for commands
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((self.action_dim,), dtype=np.float64),
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # Build ring buffer for robot state Create example state dict for ring buffer
        example = {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'joint_positions_desired': np.zeros(7),
            'joint_velocities_desired': np.zeros(7),
            'ee_twist_desired': np.zeros(6),
            'ee_pose_desired': np.zeros(16),
            'eef_pos': np.zeros(3),
            'eef_quat': np.zeros(4),
            'eef_axis_angle': np.zeros(3),
            'eef_pose': np.zeros(6), # 3 pos + 3 axis angle,
            'robot_received_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=control_rate
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        # self.obs_accumulator = None
        # self.obs_list = None
        # self.receive_keys = receive_keys

    # ====== Robot user interface methods ======
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.control_rate))
        pose = np.array(pose)
        assert pose.shape == (12,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (12,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def reset(self, time_out=30.0):
        self.ready_event.clear()
        pose = np.zeros(self.action_dim)
        target_time = 0.0
        message = {
            'cmd': Command.RESET.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

        # wait until reset is done
        t_start = time.time()
        while not self.ready_event.is_set():
            time.sleep(0.1)
            if time.time() - t_start > time_out:
                return False
        return True

    #===== Internal Robot interface methods ===== #
    @staticmethod
    def get_pos_quat(robot_interface):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]

        eef_pos = np.squeeze(np.array(current_pos))
        eef_quat = np.squeeze(np.array(transform_utils.mat2quat(current_rot))[:, None])
        return np.concatenate([eef_pos, eef_quat], axis=0)

    @staticmethod
    def get_pos_axis_angle(robot_interface):
        eef_pos_quat = PandaRobot.get_pos_quat(robot_interface)
        eef_pos, eef_quat = eef_pos_quat[:3], eef_pos_quat[3:]
        eef_axis_angle = transform_utils.quat2axisangle(np.copy(eef_quat))
        return np.concatenate([eef_pos, eef_axis_angle], axis=0)

    @staticmethod
    def get_joint_state(robot_interface):
        """Get the current state of the leader robot.
        Returns:
            T: The current state of the leader robot.
        """
        joint_positions = robot_interface._state_buffer[-1].q
        joint_velocities = robot_interface._state_buffer[-1].dq
        joint_positions_desired = robot_interface._state_buffer[-1].q_d
        joint_velocities_desired = robot_interface._state_buffer[-1].dq_d

        joint_state = {
            "joint_positions": np.array(joint_positions),
            "joint_velocities": np.array(joint_velocities),
            "joint_positions_desired": np.array(joint_positions_desired),
            "joint_velocities_desired": np.array(joint_velocities_desired)
        }
        return joint_state

    @staticmethod
    def get_observations(robot_interface):
        joint_state = PandaRobot.get_joint_state(robot_interface)

        ee_twist_desired = np.array(robot_interface._state_buffer[-1].O_dP_EE_d)
        ee_pose_desired = np.array(robot_interface._state_buffer[-1].O_T_EE_d) # 16D for 4x4 transformation matrix

        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]

        eef_pos = np.squeeze(np.array(current_pos))
        eef_quat = np.squeeze(np.array(transform_utils.mat2quat(current_rot))[:, None])
        eef_angle = transform_utils.quat2axisangle(np.copy(eef_quat))

        return {
            "joint_positions": joint_state["joint_positions"],
            "joint_velocities": joint_state["joint_velocities"], # unused?
            "joint_positions_desired": joint_state["joint_positions_desired"],
            "joint_velocities_desired": joint_state["joint_velocities_desired"],

            "ee_twist_desired": ee_twist_desired,
            "ee_pose_desired": ee_pose_desired,

            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_axis_angle": eef_angle,
            "eef_pose": np.array(eef_pos.tolist() + eef_angle.tolist()),
        }
    
    def reset_interface(self, robot_interface):
        robot_interface.reset()
        action = self.reset_position
        assert self.reset_controller_type == "JOINT_POSITION", self.reset_controller_type

        while True:
            if len(robot_interface._state_buffer) > 0:
                if (
                    np.max(
                        np.abs(
                            np.array(robot_interface._state_buffer[-1].q)
                            - np.array(self.reset_position)
                        )
                    )
                    < 2e-3
                ):
                    break
            robot_interface.control(
                controller_type=self.reset_controller_type,
                action=action,
                controller_cfg=self.reset_controller_cfg,
            )
        time.sleep(1.0)
        print("PandaRobot: RESET DONE")

    # Add process management methods
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PandaRobot] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(2.0)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    # def start_recording(self):
    #     self.obs_list = []
    #     self.obs_accumulator = ObsAccumulator()
    #
    # def stop_recording(self):
    #     ret_obs_list = self.obs_list
    #     self.obs_list = None
    #     return ret_obs_list
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # Add context manager methods
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # Main process loop
    def run(self):
        ### All object below are from the forked process. Variables are not shared to global
        # Initialize robot interface
        robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, "charmander.yml"),
            control_freq=self.control_rate,
            state_freq=100,
            control_timeout=1.0,
            has_gripper=True,
            use_visualizer=False,
        )
        self.reset_interface(robot_interface)

        action_q = []
        curr_vel = np.zeros(6)
            
        try:
            dt = 1.0 / self.control_rate

            # use monotonic time to avoid backward in control loop
            curr_pose = self.get_pos_axis_angle(robot_interface)
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            # TODO: add support for Joint Impedance controller
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # Main control loop, we all use system time here
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot, NOTE: currently only do abs control
                t_now = time.monotonic()
                action = pose_interp(t_now)

                action_vel = np.hstack((action, curr_vel)) # interpolate pose, while keeping same target vel.
                # print(f"action vel in panda deoxy {action_vel}")

                # # Vel support get next action and calculate forward euler velocity
                # # TODO: this not working for interpolator. Specify vel from the policy prediction!
                # # to leverage the interpoation setup (which is not supporing vel) we set vel here. Integrate this to interpolator
                # # next_action = pose_interp(t_now+dt)
                # # xyz_traj = np.stack((action[:3], next_action[:3]))
                # # axis_angle_traj = np.stack((action[3:6], next_action[3:6]))
                #
                # ### To avoid kink in next_action, we estimate the velocity with poses in interpolator
                # if pose_interp.poses.shape[0] == 2:
                #     # xyz_traj = pose_interp.poses[:, :3]
                #     # axis_angle_traj = pose_interp.poses[:, 3:6]
                #     # delta_t = pose_interp.times[1] - pose_interp.times[0]
                #     # linear_vel = AcUtils.compute_velocity_base_frame(xyz_traj, dt=delta_t, max_vel=1.5)
                #     # omega = AcUtils.compute_omega_base_frame(axis_angle_traj, dt=delta_t, max_omega=1.0)
                #     # action_vel = np.concatenate((action, linear_vel, omega))
                #     pass
                # # action_vel = np.concatenate((action, linear_vel, np.zeros(3)))
                # # print(f"act is {action[:6]}  vel is {action[6:]}, delta_t {delta_t}")
                # # action = np.concatenate((action, linear_vel, np.zeros(3)) # only with linear velocity
                # else:
                #     action_vel = np.concatenate((action, np.zeros(6)))
                # # action_q.append(action)
                #
                # action_vel = np.concatenate((action, np.zeros(6)))
                # # print(f"pose in interpolator is {pose_interp.poses}")

                robot_interface.control(
                    controller_type=self.controller_type,
                    action=action_vel,
                    controller_cfg=self.controller_cfg,
                )

                # Get robot state and update ring buffer
                state = self.get_observations(robot_interface)
                t_recv = time.time()
                state['robot_received_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)
                # self.obs_accumulator.put(state, time.time())
                # if self.obs_list is not None:
                #     self.obs_list.append(state)

                # fetch new commands
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose[..., :6],
                            time=t_insert,
                            curr_time=curr_time,
                        )

                        if target_pose.shape == (12,):
                            curr_vel = target_pose[...,6:]

                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        # print(f"target pose {target_pose}, tar-curr time {target_time-curr_time}")
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose[..., :6],
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )

                        if target_pose.shape == (12,):
                            curr_vel = target_pose[..., 6:]

                        last_waypoint_time = target_time
                    elif cmd == Command.RESET.value:
                        # reset robot with joint controller
                        self.reset_interface(robot_interface)
                        # reset the interpolator to avoid wrong command sent
                        curr_pose = self.get_pos_axis_angle(robot_interface)
                        curr_t = time.monotonic()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[curr_t],
                            poses=[curr_pose]
                        )
                        self.input_queue.clear()  # clear old message in the queue
                        self.ready_event.set()
                        if self.verbose:
                            print(f"[FrankaPositionalController] has been reset to initial position, ready {self.ready_event.is_set()}")
                    else:
                        keep_running = False
                        break
                        
                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # if self.verbose:
                #     print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")
                    
        finally:
            # Terminate Cleanup
            print("[FrankaPositionalController] terminating the robot interface")

            del robot_interface
            self.ready_event.set()

if __name__=="__main__":

    import h5py
    import json

    rollout_demo = "/home/mbronars/zhenyang/demos/pick_cube_1105_30demos/pick_cube_1106/pick_cube_1106_demo_remove_idle.hdf5"
    demo_file = h5py.File(rollout_demo, 'r')['data']
    demo = demo_file['demo_0']
    actions = demo['absolute_actions'][:]
    horizon = actions.shape[0]
    controller_type = "OSC_POSE"
    save = False

    policy_cmd_rate = 40 # how fast we get cmd from policy. Default to 20hz
    # NOTE: higher control rate for process control can lead to smaller position error
    process_control_rate = 100 # how fast we send out interpolated command in the process, suggest 100hz

    with SharedMemoryManager() as shm_manager:
        robot = PandaRobot(shm_manager=shm_manager, controller_type=controller_type, control_rate=process_control_rate,verbose=False)

        robot.start()

        dt = 1 / policy_cmd_rate
        error = []
        rot_error = []
        data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}

        key = input("start demo replaying? (y/n)")
        if key=='y':
            for i in range(horizon):
                action = actions[i, :6]
                # print(f"action is {action}")
                robot.schedule_waypoint(action, dt+time.time()) # NOTE: important to align the time correctly!!
                obs = robot.get_state(k=1)
                # print(f"timet at {obs['robot_timestamp']} error is {action - obs['eef_pose']}")
                error.append(np.squeeze(np.abs(action - obs['eef_pose']))[:3])
                rot_error.append(TransUtils.geodesic_distance(action[3:6], obs['eef_pose'][3:6]))

                data['action'].append(action)
                data['ee_states'].append(obs['eef_pose'])
                data['joint_states'].append(obs['joint_positions'])
                time.sleep(dt)

            avg_pos_error = np.mean(np.array(error), axis=0)
            avg_rot_error = np.mean(np.abs(np.array(rot_error)))
            print(f"avg position error {avg_pos_error}")
            print(f"avg rot geodesic error {avg_rot_error}")
        robot.stop(wait=True)

    if not save:
        print("Not saving the trajectory")
    else:
        folder = "/home/mbronars/zhenyang/demos/replay"
        # controller_type = config["controller_type"]
        base_name = os.path.basename(rollout_demo)
        save_path = f"{folder}/{policy_cmd_rate}HZ_replay_{base_name}_Kp_150_250_0115.hdf5"
        with h5py.File(save_path, "w") as h5py_file:
            config_dict = {
                # "controller_cfg": EasyDict(config["controller_cfg"]),
                "controller_type": controller_type,
            }
            grp = h5py_file.create_group("data")
            grp.attrs["config"] = json.dumps(config_dict)

            grp.create_dataset("actions", data=np.array(data["action"]))
            grp.create_dataset("ee_states", data=np.array(data["ee_states"]))
            grp.create_dataset("joint_states", data=np.array(data["joint_states"]))
            # grp.create_dataset("gripper_states", data=np.array(data["gripper_states"]))
            grp.create_dataset("avr_pos_error", data=avg_pos_error)
            grp.create_dataset("avr_rot_error", data=avg_rot_error)
            print(f"Finish replay trajectory saving at {save_path}")