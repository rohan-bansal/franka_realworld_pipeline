# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Obtained from  https://github.com/polymetis/polymetis/python/polymetis/robot_client/robotiq_gripper/third_party/robotiq_2finger_grippers/robotiq_2f_gripper.py
# Which was modified from: https://github.com/Danfoa/robotiq_2finger_grippers/blob/master/robotiq_2f_gripper_control/src/robotiq_2f_gripper_control/robotiq_2f_gripper.py

import serial
from serial.serialutil import SerialException
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import threading
# from pymodbus.client.sync import ModbusSerialClient
# from .robotiq_modbus_rtu import comModbusRtu
# import comModBusRtu 
from gello.pymodbus_robotiq import comModBusRtu

from robomimic.dev.multi_processing.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robomimic.dev.multi_processing.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from robomimic.dev.multi_processing.pose_traj_interpolator import PoseTrajectoryInterpolator
from robomimic.utils.time_utils import precise_wait

from math import ceil

import numpy as np
import array
import time
import enum

ACTION_REQ_IDX = 7
POS_INDEX = 10
SPEED_INDEX = 11
FORCE_INDEX = 12

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    BINARY_CONTROL = 2 # only open and close
    RESET = 3

class Robotiq2FingerGripper(mp.Process):
    """
    No timestamps in the state. In manual, it is recommended to send/read in 200Hz.
    input action: [-1, 1] with -1 for open and 1 for close
    underlying action: [1, 100] with 1 for open and 100 for close
    observation (get_gripper_act): [0, 1] with 0 for open and 1 for close

    NOTE: max speed is 0.15 m/s, i.e. quickest to close is 0.15 m/s. Make sure to sleep the proper time to enable proper execution..
    NOTE2: due to the different unit/transformation in the vel and pos, currently, we send MAX SPEED, and only use pos control
    """
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 device_id=0,
                 stroke=0.085,
                 comport="/dev/ttyUSB0",
                 baud=115200,
                 async_execution=True,
                 get_max_k=None,
                 frequency=20.0,
                 max_speed=0.15, # 0.085,
                 force_scale=1,
                 command_queue_size=1024,
                 receive_latency=0.0,
                 dead_zone = 10, # 10%
                 verbose=False):

        super().__init__(name="Robotiq2FingerGripper")

        self.client = comModBusRtu.communication()

        connected = self.client.connectToDevice(device=comport)
        if not connected:
            raise Exception(
                "Communication with gripper %d on serial port: %s and baud rate: %d not achieved"
                % (device_id, comport, baud)
            )

        self.init_success = True
        self.device_id = device_id + 9
        self.stroke = stroke # step size of the gripper?
        self.move_max_speed = max_speed # max speed at m/s
        self.max_speed_percentage = 100 # gripper pos action [1,100], how fast we can change at 1s
        self.max_force = force_scale # 75 destroyed the coffee pod
        self.frequency = frequency
        self.receive_latency = receive_latency
        self.initialize_communication_variables()
        self.dead_zone = dead_zone
        self.last_action = 1 # open by default

        self.message = []

        self.async_execution = async_execution
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    def _update_cmd(self):

        # Initiate command as an empty list
        self.message = []
        # Build the command with each output variable
        self.message.append(self.rACT + (self.rGTO << 3) + (self.rATR << 4))
        self.message.append(0)
        self.message.append(0)
        self.message.append(self.rPR)
        self.message.append(self.rSP)
        self.message.append(self.rFR)

    def sendCommand(self):
        """Send the command to the Gripper."""
        return self.client.sendCommand(self.message)

    def getStatus(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""

        # Acquire status from the Gripper
        status = self.client.getStatus(6)

        # Check if read was successful
        if status is None:
            return False

        # Assign the values to their respective variables
        self.gACT = (status[0] >> 0) & 0x01 # activation
        self.gGTO = (status[0] >> 3) & 0x01 # go to position
        self.gSTA = (status[0] >> 4) & 0x03 # state
        self.gOBJ = (status[0] >> 6) & 0x03 # object detection
        self.gFLT = status[2] # fault
        self.gPR = status[3] # requested position
        self.gPO = status[4] # observed position
        self.gCU = status[5] # value between 0x00 and 0xFF, approximate current equivalent is 10*value read in mA

        return True

    def initialize_communication_variables(self):
        # Out
        self.rPR = 0
        self.rSP = 255
        self.rFR = 150
        self.rARD = 1
        self.rATR = 0
        self.rGTO = 0
        self.rACT = 0
        # In
        self.gSTA = 0
        self.gACT = 0
        self.gGTO = 0
        self.gOBJ = 0
        self.gFLT = 0
        self.gPO = 0
        self.gPR = 0
        self.gCU = 0

        self._update_cmd()
        self._max_force = 100.0  # [%]

    def activate_gripper(self):
        self.rACT = 1
        self.rPR = 0
        self.rSP = 255
        self.rFR = 150
        self._update_cmd()

    def deactivate_gripper(self):
        self.rACT = 0
        self._update_cmd()

    def activate_emergency_release(self, open_gripper=True):
        self.rATR = 1
        self.rARD = 1

        if open_gripper:
            self.rARD = 0
        self._update_cmd()

    def deactivate_emergency_release(self):
        self.rATR = 0
        self._update_cmd()

    def goto(self, pos, vel, force):
        """
        Sets the command to send the gripper to a position with a desired speed and force
        Args:
            pos: in range [1, 100], 100 being closed
            vel: in range [0, 0.085]
            force: in range [1, 100]
        """
        assert 1 <= pos <= 100
        # assert 0 <= vel <= 0.085
        assert 0 <= vel <= 0.150
        assert 1 <= force <= 100
        self.rACT = 1
        self.rGTO = 1
        # TODO: check and fix
        self.rPR = int(np.clip(( -255.0) / (self.stroke * pos) + 255.0, 0, 255))
        self.rSP = int(np.clip(255.0 / (0.1 - 0.013) * vel - 0.013, 0, 255))
        self.rFR = int(np.clip(255.0 / (self._max_force) * force, 0, 255))
        # print(f"rPR: {self.rPR}, rSP: {self.rSP}, rFR: {self.rFR}")
        self._update_cmd()

    def stop_motion(self):
        self.rACT = 1
        self.rGTO = 0
        self._update_cmd()

    def is_ready(self):
        return self.gSTA == 3 and self.gACT == 1

    def is_reset(self):
        return self.gSTA == 0 or self.gACT == 0

    def is_moving(self):
        return self.gGTO == 1 and self.gOBJ == 0

    def is_stopped(self):
        return self.gOBJ != 0

    def object_detected(self):
        return self.gOBJ == 1 or self.gOBJ == 2

    def get_fault_status(self):
        return self.gFLT

    def get_pos(self):
        # return the open value in meters
        po = float(self.gPO)
        return np.clip(self.stroke / (3.0 - 230.0) * (po - 230.0), 0, self.stroke)
    
    def get_gripper_act(self):
        # return value between 0 and 1
        assert self.getStatus()
        pos = self.get_pos()
        return 1 - (pos / self.stroke)

    def get_req_pos(self):
        pr = float(self.gPR)
        return np.clip(self.stroke / (3.0 - 230.0) * (pr - 230.0), 0, self.stroke)

    def get_current(self):
        return self.gCU * 0.1

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float, rescale=False):
        """
        pos: desired position action in [1, 100], 100 for closed
        rescale: if True, assuming input has range [-1, 1] (from NN), rescale the position to [1, 100]
        """
        if rescale:
            pos = np.clip((pos + 1) * 50, 1, 100)

        # print(f"=============pos: {pos}, target_time: {target_time}")
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def reset(self):
        self.input_queue.put({
            'cmd': Command.RESET.value,
            'target_time': 0.0
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

 # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            print("Activating gripper...")
            self.activate_emergency_release()
            self.sendCommand()
            time.sleep(0.5)
            self.deactivate_emergency_release()
            self.sendCommand()
            time.sleep(0.5)
            self.activate_gripper()
            self.sendCommand()
            time.sleep(2.5)
            print("Gripper activated")

            # get initial
            curr_pos = self.get_gripper_act() # [0,1] 0 for closed, 1 for open
            curr_pos_remap = 1+ curr_pos * 99 # action in [1, 100]
            print(f"Initial position observed (get_gripper_act): {curr_pos}")
            # curr_pos = 100.0
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos_remap,0,0,0,0,0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0] # in percentage [1, 100]
                # print(f"target_pos for control {target_pos}")
                # target_vel = (target_pos - pose_interp(t_target - dt)[0]) * self.stroke / dt # in m/s
                target_pos = np.clip(target_pos, 1, 100)
                # target_vel = np.clip(np.abs(target_vel), 0, self.move_max_speed)
                # NOTE: hack, we only use pos control for now. the vel control is so dumb...
                target_vel = self.move_max_speed

                # Hack to bang-bang control with open and close only
                # if target_pos > 60:
                #     target_pos = 100
                # elif target_pos < 60:
                #     target_pos = 1
                # else:
                #     target_pos = self.last_action
                # self.last_action = target_pos

                self.goto(target_pos, target_vel, self.max_force) # update the command

                # execute commands
                if self.async_execution:
                    thread = threading.Thread(target=self.sendCommand)
                    thread.start()
                else:
                    self.sendCommand()
                # print(f"execution time {time.monotonic() - t_now}")

                # get state from robot
                #  NOTE: need debug, gripper_state will stay unchanged.
                state = {
                    'gripper_state': self.get_gripper_act(), # [0,1] as input for policy
                    'gripper_position': self.get_pos(), # in meters
                    # 'gripper_velocity': info['velocity'] / self.scale,
                    # 'gripper_force': info['force_motor'],
                    # 'gripper_measure_timestamp': info['measure_timestamp'],
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # put commands into the interpolator
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() + (target_time - time.time())
                        curr_time = t_now

                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            # max_pos_speed=self.max_speed_percentage,
                            # max_rot_speed=self.max_speed_percentage,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESET.value:
                        curr_time = t_now
                        target_time = curr_time + 1.0
                        target_pos = 1.1
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            # max_pos_speed=self.max_speed_percentage,
                            # max_rot_speed=self.max_speed_percentage,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                        time.sleep(1.1)

                        t_start = time.monotonic()
                        iter_idx = 0
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"Robotiq Gripper Disconnected from robot")

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Robotiq Controller] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        self.stop_motion()
        # stop the process loop
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
    
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(2.0)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context manager")
        self.stop()
    

if __name__ == "__main__":

    with SharedMemoryManager() as shm_manager:
        gripper = Robotiq2FingerGripper(shm_manager=shm_manager, get_max_k=10, async_execution=True, verbose=True)
        if not gripper.init_success:
            raise Exception(f"Unable to init gripper")
        if not gripper.getStatus():
            raise Exception("Failed to contact gripper")

        gripper.start()
        time.sleep(3)
        print(f"Process is alive: {gripper.is_alive()}")
        input("Press Enter to start...")

        gripper.schedule_waypoint(-0.5, time.time() + 0.5, rescale=True) # -1 for open
        time.sleep(1.5)
        state = gripper.get_state(k=1) 
        print(f"closed state after first execution: {state}")

        input("Close")
        gripper.schedule_waypoint(1.0, time.time() + 0.2, rescale=True) # 1.0 for close
        time.sleep(1.5)
        state = gripper.get_state(k=1)
        print(f"open state after first execution: {state}")

        input("Open")
        gripper.schedule_waypoint(-1.0, time.time() + 0.1, rescale=True) # -1.0 for open
        time.sleep(1.5)
        state = gripper.get_state(k=1)
        print(f"open state after first execution: {state}")

        gripper.stop()

    print("Process stopped")