"""
Camera driver for an Azure Kinect camera built on pyk4a.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

from gello.cameras.camera import CameraDriver
import pyk4a
from pyk4a import Config, connected_device_count
from pyk4a.config import FPS,ImageFormat, DepthMode, ColorResolution
from robomimic.utils.obs_utils import resize_image
from robomimic.dev.multi_processing.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robomimic.dev.multi_processing.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


# example config = {'color_resolution': pyk4a.ColorResolution.RES_720P,
#                       'fps': 30, 'resize': True, 'resize_resolution': (128, 128)}
class KinectCamera(mp.Process, CameraDriver):
    """
    Data struct follows the ringbuffer. Each frames contain
    {image, depth, timestamp}
    NOTE: make sure you check the timestamp when accessing the data (and limit the access rate on upstream task)
    """
    def __repr__(self) -> str:
        return f"KinectCamera(name= {self.cam_name}, serial_number={self.serial_number})"


    def __init__(self, cam_name, config,
                 shm_manager: SharedMemoryManager,
                 depth: bool=False,
                 receive_latency: float=0.09,
                 get_max_k: int=1000,
                 verbose: bool=False):

        super().__init__(name="KinectCamera")
        # Setting some class vars
        self.cam_name = cam_name
        self.serial_number = config['sn']
        self.depth = depth
        self.receive_latency = receive_latency

        # Setting the config for the camera
        pyk4a_config = Config()
        pyk4a_config.camera_fps = config.get('fps', FPS.FPS_30)
        pyk4a_config.color_resolution = config.get('color_resolution', ColorResolution.RES_720P)
        pyk4a_config.color_format = config.get('color_format', ImageFormat.COLOR_BGRA32)
        pyk4a_config.depth_mode = config.get('depth_mode', DepthMode.NFOV_UNBINNED)
        pyk4a_config.depth_resolution = config.get('depth_resolution', (576, 640))
        pyk4a_config.synchronized_images_only = config.get("synchronize_depth", True)
        self.pyk4a_config = pyk4a_config

        # Do we resize the images from the camera?
        self.resize = config.get('resize', False)
        if self.resize:
            self.resize_resolution = config.get('resize_resolution', (128, 128))
        else:
            self.resize_resolution = (1280, 720) # default resolution

        # Create ring buffer
        resolution = tuple(self.resize_resolution)
        image_shape = resolution[::-1]
        examples = {
            'rgb': np.empty(
                shape=image_shape+(3,), dtype=np.uint8),
        }
        if self.depth:
            depth_shape = pyk4a_config.depth_resolution
            examples['depth'] = np.empty(shape=depth_shape+(1,), dtype=np.uint16) # 11 bits
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        put_fps = config.get('fps', FPS.FPS_30)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.05, # NOTE: 0.2 originally, but don't konw why so high..
            put_desired_frequency=put_fps
        )

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        # self.vis_ring_buffer = vis_ring_buffer
        # self.command_queue = command_queue


    @property
    def type(self):
        return "Kinect"

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resize_resolution[::-1]
        data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        # self.video_recorder.start(
        #     shm_manager=self.shm_manager, 
        #     data_example=data_example)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        # self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(2.0)
        # self.video_recorder.start_wait()
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
        # self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    # def get_vis(self, out=None):
    #     return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float=-1):
        pass
        
    def stop_recording(self):
        pass
    
    def restart_put(self, start_time):
        pass

    # ===== interval API ===== #
    def run(self):

        # ===== Start the camera in subprocess ===== #
        cnt = connected_device_count()
        if not cnt:
            print(f"No devices available. Cannot start Kinect with serial number: {self.serial_number}")
            exit()

        camera_id = None
        for device_id in range(cnt):
            device = pyk4a.PyK4A(device_id=device_id)
            if device.opened:
                continue
            else:
                device.open()
                if device.serial == self.serial_number:  # we found our camera_id
                    camera_id = device_id
                device.close()
        if camera_id is None:
            raise Exception(f"Could not find Kinect with serial number: {self.serial_number}")

        camera = pyk4a.PyK4A(device_id=camera_id, config=self.pyk4a_config)
        camera.start()
        print(f"Starting Kinect camera with sn: {self.serial_number} at process {self.pid}")

        try:
            # ====== Main Loop ===== #
            while not self.stop_event.is_set():
                ts = time.time()
                # grab the color and depth images
                capture = camera.get_capture()
                color_image = capture.color[:, :, :3]
                if self.depth:
                    try:
                        depth_image = capture.depth[..., np.newaxis]
                    except:
                        print("Failed to grab depth image from camera")

                t_recv = time.time()
                mt_cap = capture.color_system_timestamp_nsec / 1e9 # monotonic time from time 0.0
                t_cap = (mt_cap - time.monotonic()) + time.time() # convert to system capture time
                t_cal = t_recv - self.receive_latency

                if not np.any(capture.color):
                    print("Failed to grab frame from camera")
                    return dict()

                # kinect returns images in BGR, we convert them to RGB for Robomimic
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                if self.resize:
                    color_image = resize_image(color_image, self.resize_resolution, image_type='rgb')

                data = {}
                data['rgb'] = color_image
                if self.depth:
                    data['depth'] = depth_image
                data['camera_capture_timestamp'] = t_cap
                data['camera_receive_timestamp'] = t_recv
                data['timestamp'] = t_cal
                # print(f"delta {t_recv-t_cap} capture time {t_cap}, receive time {t_recv} cal time {t_cal}")
                self.ring_buffer.put(data, wait=True)

                if not self.ready_event.is_set():
                    self.ready_event.set()
        finally:
            self.ready_event.set()
            camera.stop()

    # ========= context manager ===========
    def __enter__(self):
        self.start(wait=False) # TODO: check
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

if __name__ == "__main__":
    from pyk4a import PyK4A, connected_device_count
    from robomimic.utils.time_utils import precise_wait, precise_sleep

    cam_name = "agentview"
    serial_number = "001039114912"
    config = {"sn": serial_number}

    data_list = []
    camera_fps = 30.0
    with SharedMemoryManager() as shm_manager:
        camera = KinectCamera(cam_name=cam_name, config=config, shm_manager=shm_manager)
        camera.start(wait=False)

        cv2.namedWindow("kinect_rgb")
        while cv2.waitKey(1) != ord('q'):
            data = camera.get()
            rgb = data['rgb']
            t_cap = data['camera_capture_timestamp']
            t_recv = data['camera_receive_timestamp']
            t_cal = data['timestamp']
            # depth = data['depth']
            # depth = np.concatenate([depth, depth, depth], axis=-1)
            # print(rgb.shape)

            print(f"t_cap: {t_cap}, t_recv: {t_recv}, t_cal: {t_cal}")
            cv2.imshow("kinect_rgb", rgb)

            data_list.append(data)
            precise_sleep(1/camera_fps)

        # import pickle
        # with open("kinect_data_list.pkl", "wb") as f:
        #     pickle.dump(data_list, f)

        camera.stop()

