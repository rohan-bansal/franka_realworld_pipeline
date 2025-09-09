"""
Camera driver for an Azure Kinect camera built on pyk4a.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
import time
from cv2 import COLOR_BGRA2BGR, COLOR_BGRA2RGB
from copy import deepcopy
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

from gello.cameras.camera import CameraDriver
import pyzed.sl as sl

from robomimic.utils.obs_utils import resize_image
from robomimic.dev.multi_processing.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from robomimic.dev.multi_processing.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


class ZedCamera(mp.Process, CameraDriver):
    """
    Data struct follows the ringbuffer. Each frames contain
    {image, depth, timestamp}
    NOTE: make sure you check the timestamp when accessing the data (and limit the access rate on upstream task)
    """
    def __repr__(self) -> str:
        return f"ZedCamera(name= {self.cam_name}, serial_number={self.serial_number})"

    def __init__(self,
                 cam_name,
                 config,
                 shm_manager: SharedMemoryManager,
                 depth: bool=False,
                 # camera_pos = "left", # "dual" "left" or "right"
                 receive_latency: float=0.076,
                 get_max_k: int=1000,
                 verbose: bool=False):

        super().__init__(name="ZedCamera")
        # Setting some class vars
        self.cam_name = cam_name
        self.serial_number = config['sn']
        self.depth = depth
        self.receive_latency = receive_latency
        self.camera_pos = config['camera_pos']
        self.fps = config.get('fps', 30)
        cameras = sl.Camera.get_device_list()

        camera_properties = None
        for cam in cameras:
            if cam.serial_number == self.serial_number:
                camera_properties = cam
        if camera_properties is None:
            raise Exception(f"Could not find Zed camera with serial number: {self.serial_number}")

        self.serial_number = str(camera_properties.serial_number)
        self.resolution = config.get('zed_resolution', sl.RESOLUTION.HD720) # 480P sl.RESOLUTION.VGA RESOLUTION.HD720

        # Do we resize the images from the camera?
        self.resize = config.get('resize', False)
        if self.resize:
            self.resize_resolution = config.get('resize_resolution', (128, 128))
        else:
            self.resize_resolution = (1280, 720) # default resolution

        # Create ring buffer
        resolution = tuple(self.resize_resolution)
        image_shape = resolution[::-1]
        if self.camera_pos == "dual":
            examples = {
                'left_rgb': np.empty(shape=image_shape+(3,), dtype=np.uint8),
                'right_rgb': np.empty(shape=image_shape + (3,), dtype=np.uint8)
            }
            if self.depth:
                examples['left_depth'] = np.empty(shape=self.resolution + (1,), dtype=np.uint16)
                examples['right_depth'] = np.empty(shape=self.resolution+(1,), dtype=np.uint16)
        else:
            examples = {
                'rgb': np.empty(shape=image_shape+(3,), dtype=np.uint8)
            }
            if self.depth:
                examples['depth'] = np.empty(shape=self.resolution+(1,), dtype=np.uint16) # 11 bits
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.02, # NOTE: 0.2 originally, but don't konw why so high..
            put_desired_frequency=self.fps
        )

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer

    @property
    def type(self):
        return "Zed"

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
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(self.serial_number))
        init_params.camera_resolution = self.resolution
        init_params.camera_fps = self.fps
        init_params.enable_right_side_measure = True

        camera = sl.Camera()

        ret = camera.open(init_params)
        if ret != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : " + repr(ret) + ". Exit program.")
            exit()

        # Set some vars to get the camera frames
        _runtime_params = sl.RuntimeParameters()
        _left_img = sl.Mat()
        _right_img = sl.Mat()
        _left_depth = sl.Mat()
        _right_depth = sl.Mat()

        camera_information = camera.get_camera_information()
        all_intrinsics = camera_information.camera_configuration.calibration_parameters

        print(f"Starting Zed camera with sn: {self.serial_number} at process {self.pid}")

        try:
            # ====== Main Loop ===== #
            while not self.stop_event.is_set():
                ts = time.time()

                # grab the color and depth images
                ret = camera.grab(_runtime_params)
                if ret != sl.ERROR_CODE.SUCCESS:
                    print("Failed to grab frame from camera")
                    return dict()

                data = {}
                if self.camera_pos == "dual":
                    # return images from both camera
                    camera.retrieve_image(_left_img, sl.VIEW.LEFT, resolution=self.resolution)
                    camera.retrieve_image(_right_img, sl.VIEW.RIGHT, resolution=self.resolution)
                    camera.retrieve_measure(_left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
                    camera.retrieve_measure(_right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.resolution)

                    # Zed returns images in BGRA, we convert them to RGB for Robomimic training, should this be a parameter?
                    data['left_rgb'] = cv2.cvtColor(deepcopy(_left_img.get_data()), COLOR_BGRA2RGB)
                    data['right_rgb'] = cv2.cvtColor(deepcopy(_right_img.get_data()), COLOR_BGRA2RGB)

                    if self.depth:
                        data['left_depth'] = deepcopy(_left_depth.get_data())[:, :, None]
                        data['right_depth'] = deepcopy(_right_depth.get_data())[:, :, None]
                    if self.resize:
                        data['left_rgb'] = resize_image(data['left_rgb'], self.resize_resolution, image_type='rgb')
                        data['right_rgb'] = resize_image(data['right_rgb'], self.resize_resolution, image_type='rgb')
                        if self.depth:
                            data['left_depth'] = resize_image(data['left_depth'], self.resize_resolution, image_type='depth')
                            data['right_depth'] = resize_image(data['right_depth'], self.resize_resolution, image_type='depth')
                else:
                    # return images from single
                    # Zed returns images in BGRA, we convert them to RGB for Robomimic training, should this be a parameter?

                    # Left Camera
                    if self.camera_pos == "left":
                        camera.retrieve_image(_left_img, sl.VIEW.LEFT, resolution=self.resolution)
                        camera.retrieve_measure(_left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
                        data['rgb'] = cv2.cvtColor(deepcopy(_left_img.get_data()), COLOR_BGRA2RGB)
                        if self.depth:
                            data['depth'] = deepcopy(_left_depth.get_data())[:, :, None]
                    # Right camera
                    elif self.camera_pos == "right":
                        camera.retrieve_image(_right_img, sl.VIEW.RIGHT, resolution=self.resolution)
                        camera.retrieve_measure(_right_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
                        data['rgb'] = cv2.cvtColor(deepcopy(_right_img.get_data()), COLOR_BGRA2RGB)
                        if self.depth:
                            data['depth'] = deepcopy(_right_depth.get_data())[:, :, None]
                    else:
                        raise ValueError(f"Wrong camera pose {self.camera_pos} for Zed camrea")

                    if self.resize:
                        data['rgb'] = resize_image(data['rgb'], self.resize_resolution, image_type='rgb')
                        if self.depth:
                            data['depth'] = resize_image(data['depth'], self.resize_resolution, image_type='depth')

                timestamp_image = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
                t_recv = time.time()
                mt_cap = timestamp_image / 1e6 # monotonic time from time 0.0
                t_cap = mt_cap
                t_cal = t_recv - self.receive_latency

                data['camera_capture_timestamp'] = t_cap
                data['camera_receive_timestamp'] = t_recv
                data['timestamp'] = t_cal
                # print(f"delta {t_recv-t_cap} capture time {t_cap}, receive time {t_recv} cal time {t_cal}")
                self.ring_buffer.put(data, wait=False)

                if not self.ready_event.is_set():
                    self.ready_event.set()
        finally:
            self.ready_event.set()
            camera.close()

    # ========= context manager ===========
    def __enter__(self):
        self.start(wait=False) # TODO: check
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

if __name__ == "__main__":
    from robomimic.utils.time_utils import precise_wait, precise_sleep

    cam_name = "wrist"
    serial_number = 14620168
    camera_fps = 60.0
    config = {"sn": serial_number, "fps": camera_fps, "camera_pos": "left"}
    data_list = []

    with SharedMemoryManager() as shm_manager:
        camera = ZedCamera(cam_name=cam_name, config=config, shm_manager=shm_manager)
        camera.start(wait=False)
        # camera.start()

        cv2.namedWindow("zed_rgb")
        while cv2.waitKey(1) != ord('q'):
            data = camera.get()
            rgb = data['rgb']
            t_cap = data['camera_capture_timestamp']
            t_recv = data['camera_receive_timestamp']
            t_cal = data['timestamp']
            # depth = data['depth']
            # depth = np.concatenate([depth, depth, depth], axis=-1)
            # print(rgb.shape)

            # print(f"t_cap: {t_cap}, t_recv: {t_recv}, t_cal: {t_cal}")
            cv2.imshow("zed_rgb", rgb)

            data_list.append(data)
            precise_sleep(1/camera_fps)

        # import pickle
        # with open("zed_data_list60.pkl", "wb") as f:
        #     pickle.dump(data_list, f)

        camera.stop()

