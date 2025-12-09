"""
Camera driver for an Azure Kinect camera built on pyk4a.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
import time
import sys

from gello.cameras.camera import CameraDriver
import pyk4a
from pyk4a import Config, connected_device_count, Calibration, ColorControlMode
from pyk4a.config import FPS,ImageFormat, DepthMode, ColorResolution
from pyk4a.calibration import CalibrationType
import pickle

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_CIVIDIS,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

# example config = {'color_resolution': pyk4a.ColorResolution.RES_720P,
#                       'fps': 30, 'resize': True, 'resize_resolution': (128, 128)}
class KinectCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"KinectCamera(name= {self.cam_name}, serial_number={self.serial_number})"


    def __init__(self, cam_name, config):

        # Setting some class vars
        # TODO: do we need a cam name?
        self.cam_name = cam_name
        self.serial_number = config['sn']

        # Setting the config for the camera
        pyk4a_config = Config()
        pyk4a_config.camera_fps = config.get('fps', FPS.FPS_30)
        pyk4a_config.color_resolution = config.get('color_resolution', ColorResolution.RES_720P)
        pyk4a_config.color_format = config.get('color_format', ImageFormat.COLOR_BGRA32)
        pyk4a_config.depth_mode = config.get('depth_mode', DepthMode.NFOV_UNBINNED)
        pyk4a_config.synchronized_images_only = config.get("synchronize_depth", True)
        self.pyk4a_config = pyk4a_config

        # Start the camera
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
                if device.serial == self.serial_number: # we found our camera_id
                    camera_id = device_id
                device.close()
        if camera_id is None:
            raise  Exception(f"Could not find Kinect with serial number: {self.serial_number}")


        self._camera = pyk4a.PyK4A(device_id=camera_id, config=pyk4a_config)
        print(f"Starting Kinect camera with sn: {self.serial_number}")
        self._camera.start()

        self.resize = config.get('resize', False)
        if self.resize:
            self.resize_resolution = config.get('resize_resolution', (128, 128))

        self._camera.exposure_mode_auto = False
        self._camera.whitebalance_mode_auto = False
        self._camera.exposure = 8000 # 9000
        self._camera.whitebalance = 4510

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> dict:

        if img_size is not None:
            self.resize_resolution = img_size
            self.resize = True

        # First, grab a frame
        capture = self._camera.get_capture()

        if not np.any(capture.color):
            print("Failed to grab frame from camera")
            return dict()

        # Next get the individual color/depth images
        color_image = capture.color[:, :, :3]
        # TODO: kinect returns images in BGR, we convert them to RGB for Robomimic training, should this be a parameter?
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # if self.resize:
        #     # print(color_image.shape)
        #     color_image = cv2.resize(color_image, self.resize_resolution, interpolation=cv2.INTER_AREA)

        # Process depth
        # if np.any(capture.depth):
        #     depth = capture.depth
        #     if self.resize:
        #         depth = cv2.resize(depth, self.resize_resolution, cv2.INTER_NEAREST)
        # else:
        #     depth = None

        depth = capture.depth

        data = {}
        data['rgb'] = color_image
        data['depth'] = depth
        data["transformed_color"] = capture.transformed_color
        data["transformed_depth"] = capture.transformed_depth
        data["depth_point_cloud"] = capture.depth_point_cloud
        return data

    @property
    def type(self):
        return "Kinect"

    def calibration(self):
        raw_calibration = self._camera.calibration_raw
        calibration = Calibration.from_raw(
            raw_calibration, depth_mode=self.pyk4a_config.depth_mode, color_resolution=self.pyk4a_config.color_resolution
        )
        color_intrinsics = calibration.get_camera_matrix(CalibrationType.COLOR)
        depth_intrinsics = calibration.get_camera_matrix(CalibrationType.DEPTH)
        color_distortion = calibration.get_distortion_coefficients(CalibrationType.COLOR)
        depth_distortion = calibration.get_distortion_coefficients(CalibrationType.DEPTH)
        all_intrinsics = {
            'color_intrinsics': color_intrinsics,
            'depth_intrinsics': depth_intrinsics,
            'color_distortion': color_distortion,
            'depth_distortion': depth_distortion,
        }

        # print(color_intrinsics)
        # exit()
        return all_intrinsics
    
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


def debug_read(camera):
    cv2.namedWindow("kinect_rgb")
    cv2.namedWindow("kinect_depth")

    print(camera.calibration())

    try:
        depth = None
        rgb = None
        while True:

            data = camera.read()

            rgb = data["rgb"]
            rgb = obs_preprocess(rgb)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Color", rgb)
            cv2.imshow("Depth", colorize(data["depth"], (None, 5000)))
            cv2.imshow("Transformed Color", data["transformed_color"])
            cv2.imshow("Transformed Depth", colorize(data["transformed_depth"], (None, 5000)))


            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
            
    except KeyboardInterrupt:
        print(data["depth_point_cloud"].shape)
        print(data["rgb"].shape)
        np.save("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/depth_capture.npy", data["depth"])
        np.save("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/transformed_color_capture.npy", data["transformed_color"])
        np.save("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/rgb_capture.npy", data['rgb'])
        sys.exit()


if __name__ == "__main__":
    from pyk4a import PyK4A, connected_device_count

    cam_name = "agentview"
    serial_number = "001039114912"
    config = {"sn": serial_number, "resize": True, "resize_resolution": (640,576)}
    camera = KinectCamera(cam_name=cam_name, config=config)
    debug_read(camera)
    # data = camera.calibration()
    # with open("./intrinsics.pkl", "wb") as handle:
    #     pickle.dump(data, handle)

