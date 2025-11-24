import os
import time
from typing import List, Optional, Tuple
import cv2


import numpy as np

from gello.cameras.camera import CameraDriver



def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        enable_depth: bool = True,
        flip: bool = False,
    ):
        import pyrealsense2 as rs
        import time

        self._device_id = device_id
        self._flip = flip
        self._enable_depth = enable_depth

        self._pipeline = rs.pipeline()
        config = rs.config()

        if device_id is not None:
            config.enable_device(device_id)
        else:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)

        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        self._pipeline.start(config)
        print(f"STARTED REALSENSE WITH ID : {self._device_id} | RES: {width}x{height} | DEPTH: {enable_depth}")

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> dict:
        import cv2
        import numpy as np

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Initialize output dictionary
        data = {}

        if img_size is None:
            image = color_image[:, :, ::-1]
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]

        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)

        data["rgb"] = image

        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            if img_size is not None:
                depth_image = cv2.resize(depth_image, img_size)
            depth_image = depth_image[:, :, None]
            if self._flip:
                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
            data["depth"] = depth_image
        else:
            data["depth"] = None

        return data

    @property
    def type(self):
        return "RealSense"


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        res = camera.read()
        rgb = res["rgb"]
        depth = res["depth"]
        # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth * 255.0
        depth_display = np.uint8(depth_normalized)

        # Optional: apply a color map
        # depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Show the depth image
        key = cv2.waitKey(1)
        cv2.imshow("image", rgb)
        cv2.imshow("depth", depth_display)
        # if key == ord("s"):
        #     cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
        #     cv2.imwrite(f"images/depth_{counter}.png", depth)
        # if save_datastream:
        #     cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
        #     cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=False, device_id=device_ids[0], enable_depth=True)
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)