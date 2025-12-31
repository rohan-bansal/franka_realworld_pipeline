#!/usr/bin/env python3
"""
Minimal Azure Kinect recorder (pyk4a):

Keys:
  s : start recording
  e : stop recording + save
  q / ESC : quit

Saves:
  rgb, depth, transformed_rgb, transformed_depth
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, connected_device_count
from pyk4a.config import FPS, ImageFormat, DepthMode, ColorResolution


def colorize_depth(depth: np.ndarray,
                   clipping_range: Tuple[Optional[int], Optional[int]] = (None, 5000),
                   colormap: int = cv2.COLORMAP_CIVIDIS) -> np.ndarray:
    """Depth (uint16) -> colorized uint8 for visualization."""
    if depth is None or depth.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    lo, hi = clipping_range
    img = depth.copy()
    if lo is not None or hi is not None:
        img = img.clip(lo if lo is not None else img.min(),
                       hi if hi is not None else img.max())
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(img, colormap)


def find_device_id_by_serial(target_serial: str) -> int:
    cnt = connected_device_count()
    if cnt <= 0:
        raise RuntimeError("No Azure Kinect devices detected (connected_device_count() == 0).")

    for device_id in range(cnt):
        dev = pyk4a.PyK4A(device_id=device_id)
        if dev.opened:
            continue
        dev.open()
        serial = dev.serial
        dev.close()
        if serial == target_serial:
            return device_id

    raise RuntimeError(f"Could not find Azure Kinect with serial number: {target_serial}")


def make_camera(serial_number: str) -> pyk4a.PyK4A:
    device_id = find_device_id_by_serial(serial_number)

    cfg = Config(
        camera_fps=FPS.FPS_30,
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_BGRA32,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )

    cam = pyk4a.PyK4A(device_id=device_id, config=cfg)
    cam.start()

    # Match your code’s manual settings (optional but consistent)
    cam.exposure_mode_auto = False
    cam.whitebalance_mode_auto = False
    cam.exposure = 8000
    cam.whitebalance = 4510

    return cam


def main():
    # ---- user config ----
    serial_number = "001039114912"
    out_dir = Path("./kinect_recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------

    cam = make_camera(serial_number)

    # Recording buffers (lists -> stacked on save)
    recording = False
    rgb_buf: List[np.ndarray] = []
    depth_buf: List[np.ndarray] = []
    trgb_buf: List[np.ndarray] = []
    tdepth_buf: List[np.ndarray] = []

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Transformed RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Transformed Depth", cv2.WINDOW_NORMAL)

    print("Ready. Press 's' to start recording, 'e' to end+save, 'q'/ESC to quit.")

    try:
        while True:
            cap = cam.get_capture()

            if cap.color is None or not np.any(cap.color):
                # Occasional empty frame; skip
                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break
                continue

            # ---- Extract frames ----
            # cap.color is BGRA; convert to RGB (uint8)
            color_bgr = cap.color[:, :, :3]  # BGR
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            depth = cap.depth  # uint16 (H,W), may be None depending on config
            transformed_rgb = cap.transformed_color  # often BGRA aligned to depth
            transformed_depth = cap.transformed_depth

            # ---- Visualize ----
            cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if depth is not None:
                cv2.imshow("Depth", colorize_depth(depth, (None, 5000)))
            else:
                cv2.imshow("Depth", np.zeros((480, 640, 3), dtype=np.uint8))

            if transformed_rgb is not None and np.any(transformed_rgb):
                cv2.imshow("Transformed RGB", transformed_rgb)
            else:
                cv2.imshow("Transformed RGB", np.zeros((480, 640, 3), dtype=np.uint8))

            if transformed_depth is not None and np.any(transformed_depth):
                cv2.imshow("Transformed Depth", colorize_depth(transformed_depth, (None, 5000)))
            else:
                cv2.imshow("Transformed Depth", np.zeros((480, 640, 3), dtype=np.uint8))

            # ---- Record ----
            if recording:
                rgb_buf.append(rgb.copy())
                if depth is not None:
                    depth_buf.append(depth.copy())
                if transformed_rgb is not None:
                    trgb_buf.append(transformed_rgb.copy())
                if transformed_depth is not None:
                    tdepth_buf.append(transformed_depth.copy())

                cv2.setWindowTitle("RGB", "RGB  [REC]")
            else:
                cv2.setWindowTitle("RGB", "RGB")

            # ---- Keys ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if not recording:
                    recording = True
                    rgb_buf.clear()
                    depth_buf.clear()
                    trgb_buf.clear()
                    tdepth_buf.clear()
                    print("Recording started.")
                else:
                    print("Already recording.")
            elif key == ord("e"):
                if recording:
                    recording = False
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = out_dir / f"kinect_{serial_number}_{ts}.npz"

                    # Stack with best-effort consistency checks
                    rgb_arr = np.stack(rgb_buf, axis=0) if rgb_buf else np.empty((0,))
                    depth_arr = np.stack(depth_buf, axis=0) if depth_buf else np.empty((0,))
                    trgb_arr = np.stack(trgb_buf, axis=0) if trgb_buf else np.empty((0,))
                    tdepth_arr = np.stack(tdepth_buf, axis=0) if tdepth_buf else np.empty((0,))

                    np.savez_compressed(
                        out_path,
                        rgb=rgb_arr,
                        depth=depth_arr,
                        transformed_rgb=trgb_arr,
                        transformed_depth=tdepth_arr,
                    )
                    print(f"Saved: {out_path}")
                else:
                    print("Not recording; press 's' first.")
            elif key in (ord("q"), 27):  # q or ESC
                break

    finally:
        cv2.destroyAllWindows()
        try:
            cam.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
