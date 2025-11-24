import os
import cv2
import numpy as np
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation

# ---- import your KinectCamera ----
from gello.cameras.kinect_camera import KinectCamera   # <-- change to actual import

def draw_pose_axes(overlay, camera_params, tag_size, pose, center):
    """
    Draw XYZ axes on the image for the detected tag pose.
    camera_params: [fx, fy, cx, cy]
    pose: 3x4 or 4x4 [R|t]
    """

    fx, fy, cx, cy = camera_params
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1 ]], dtype=np.float32)

    # Ensure pose is at least 3x4
    R_cam_tag = pose[:3, :3]
    t_cam_tag = pose[:3, 3]

    rvec, _ = cv2.Rodrigues(R_cam_tag)
    tvec = t_cam_tag

    dcoeffs = np.zeros(5, dtype=np.float32)

    # 3D axis points in tag frame (X red, Y green, Z blue)
    opoints = np.float32([
        [1, 0, 0],   # X
        [0, -1, 0],  # Y
        [0, 0, -1],  # Z
    ]) * tag_size

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)

    center = np.round(center).astype(int)
    center = tuple(center.ravel())

    cv2.line(overlay, center, tuple(ipoints[0].ravel()), (0, 0, 255), 2)   # X - red
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), (0, 255, 0), 2)   # Y - green
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), (255, 0, 0), 2)   # Z - blue


def main():
    # ---- Kinect setup ----
    cam_name = "agentview"
    serial_number = "001039114912"  # change if needed
    config = {
        "sn": serial_number,
        "resize": False,            # keep native resolution to match intrinsics
    }
    camera = KinectCamera(cam_name=cam_name, config=config)

    # ---- Intrinsics for Kinect (you gave this) ----
    K = np.array([
        [608.01702881,   0.0,        640.24462891],
        [0.0,            607.79614258, 364.64956665],
        [0.0,              0.0,        1.0],
    ], dtype=np.float32)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # dt_apriltags expects [fx, fy, cx, cy]
    cam_params = [fx, fy, cx, cy]

    # ---- AprilTag detector ----
    at_detector = Detector(
        families='tagStandard41h12',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    # ---- IO dirs (optional, same as your RealSense code) ----
    cwd = os.getcwd()
    raw_directory = os.path.join(cwd, 'color') + '/'
    tags_directory = os.path.join(cwd, 'detected_tags') + '/'
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(tags_directory, exist_ok=True)

    index = 0

    try:
        while True:
            data = camera.read()
            if "rgb" not in data:
                print("No RGB frame from Kinect")
                continue

            # data["rgb"] is RGB (because you cvtColor(BGR2RGB) in KinectCamera.read)
            rgb_image = data["rgb"]

            # For OpenCV drawing / imshow we want BGR
            color_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            overlay = color_bgr.copy()

            # Grayscale for AprilTag detection (RGB -> Gray)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

            # Detect tags
            tags = at_detector.detect(
                gray_image,
                estimate_tag_pose=True,
                camera_params=cam_params,
                tag_size=0.06 * 5 / 9,   # keep your original tag size
            )

            if len(tags) >= 0:
                for tag in tags:
                    print("rotation")
                    rot = Rotation.from_matrix(tag.pose_R)
                    print(rot.as_quat())
                    print("translation")
                    print(tag.pose_t)

                    # Build 3x4 pose matrix [R|t]
                    pose_3x4 = np.concatenate([tag.pose_R, tag.pose_t], axis=1)

                    draw_pose_axes(
                        overlay,
                        cam_params,
                        tag_size=0.05,    # same as your draw_pose_axes call
                        pose=pose_3x4,
                        center=tag.center,
                    )

            # Save images (optional)
            cv2.imwrite(tags_directory + f"{index}.png", overlay)
            cv2.imwrite(raw_directory + f"{index}.png", color_bgr)

            # Save intrinsics once (overwrite is fine)
            with open(os.path.join(cwd, 'cam_params.txt'), 'w') as f:
                f.write(f"{cam_params}")

            # Show image
            cv2.imshow("Kinect AprilTag", overlay)
            index += 1
            print(index)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        # KinectCamera has no explicit stop, but you can add one if you implement it.
        # e.g., camera._camera.stop()


if __name__ == "__main__":
    main()
