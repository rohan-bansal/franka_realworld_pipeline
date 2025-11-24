import numpy as np
import plotly.graph_objects as go
import pickle
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import cv2

# =================================================================================
# --- CONFIGURATION (Identical to before) ---
# =================================================================================

DEPTH_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/depth_capture.npy"
RGB_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/transformed_color_capture.npy"
PC_K_NEIGHBORS = 50
PC_STD_DEV_RATIO = 1.0

OUTPUT_HTML_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/scripts/output_traj_replay.html"


# =================================================================================
# --- SHARED CAMERA PARAMETERS (Identical to before) ---
# =================================================================================
CAMERA_POSE = np.array([
    [-0.042057, 0.633554, -0.772554, 0.908922],
    [0.998585, 0.051834, -0.011854, 0.040206],
    [0.032534, -0.771960, -0.634838, 0.523073],
    [0.000000, 0.000000, 0.000000, 1.000000],
])

T_base_to_cam = np.linalg.inv(CAMERA_POSE)

INTRINSICS = np.array([
    [505.48239136,   0.,         328.92779541],
    [  0.,         505.63397217, 326.85336304],
    [  0.,           0.,           1.        ]
])
DEPTH_DISTORTION = np.array([ 5.49890161e-01,  2.55133994e-02,  1.38819945e-04,  1.37125913e-04,
       -2.59828317e-04,  8.91191602e-01,  1.37836039e-01, -9.21121318e-05])

# =================================================================================
# --- HELPER FUNCTIONS (Identical to before) ---
# =================================================================================
def statistical_outlier_removal(points: np.ndarray, k: int, std_ratio: float) -> tuple:
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = knn.kneighbors(points)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    mean_of_avg_distances = np.mean(avg_distances)
    std_of_avg_distances = np.std(avg_distances)
    distance_threshold = mean_of_avg_distances + std_ratio * std_of_avg_distances
    inlier_mask = avg_distances < distance_threshold
    return points[inlier_mask], inlier_mask

def create_colored_point_cloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray = None,
) -> tuple:
    """
    Build a colored point cloud from a depth image and RGB image.
    If `distortion` is provided, we undistort pixel coordinates using cv2.undistortPoints.
    """
    height, width = depth_image.shape

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Pixel grid
    rows, cols = np.mgrid[0:height, 0:width]

    depths = depth_image.flatten()
    valid_mask = depths > 0

    rows = rows.flatten()[valid_mask]
    cols = cols.flatten()[valid_mask]
    depths = depths[valid_mask]

    # Colors from original (distorted) RGB image
    colors = rgb_image[rows, cols]

    if distortion is None:
        # Old behavior: no distortion correction
        x = (cols - cx) * depths / fx
        y = (rows - cy) * depths / fy
    else:
        # Use OpenCV to undistort pixel locations.
        # pts = (u, v) = (col, row) in image coordinates.
        pts = np.stack([cols, rows], axis=-1).astype(np.float32)  # (N, 2)

        undist_norm = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            intrinsics.astype(np.float32),
            distortion.astype(np.float32),
            None,   # R
            None,   # P (newCameraMatrix); None => normalized image coords
        ).reshape(-1, 2)  # (N, 2), each = (x_norm, y_norm)

        # Normalized coords -> 3D
        x = undist_norm[:, 0] * depths
        y = undist_norm[:, 1] * depths

    z = depths
    points = np.stack((x, y, z), axis=-1)  # (N, 3)

    return points, colors

def pose_to_mat(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    mat = np.identity(4)
    mat[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    mat[:3, 3] = position
    return mat


# =================================================================================
# --- MAIN SCRIPT ---
# =================================================================================

def main():
    # --- 1. Load and Process ALL STATIC Data First ---
    print("Loading and processing static point cloud...")
    depth_image = np.load(DEPTH_IMG_PATH, allow_pickle=True).astype(np.float32) / 1000.0

    rgb_image = np.load(RGB_IMG_PATH, allow_pickle=True)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    points_camera, colors_camera = create_colored_point_cloud(
        depth_image,
        rgb_image,
        INTRINSICS,
        DEPTH_DISTORTION,
    )

    points_homogeneous = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
    points_world = (CAMERA_POSE @ points_homogeneous.T).T[:, :3]
    sor_filtered_points, sor_inlier_mask = statistical_outlier_removal(points_world, k=PC_K_NEIGHBORS, std_ratio=PC_STD_DEV_RATIO)
    sor_filtered_colors = colors_camera[sor_inlier_mask]
    coordinate_mask = (sor_filtered_points[:, 0] >= 0) & (sor_filtered_points[:, 2] >= 0)
    final_points, final_colors = sor_filtered_points[coordinate_mask], sor_filtered_colors[coordinate_mask]
    
    point_cloud_trace = go.Scatter3d(
        x=final_points[:, 0], y=final_points[:, 1], z=final_points[:, 2],
        mode='markers', marker=dict(size=2, color=final_colors, opacity=1.0), name='Point Cloud'
    )
    print(f"Point cloud processed: {final_points.shape[0]} points.")

    static_traces = [point_cloud_trace]
    
    fig = go.Figure(data=static_traces)
    
    fig.update_layout(
        title=dict(font=dict(color='white')), # Make title text white
        scene=dict(
            xaxis=dict(
                title='World X (m)', range=[0, 1], autorange=False,
                gridcolor='rgb(70, 70, 70)',       # Dimmer grid lines
                showbackground=False,              # <-- KEY CHANGE: Hide the light-colored pane
                zerolinecolor='rgb(150, 150, 150)',
                color='white'
            ),
            yaxis=dict(
                title='World Y (m)', range=[-0.5, 0.5], autorange=False,
                gridcolor='rgb(70, 70, 70)',
                showbackground=False,              # <-- KEY CHANGE: Hide the light-colored pane
                zerolinecolor='rgb(150, 150, 150)',
                color='white'
            ),
            zaxis=dict(
                title='World Z (m)', range=[0, 1], autorange=False,
                gridcolor='rgb(70, 70, 70)',
                showbackground=False,              # <-- KEY CHANGE: Hide the light-colored pane
                zerolinecolor='rgb(150, 150, 150)',
                color='white'
            ),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
            # bgcolor='black' # The main scene background
        ),
        legend=dict(
            x=0.01, y=0.99,
            font=dict(color='white'),
            bgcolor='rgba(0, 0, 0, 0.5)'
        )
    )
    
    print(f"Saving combined animated plot to {OUTPUT_HTML_PATH}")
    fig.write_html(OUTPUT_HTML_PATH)
    print("Done! ✅")

if __name__ == '__main__':
    main()