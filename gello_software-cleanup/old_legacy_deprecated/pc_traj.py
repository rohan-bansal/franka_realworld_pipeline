import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# =================================================================================
# --- CONFIGURATION ---
# =================================================================================

# Image and Camera Parameters
WIDTH, HEIGHT = 640, 576

# File Paths
DEPTH_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/depth_capture.npy"
RGB_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/transformed_color_capture.npy"
OUTPUT_HTML_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/scripts/graph_traj_on_pc_cleaned_filtered.html"

# Point Cloud Generation & Filtering Parameters
DOWNSAMPLE_FACTOR = 1
K_NEIGHBORS = 50
STD_DEV_RATIO = 1.0

# =================================================================================
# --- HELPER FUNCTIONS ---
# =================================================================================

def scale_intrinsics(K: np.ndarray, orig_size: tuple, new_size: tuple) -> np.ndarray:
    """Scales a 3x3 intrinsics matrix K from an original to a new image size."""
    W_orig, H_orig = orig_size
    W_new, H_new = new_size

    sx = W_new / W_orig
    sy = H_new / H_orig

    K_new = K.copy()
    K_new[0, 0] *= sx  # fx
    K_new[1, 1] *= sy  # fy
    K_new[0, 2] *= sx  # cx
    K_new[1, 2] *= sy  # cy
    return K_new

def statistical_outlier_removal(points: np.ndarray, k: int, std_ratio: float) -> tuple:
    """Removes outliers from a point cloud using the SOR method."""
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = knn.kneighbors(points)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    mean_of_avg_distances = np.mean(avg_distances)
    std_of_avg_distances = np.std(avg_distances)
    distance_threshold = mean_of_avg_distances + std_ratio * std_of_avg_distances

    inlier_mask = avg_distances < distance_threshold
    return points[inlier_mask], inlier_mask

def create_colored_point_cloud(depth_image: np.ndarray, rgb_image: np.ndarray, intrinsics: np.ndarray, downsample: int) -> tuple:
    """Converts a depth image to a 3D point cloud with corresponding RGB colors."""
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    rows, cols = np.mgrid[0:height:downsample, 0:width:downsample]
    depths = depth_image[rows, cols].flatten()
    
    valid_mask = depths > 0
    rows, cols, depths = rows.flatten()[valid_mask], cols.flatten()[valid_mask], depths[valid_mask]
    
    colors = rgb_image[rows, cols]

    x = (cols - cx) * depths / fx
    y = (rows - cy) * depths / fy
    z = depths
    
    points = np.vstack((x, y, z)).T
    return points, colors

def add_pose_to_figure(fig: go.Figure, pose_matrix: np.ndarray, name: str, color: str = 'blue', axis_length: float = 0.1):
    """Helper function to draw a 3D pose (origin and axes) on a Plotly figure."""
    position = pose_matrix[:3, 3]
    axes = pose_matrix[:3, :3].T  # Each row is an axis vector (X, Y, Z)
    
    fig.add_trace(go.Scatter3d(
        x=[position[0]], y=[position[1]], z=[position[2]],
        mode='markers', marker=dict(size=6, color=color), name=name
    ))
    
    axes_colors = ['red', 'green', 'blue']
    for axis, color_ax in zip(axes, axes_colors):
        end_point = position + axis_length * axis
        fig.add_trace(go.Scatter3d(
            x=[position[0], end_point[0]],
            y=[position[1], end_point[1]],
            z=[position[2], end_point[2]],
            mode='lines', line=dict(width=4, color=color_ax), showlegend=False
        ))

# =================================================================================
# --- MAIN SCRIPT ---
# =================================================================================

def main():
    """Main function to load data, process it, and generate the visualization."""
    
    # --- 1. Load and Prepare Data ---
    print("Loading data...")
    depth_image = np.load(DEPTH_IMG_PATH, allow_pickle=True)
    depth_image = depth_image.astype(np.float32) / 1000.0 # mm to m
    rgb_image = np.load(RGB_IMG_PATH, allow_pickle=True)
    print(f"DEPTH IMAGE SHAPE: {depth_image.shape}")
    print(f"RGB IMAGE SHAPE: {rgb_image.shape}")
    
    # Hardcoded camera parameters
    camera_pose = np.array([
        [-0.01375253,  0.65322273, -0.7570409,   0.89966637],
        [ 0.99985662,  0.00150281, -0.01686684,  0.04009337],
        [-0.00988011, -0.75716432, -0.65314974,  0.47382265],
        [ 0.,          0.,          0.,          1.        ]
    ])
    intrinsics_orig = np.array([
        [505.48239136,   0.,         328.92779541],
        [  0.,         505.63397217, 326.85336304],
        [  0.,           0.,           1.        ]
    ])
    
    # --- 2. Process Point Cloud ---
    print("Generating and processing point cloud...")
    points_camera, colors_camera = create_colored_point_cloud(
        depth_image, rgb_image, intrinsics_orig, DOWNSAMPLE_FACTOR
    )

    points_homogeneous = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
    points_world = (camera_pose @ points_homogeneous.T).T[:, :3]
    print(f"Total points generated: {points_world.shape[0]}")

    # --- ❗ REORDERED STEP 1: First, remove statistical outliers from the FULL cloud ---
    sor_filtered_points, sor_inlier_mask = statistical_outlier_removal(
        points_world, k=K_NEIGHBORS, std_ratio=STD_DEV_RATIO
    )
    sor_filtered_colors = colors_camera[sor_inlier_mask]
    print(f"Points after SOR filter: {sor_filtered_points.shape[0]}")

    # --- ❗ REORDERED STEP 2: Now, apply the coordinate filter to the CLEANED cloud ---
    coordinate_mask = (sor_filtered_points[:, 0] >= 0) & (sor_filtered_points[:, 2] >= 0)
    
    final_points = sor_filtered_points[coordinate_mask]
    final_colors = sor_filtered_colors[coordinate_mask]
    print(f"Points after coordinate filter: {final_points.shape[0]}")
    
    # --- 3. Create Visualization ---
    print("Creating visualization...")
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=final_points[:, 0],
        y=final_points[:, 1],
        z=final_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=final_colors, opacity=1.0),
        name='Filtered Point Cloud'
    ))

    add_pose_to_figure(fig, camera_pose, 'Camera Pose', color='black', axis_length=0.2)
    
    # --- 4. Finalize and Save Plot ---
    fig.update_layout(
        title='3D View: Colored Point Cloud (Filtered)',
        scene=dict(
            xaxis_title='World X (m)',
            yaxis_title='World Y (m)',
            zaxis_title='World Z (m)',
            aspectmode='data',
            bgcolor='rgb(230, 230, 230)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    print(f"Saving plot to {OUTPUT_HTML_PATH}")
    fig.write_html(OUTPUT_HTML_PATH)
    print("Done!")
    
if __name__ == '__main__':
    main()