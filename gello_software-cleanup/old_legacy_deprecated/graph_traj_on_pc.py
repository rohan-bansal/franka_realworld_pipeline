import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pickle
from sklearn.neighbors import NearestNeighbors

# --- 1. SIMULATE INPUT DATA ---

# -- Camera Parameters --
WIDTH, HEIGHT = 640, 480

# extrinsics
camera_pose = np.array([
    [-0.01375253,  0.65322273, -0.7570409,   0.89966637],
    [ 0.99985662,  0.00150281, -0.01686684,  0.04009337],
    [-0.00988011, -0.75716432, -0.65314974,  0.47382265],
    [ 0.,          0.,          0.,          1.        ]])

intrinsics = np.array([
    [505.48239136,   0.,         328.92779541],
    [  0.,         505.63397217, 326.85336304],
    [  0.,           0.,           1.        ]])

def scale_intrinsics(K: np.ndarray, orig_size: tuple, new_size: tuple) -> np.ndarray:
    """
    Scale a 3x3 intrinsics matrix K from orig_size -> new_size.

    Args:
        K: (3,3) numpy array (fx, fy, cx, cy form)
        orig_size: (width, height) of original image
        new_size: (width, height) of resized image

    Returns:
        K_new: (3,3) numpy array scaled intrinsics
    """
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

intrinsics = scale_intrinsics(intrinsics, (640, 576), (640, 480))


# We will also need the inverse to go from WORLD to CAMERA
camera_pose_inv = np.linalg.inv(camera_pose)

depth_image = np.load("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/depth_capture.npy", allow_pickle=True)
print("DEPTH IMAGE SHAPE", depth_image.shape)
rgb_image = np.load("/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/rgb_capture.npy", allow_pickle=True)
print("RGB IMAGE SHAPE", rgb_image.shape)

def statistical_outlier_removal(points, k, std_ratio):
    """
    Removes outliers from a point cloud using the Statistical Outlier Removal (SOR) method.

    Args:
        points (np.array): The (N, 3) point cloud.
        k (int): The number of nearest neighbors to consider for each point.
        std_ratio (float): The standard deviation ratio. Points with an average distance
                           greater than (mean + std_ratio * std_dev) will be removed.

    Returns:
        np.array: The filtered (M, 3) point cloud, where M <= N.
        np.array: The indices of the removed outlier points.
    """
    # 1. Find the k-nearest neighbors for each point
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    # distances are to the k+1 neighbors (including the point itself)
    distances, indices = knn.kneighbors(points)

    # 2. Calculate the average distance for each point to its k neighbors
    # We slice [:, 1:] to exclude the distance to the point itself (which is 0)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # 3. Calculate the global mean and standard deviation of these average distances
    mean_of_avg_distances = np.mean(avg_distances)
    std_of_avg_distances = np.std(avg_distances)

    # 4. Define the distance threshold
    distance_threshold = mean_of_avg_distances + std_ratio * std_of_avg_distances

    # 5. Find the inliers (points that are NOT outliers)
    inlier_mask = avg_distances < distance_threshold
    
    # Also find the outlier indices for visualization
    outlier_indices = np.where(inlier_mask == False)[0]

    return points[inlier_mask], inlier_mask

# -- Action Poses --
# A few 4x4 action poses in the WORLD frame.
# Let's create a simple trajectory.
action_poses = []
for i in range(4):
    pose = np.identity(4)
    # Move along the x-axis in the world
    pose[0, 3] = 0.5 + i * 0.3
    # With some height
    pose[2, 3] = 0.2
    action_poses.append(pose)

action_poses = np.array(action_poses)


def create_colored_point_cloud(depth_image, rgb_image, intrinsics, downsample=4):
    """
    Converts a depth image to a 3D point cloud, extracting color from an RGB image.
    
    Args:
        depth_image (np.array): The HxW depth image.
        rgb_image (np.array): The HxWx3 RGB image.
        intrinsics (np.array): The 3x3 camera intrinsic matrix.
        downsample (int): Factor to downsample for performance.
    
    Returns:
        np.array: An (N, 3) array of (X, Y, Z) points.
        np.array: An (N, 3) array of (R, G, B) colors.
    """
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Get pixel coordinates, downsampled for efficiency
    rows, cols = np.mgrid[0:height:downsample, 0:width:downsample]
    
    # Get depth values and filter out invalid points (depth == 0)
    depths = depth_image[rows, cols].flatten()
    valid_mask = depths > 0
    
    # Apply mask to coordinates and depths
    rows, cols = rows.flatten()[valid_mask], cols.flatten()[valid_mask]
    depths = depths[valid_mask]
    
    # --- NEW: Extract colors for the valid points ---
    colors = rgb_image[rows, cols]

    # Unproject 2D pixels to 3D points in camera frame
    x = (cols - cx) * depths / fx
    y = (rows - cy) * depths / fy
    z = depths
    
    points = np.vstack((x, y, z)).T
    
    return points, colors

# --- Generate and Transform the Point Cloud ---

# 1. Create colored point cloud in the CAMERA frame
points_camera, colors_camera = create_colored_point_cloud(
    depth_image, 
    rgb_image, 
    intrinsics, 
    downsample=2  # Increased downsampling for faster rendering
)

# 2. Transform points to the WORLD frame
points_homogeneous = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
points_world = (camera_pose @ points_homogeneous.T).T[:, :3]

k_neighbors = 50
std_dev_ratio = 1.0

filtered_points, inlier_mask = statistical_outlier_removal(
    points_world,
    k=k_neighbors,
    std_ratio=std_dev_ratio
)

filtered_colors = colors_camera[inlier_mask]

print(f"Original points: {points_world.shape[0]}")
print(f"Filtered points: {filtered_points.shape[0]}")

def add_pose_to_figure(fig, pose_matrix, name, color='blue', axis_length=0.1):
    """Helper function to draw a pose (origin and axes) on a Plotly figure."""
    # Position is the last column
    position = pose_matrix[:3, 3]
    # Axes are the first three columns
    x_axis, y_axis, z_axis = pose_matrix[:3, 0], pose_matrix[:3, 1], pose_matrix[:3, 2]

    # Add the origin point
    fig.add_trace(go.Scatter3d(
        x=[position[0]], y=[position[1]], z=[position[2]],
        mode='markers', marker=dict(size=6, color=color), name=name
    ))
    
    # Add the axes lines (X=Red, Y=Green, Z=Blue)
    axes_colors = ['red', 'green', 'blue']
    for axis, color_ax in zip([x_axis, y_axis, z_axis], axes_colors):
        fig.add_trace(go.Scatter3d(
            x=[position[0], position[0] + axis_length * axis[0]],
            y=[position[1], position[1] + axis_length * axis[1]],
            z=[position[2], position[2] + axis_length * axis[2]],
            mode='lines', line=dict(width=4, color=color_ax), showlegend=False
        ))

fig_3d_color = go.Figure()

# Add the colored Point Cloud using the filtered arrays
fig_3d_color.add_trace(go.Scatter3d(
    x=filtered_points[:, 0], # <-- Use filtered points
    y=filtered_points[:, 1], # <-- Use filtered points
    z=filtered_points[:, 2], # <-- Use filtered points
    mode='markers',
    marker=dict(
        size=2,
        color=filtered_colors,   # <-- Use filtered colors
        opacity=1.0
    ),
    name='Filtered Point Cloud'
))


add_pose_to_figure(fig_3d_color, camera_pose, 'Kinect Camera', color='black', axis_length=0.2)
for i, pose in enumerate(action_poses):
    add_pose_to_figure(fig_3d_color, pose, f'Action Pose {i}', color='magenta')


# Update Layout
fig_3d_color.update_layout(
    title='3D View: Colored Point Cloud and Action Poses',
    scene=dict(
        xaxis_title='World X (m)',
        yaxis_title='World Y (m)',
        zaxis_title='World Z (m)',
        aspectmode='data',
        bgcolor='rgb(230, 230, 230)' # A light gray background helps colors pop
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig_3d_color.show()