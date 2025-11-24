import numpy as np
import plotly.graph_objects as go
import pickle
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import cv2

# =================================================================================
# --- CONFIGURATION (Identical to before) ---
# =================================================================================

# --- Point Cloud File Paths & Parameters ---
DEPTH_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/depth_capture.npy"
RGB_IMG_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/gello/cameras/transformed_color_capture.npy"
PC_K_NEIGHBORS = 50
PC_STD_DEV_RATIO = 1.0

# --- Trajectory File Path & Parameters ---
BAG_FILE_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data.pkl"
OUTPUT_HTML_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/scripts/animated_traj_on_pc_optimized.html"

# --- Animation & Visualization Parameters ---
ANIMATION_TIMESTEPS = 100000000
PREDICTION_HORIZON = 5


CUSTOM_GRADIENT = [
    [0.0, 'rgb(173, 216, 230)'],  # Light Sky Blue at the start (newest)
    [1.0, 'rgb(0, 0, 139)']        # Dark Blue at the end (oldest)
]

CUBE_HALF_SIZE = 0.04
VIRTUAL_EEF_POINTS = np.array([
    [-CUBE_HALF_SIZE, -CUBE_HALF_SIZE, -CUBE_HALF_SIZE], [-CUBE_HALF_SIZE, -CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [-CUBE_HALF_SIZE,  CUBE_HALF_SIZE, -CUBE_HALF_SIZE], [-CUBE_HALF_SIZE,  CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE, -CUBE_HALF_SIZE, -CUBE_HALF_SIZE], [ CUBE_HALF_SIZE, -CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE,  CUBE_HALF_SIZE, -CUBE_HALF_SIZE], [ CUBE_HALF_SIZE,  CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [0.0, 0.0, 0.0],
])

# =================================================================================
# --- SHARED CAMERA PARAMETERS (Identical to before) ---
# =================================================================================
CAMERA_POSE = np.array([
    [-0.01375253,  0.65322273, -0.7570409,   0.89966637],
    [ 0.99985662,  0.00150281, -0.01686684,  0.04009337],
    [-0.00988011, -0.75716432, -0.65314974,  0.47382265],
    [ 0.,          0.,          0.,          1.        ]
])
INTRINSICS = np.array([
    [505.48239136,   0.,         328.92779541],
    [  0.,         505.63397217, 326.85336304],
    [  0.,           0.,           1.        ]
])

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

def create_colored_point_cloud(depth_image: np.ndarray, rgb_image: np.ndarray, intrinsics: np.ndarray) -> tuple:
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    rows, cols = np.mgrid[0:height, 0:width]
    depths = depth_image.flatten()
    valid_mask = depths > 0
    rows, cols, depths = rows.flatten()[valid_mask], cols.flatten()[valid_mask], depths[valid_mask]
    colors = rgb_image[rows, cols]
    x = (cols - cx) * depths / fx
    y = (rows - cy) * depths / fy
    z = depths
    points = np.vstack((x, y, z)).T
    return points, colors

def pose_to_mat(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    mat = np.identity(4)
    mat[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    mat[:3, 3] = position
    return mat

def create_pose_traces(pose: np.ndarray, name: str, color: str = 'blue', length: float = 0.1):
    traces = []
    pos = pose[:3, 3]
    axes = pose[:3, :3].T 
    traces.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=5, color=color), name=name, showlegend=False))
    for axis, color_ax in zip(axes, ['red', 'green', 'blue']):
        end = pos + length * axis
        traces.append(go.Scatter3d(x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]], mode='lines', line=dict(width=3, color=color_ax), showlegend=False))
    return traces

def generate_dynamic_traces(t, bag_data):
    dynamic_traces = []
    current_pose_mat = pose_to_mat(bag_data[t]['input']['ee_pos'], bag_data[t]['input']['ee_ori'])
    predicted_pose_horizon = bag_data[t]['output']['predicted_target_poses']
    
    num_pred_points = len(predicted_pose_horizon)
    gradient_colors = np.linspace(1, 0, num_pred_points)

    current_pos = current_pose_mat[:3, 3]
    dynamic_traces.append(go.Scatter3d(
        x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
        mode='markers', marker=dict(size=7, color='purple', symbol='diamond'), showlegend=False
    ))

    for i, p_eef in enumerate(VIRTUAL_EEF_POINTS):
        p_eef_hom = np.append(p_eef, 1)
        world_path = np.array([(wTe @ p_eef_hom)[:3] for wTe in predicted_pose_horizon])
        dynamic_traces.append(go.Scatter3d(
            x=world_path[:, 0], y=world_path[:, 1], z=world_path[:, 2],
            mode='lines+markers',
            line=dict(width=5, color=gradient_colors, colorscale=CUSTOM_GRADIENT),
            marker=dict(size=3, color=gradient_colors, colorscale=CUSTOM_GRADIENT, showscale=False),
            name='Predicted Trajectory', showlegend=(i==0)
        ))
    return dynamic_traces

# =================================================================================
# --- MAIN SCRIPT ---
# =================================================================================

def main():
    # --- 1. Load and Process ALL STATIC Data First ---
    print("Loading and processing static point cloud...")
    depth_image = np.load(DEPTH_IMG_PATH, allow_pickle=True).astype(np.float32) / 1000.0
    rgb_image = np.load(RGB_IMG_PATH, allow_pickle=True)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    points_camera, colors_camera = create_colored_point_cloud(depth_image, rgb_image, INTRINSICS)
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

    print(f"Loading trajectory data from {BAG_FILE_PATH}...")
    with open(BAG_FILE_PATH, 'rb') as f: bag_data = pickle.load(f)
    num_frames = min(ANIMATION_TIMESTEPS, len(bag_data) - PREDICTION_HORIZON)

    ground_truth_positions = np.array([bag_data[t]['input']['ee_pos'] for t in range(num_frames)])
    static_ground_truth_trace = go.Scatter3d(
        x=ground_truth_positions[:, 0], y=ground_truth_positions[:, 1], z=ground_truth_positions[:, 2],
        mode='lines', line=dict(color='rgba(255, 0, 255, 0.5)', width=4, dash='dash'), name='urrent EEF Pose Traj'
    )
    
    static_camera_traces = create_pose_traces(CAMERA_POSE, 'Camera Pose', color='black', length=0.1)

    # --- 2. Define the Initial Figure with ALL Traces ---
    static_traces = [point_cloud_trace, static_ground_truth_trace, *static_camera_traces]
    initial_dynamic_traces = generate_dynamic_traces(0, bag_data)
    
    fig = go.Figure(data=static_traces + initial_dynamic_traces)
    
    num_static_traces = len(static_traces)
    num_dynamic_traces = len(initial_dynamic_traces)
    dynamic_trace_indices = list(range(num_static_traces, num_static_traces + num_dynamic_traces))

    # --- 3. Generate Frames that ONLY contain data for the DYNAMIC traces ---
    print(f"Generating {num_frames} animation frames...")
    frames = []
    for t in range(num_frames):
        frames.append(go.Frame(
            data=generate_dynamic_traces(t, bag_data),
            name=str(t),
            traces=dynamic_trace_indices
        ))
    fig.frames = frames

    # --- 4. ❗ MODIFIED: Add Animation Controls with redraw=True ---
    fig.update_layout(
        title=f'Animated EEF Trajectory on Point Cloud (Optimized) | Frame: 0/{num_frames-1}',
        updatemenus=[{'type': 'buttons', 'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]}],
        sliders=[{'steps': [{'label': str(i), 'method': 'animate', 'args': [[str(i)], {'frame': {'duration': 30, 'redraw': True}, 'mode': 'immediate'}]} for i in range(num_frames)],
                  'active': 0, 'currentvalue': {'prefix': 'Frame: ', 'visible': True}}]
    )
    

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
            bgcolor='black' # The main scene background
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