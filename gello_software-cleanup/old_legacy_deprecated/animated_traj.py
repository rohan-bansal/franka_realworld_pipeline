import numpy as np
import plotly.graph_objects as go
import pickle
from scipy.spatial.transform import Rotation

# =================================================================================
# --- CONFIGURATION ---
# =================================================================================

# File Paths
BAG_FILE_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/experiments/recorded_data.pkl"
OUTPUT_HTML_PATH = "/media/robot/Data_2/rohan/mfm/workspace/gello_software-cleanup/scripts/animated_traj_with_gradient.html"

# Animation & Visualization Parameters
ANIMATION_TIMESTEPS = 100000000000000000
PREDICTION_HORIZON = 4
GRADIENT_COLORSCALE = 'Blues' # You can change this to 'Reds', 'Greens', 'Viridis', etc.

# End-Effector Visualization Parameters
CUBE_HALF_SIZE = 0.04  # This creates an 8cm x 8cm x 8cm cube
VIRTUAL_EEF_POINTS = np.array([
    # 8 corners of a cube centered at the origin
    [-CUBE_HALF_SIZE, -CUBE_HALF_SIZE, -CUBE_HALF_SIZE],
    [-CUBE_HALF_SIZE, -CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [-CUBE_HALF_SIZE,  CUBE_HALF_SIZE, -CUBE_HALF_SIZE],
    [-CUBE_HALF_SIZE,  CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE, -CUBE_HALF_SIZE, -CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE, -CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE,  CUBE_HALF_SIZE, -CUBE_HALF_SIZE],
    [ CUBE_HALF_SIZE,  CUBE_HALF_SIZE,  CUBE_HALF_SIZE],
    # You can optionally include the origin point as well
    [0.0, 0.0, 0.0],
])
# TRAJECTORY_COLORS has been removed as we are now using a gradient.

# =================================================================================
# --- DATA & CAMERA PARAMETERS ---
# =================================================================================
CAMERA_POSE = np.array([
    [-0.01375253,  0.65322273, -0.7570409,   0.89966637],
    [ 0.99985662,  0.00150281, -0.01686684,  0.04009337],
    [-0.00988011, -0.75716432, -0.65314974,  0.47382265],
    [ 0.,          0.,          0.,          1.        ]
])

# =================================================================================
# --- HELPER FUNCTIONS ---
# =================================================================================
def pose_to_mat(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    mat = np.identity(4)
    mat[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    mat[:3, 3] = position
    return mat

def create_pose_traces(pose: np.ndarray, name: str, color: str = 'blue', length: float = 0.1):
    """Creates traces for a full pose (marker + 3 axes)."""
    traces = []
    pos = pose[:3, 3]
    axes = pose[:3, :3].T 
    traces.append(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=5, color=color), name=name, showlegend=False))
    for axis, color_ax in zip(axes, ['red', 'green', 'blue']):
        end = pos + length * axis
        traces.append(go.Scatter3d(x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]], mode='lines', line=dict(width=3, color=color_ax), showlegend=False))
    return traces

# =================================================================================
# --- MAIN SCRIPT ---
# =================================================================================

def main():
    """Load data and generate an ANIMATED visualization with a STATIC ground-truth path."""
    
    # --- 1. Load Trajectory Data ---
    print(f"Loading trajectory data from {BAG_FILE_PATH}...")
    with open(BAG_FILE_PATH, 'rb') as f:
        bag_data = pickle.load(f)

    max_timesteps = len(bag_data) - PREDICTION_HORIZON
    if max_timesteps < 1:
        print(f"Error: Not enough data in bag file for the animation.")
        return
    num_frames = min(ANIMATION_TIMESTEPS, max_timesteps)

    # Pre-calculate the full ground-truth trajectory
    print("Extracting full ground-truth path...")
    ground_truth_positions = np.array([bag_data[t]['input']['ee_pos'] for t in range(num_frames)])
    static_ground_truth_trace = go.Scatter3d(
        x=ground_truth_positions[:, 0], y=ground_truth_positions[:, 1], z=ground_truth_positions[:, 2],
        mode='lines', line=dict(color='rgba(255, 0, 255, 0.5)', width=4, dash='dash'),
        name='Full Ground-Truth Path'
    )

    # --- 2. Generate a Frame for Each Timestep ---
    print(f"Generating {num_frames} animation frames...")
    frames = []
    static_camera_traces = create_pose_traces(CAMERA_POSE, 'Camera Pose', color='black', length=0.1)

    for t in range(num_frames):
        frame_traces = []
        
        current_pose_mat = pose_to_mat(bag_data[t]['input']['ee_pos'], bag_data[t]['input']['ee_ori'])
        predicted_pose_horizon = bag_data[t]['output']['predicted_target_poses']
        full_predicted_trajectory = predicted_pose_horizon
        
        # --- ❗ NEW: Create a gradient color array for the prediction horizon ---
        num_pred_points = len(full_predicted_trajectory)
        # np.linspace creates an array from 0.0 to 1.0, which maps to the colorscale
        # This makes the start (newest) light and the end (oldest) dark.
        gradient_colors = np.linspace(1, 0, num_pred_points)
        # --- End of new code block ---

        # Add current EEF ground-truth position as a single, large marker
        current_pos = current_pose_mat[:3, 3]
        frame_traces.append(go.Scatter3d(
            x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
            mode='markers', marker=dict(size=7, color='purple', symbol='diamond'),
            name=f'Current EEF T={t}', showlegend=False
        ))

        # --- ❗ MODIFIED: Add traces for the predicted trajectories with the gradient ---
        for i, p_eef in enumerate(VIRTUAL_EEF_POINTS):
            p_eef_hom = np.append(p_eef, 1)
            world_path = np.array([(wTe @ p_eef_hom)[:3] for wTe in full_predicted_trajectory])
            
            frame_traces.append(go.Scatter3d(
                x=world_path[:, 0], y=world_path[:, 1], z=world_path[:, 2],
                mode='lines+markers',
                # The numeric gradient_colors array is mapped to the colorscale for both line and markers
                line=dict(width=5, color=gradient_colors, colorscale=GRADIENT_COLORSCALE),
                marker=dict(size=3, color=gradient_colors, colorscale=GRADIENT_COLORSCALE, showscale=False),
                name='Predicted Trajectory', # Generic name for the legend
                showlegend=(i==0) # Only show one legend entry for all 4 prediction traces
            ))
        
        # Add all static traces (camera and full path) to every frame
        all_traces_for_frame = [*static_camera_traces, *frame_traces, static_ground_truth_trace]
        frames.append(go.Frame(data=all_traces_for_frame, name=str(t)))
    
    # --- 3. Create Figure and Add Controls ---
    fig = go.Figure(data=frames[0].data)
    fig.frames = frames

    # --- 4. Add Animation Controls and SET AXIS RANGES ---
    fig.update_layout(
        title=f'Animated EEF Trajectory (World Frame) | Frame: 0/{num_frames-1}',
        updatemenus=[{'type': 'buttons', 'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ]}],
        sliders=[{'steps': [{'label': str(i), 'method': 'animate', 'args': [[str(i)], {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}]} for i in range(num_frames)],
                  'active': 0, 'currentvalue': {'prefix': 'Frame: ', 'visible': True}}]
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='World X (m)', range=[0, 1], autorange=False),
            yaxis=dict(title='World Y (m)', range=[-0.5, 0.5], autorange=False),
            zaxis=dict(title='World Z (m)', range=[0, 1], autorange=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            bgcolor='rgb(230, 230, 230)'
        ),
        legend=dict(x=0.01, y=0.99)
    )
    
    print(f"Saving animated plot to {OUTPUT_HTML_PATH}")
    fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
    print("Done! ✅")

if __name__ == '__main__':
    main()