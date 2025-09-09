import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import json

# File path and demo group
file_name = 'pick_cube_1205_20hz_seq.hdf5'
file_subname = '_seg_ldlj_len8'
file_path = f'/nethome/wshin49/flash8/robomimic/{file_name}'
demo_group = "data/demo_0/obs/eef_pos"

# Sampling frequency and time step
sampling_rate = 20.0  # Hz
dt = 1.0 / sampling_rate

# Segment length (in number of timesteps)
segment_length = 8

# Load the full trajectory
with h5py.File(file_path, 'r') as hdf_file:
    trajectory = hdf_file[demo_group][:]  # Shape (N, 3)

N = trajectory.shape[0]
time_steps = np.arange(N) * dt

# Compute velocity, acceleration, and jerk globally
velocity = (trajectory[1:] - trajectory[:-1]) / dt           # (N-1, 3)
acceleration = (velocity[1:] - velocity[:-1]) / dt            # (N-2, 3)
jerk = (acceleration[1:] - acceleration[:-1]) / dt            # (N-3, 3)

time_steps_vel = np.arange(N-1) * dt
time_steps_acc = np.arange(N-2) * dt
time_steps_jerk = np.arange(N-3) * dt

def compute_ldlj_for_trajectory(segment_indices, dt, segment_length):
    """
    Compute DLJ and LDLJ for a given trajectory segment using precomputed velocity and jerk.

    Parameters
    ----------
    segment_indices : tuple
        Indices of the start and end of the segment.
    dt : float
        Time step.

    Returns
    -------
    LDLJ : float
    components : dict
    """
    start_idx, end_idx = segment_indices

    if end_idx - start_idx < segment_length:
        return np.nan, {}

    # Extract relevant velocity and acceleration values
    segment_velocity = velocity[start_idx:end_idx]
    segment_speed = np.linalg.norm(segment_velocity, axis=1)  # Speed

    if len(segment_speed) <= 2:
        return np.nan, {}

    # Compute dv/dt and d2v/dt2
    dvdt = (segment_speed[1:] - segment_speed[:-1]) / dt
    d2vdt2 = (dvdt[1:] - dvdt[:-1]) / dt

    # Compute v_peak
    v_peak = np.max(segment_speed)
    if v_peak == 0:
        return np.nan, {}

    # Compute integral of |d2v/dt2|^2
    integrand = np.abs(d2vdt2)**2
    integral = np.sum(integrand) * dt

    # Compute movement time
    t1 = start_idx * dt
    t2 = (end_idx - 1) * dt
    movement_time = t2 - t1

    # Compute DLJ and LDLJ
    DLJ = - (movement_time**5 / (v_peak**2)) * integral
    if DLJ == 0:
        return np.nan, {}

    LDLJ = -math.log(abs(DLJ))

    # Save components in a dictionary
    components = {
        "movement_time": movement_time,
        "v_peak": v_peak,
        "integral": integral,
        "DLJ": DLJ
    }

    return LDLJ, components

# Identify how many full segments we have
num_segments = N // segment_length

local_ldlj_values = []
component_list = []

# Compute LDLJ for each segment
for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = start_idx + segment_length
    segment_ldlj, components = compute_ldlj_for_trajectory((start_idx, end_idx), dt, segment_length)
    if not np.isnan(segment_ldlj):
        local_ldlj_values.append((segment_ldlj, start_idx, end_idx))
        component_list.append(components)

# Calculate total LDLJ over all valid segments
total_ldlj = np.mean([val[0] for val in local_ldlj_values]) if local_ldlj_values else np.nan

# Save components to JSON
output_json = {
    "total_ldlj": total_ldlj,
    "segments": component_list
}
with open(f'{file_name}{file_subname}.json', 'w') as json_file:
    json.dump(output_json, json_file, indent=4)
print(f'Saved LDLJ components to {file_name}{file_subname}.json')

# Plot all data in a 9-row, 1-column format
labels = ['X', 'Y', 'Z']
fig, axes = plt.subplots(9, 1, figsize=(10, 20), sharex=True)

# Plot position
for i in range(3):
    axes[i].plot(time_steps, trajectory[:, i], label=f'Position {labels[i]}', color='b')
    axes[i].set_ylabel(f'{labels[i]} Pos')
    axes[i].grid(True)
    axes[i].legend()

# Plot velocity
for i in range(3, 6):
    comp = i - 3
    axes[i].plot(time_steps_vel, velocity[:, comp], label=f'Velocity {labels[comp]}', color='g')
    axes[i].set_ylabel(f'{labels[comp]} Vel')
    axes[i].grid(True)
    axes[i].legend()

# Plot jerk
for i in range(6, 9):
    comp = i - 6
    axes[i].plot(time_steps_jerk, jerk[:, comp], label=f'Jerk {labels[comp]}', color='r')
    axes[i].set_ylabel(f'{labels[comp]} Jerk')
    axes[i].grid(True)
    axes[i].legend()

# Add vertical lines to show segment boundaries
for i in range(1, num_segments):
    vertical_line_x = time_steps[i * segment_length]
    for ax in axes:
        ax.axvline(x=vertical_line_x, color='k', linestyle='--', alpha=0.5)

# Add local LDLJ values as text on the top plot (X position)
if len(local_ldlj_values) > 0:
    max_val_x = np.max(trajectory[:, 0])
    for segment_ldlj, start_idx, end_idx in local_ldlj_values:
        segment_center_time = (start_idx + end_idx) / 2 * dt
        axes[0].text(segment_center_time, 1.01 * max_val_x, 
                     f"{segment_ldlj:.2f}", color="blue", fontsize=5, ha='center')

# Highlight top 10 segments with the smallest LDLJ
if len(local_ldlj_values) > 0:
    # Sort by LDLJ descending and pick top 10
    sorted_segments = sorted(local_ldlj_values, key=lambda x: x[0], reverse=False)
    top_segments = sorted_segments[:10]

    # Highlight these segments with a semi-transparent background
    highlight_color = 'yellow'
    for segment_ldlj, start_idx, end_idx in top_segments:
        start_time = start_idx * dt
        end_time = end_idx * dt
        for ax in axes:
            ax.axvspan(start_time, end_time, color=highlight_color, alpha=0.3)

axes[-1].set_xlabel('Time (s)')

# Add title with total LDLJ
fig.suptitle(f'{file_name} LDLJ: {total_ldlj:.4f}', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{file_name}{file_subname}.png')
plt.show()

