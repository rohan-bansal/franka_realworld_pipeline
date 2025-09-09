import numpy as np
import math
import pickle
import h5py
import matplotlib.pyplot as plt

def compute_sparc_pos(trajectory, dt):
    """
    Compute SPARC (Spectral Arc Length) for a given trajectory.
    This implementation:
      - Extracts 3D end-effector positions from the trajectory.
      - Computes speed (magnitude of velocity).
      - Computes the real FFT of the speed.
      - Calculates the arc length in frequency domain.
      - Returns SPARC = -log(arc_length + epsilon).

    Args:
        positions: obs robot0_eef_pos from a rollout/demo
        dt:
    """
    positions = trajectory['eef_pos'][:]
    if len(positions) < 2:
        return 0.0

    # Compute velocity and speed
    vel = (positions[1:] - positions[:-1]) / dt  # shape (N-1, 3)
    speed = np.linalg.norm(vel, axis=1)          # shape (N-1,)

    # If there's not enough data, return 0.0
    if len(speed) < 2:
        return 0.0

    # Compute FFT of the speed
    # rfft gives us the positive frequencies only
    speed_fft = np.fft.rfft(speed)
    freqs = np.fft.rfftfreq(len(speed), d=dt)

    # Compute arc length in the frequency domain
    # arc_length = sum over i of sqrt((Δfreq)^2 + (Δ|FFT|)^2)
    arc_length = 0.0
    for i in range(len(freqs) - 1):
        df = freqs[i+1] - freqs[i]
        dmag = np.abs(speed_fft[i+1]) - np.abs(speed_fft[i])
        arc_length += np.sqrt(df**2 + dmag**2)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    sparc = -math.log(arc_length + epsilon)

    return sparc

def compute_ldlj_pos(trajectory):
    positions = trajectory['eef_pos'][:]

    vel = (positions[1:] - positions[:-1]) / dt

    # 1. Compute velocity
    if vel is None:
        vel = np.diff(positions, axis=0) / dt

    # 2. Compute speed
    speed = np.linalg.norm(vel, axis=1)  # shape (N-1,)

    # 3. Compute first derivative of speed (dv/dt)
    dvdt = (speed[1:] - speed[:-1]) / dt  # shape (N-2,)

    # 4. Compute second derivative of speed (d2v/dt2)
    d2vdt2 = (dvdt[1:] - dvdt[:-1]) / dt  # shape (N-3,)

    # Now we have d2v/dt2 defined on a shorter time interval.
    # Let's define our effective start and end times:
    # Original time array for position: t = 0, dt, 2*dt, ..., (N-1)*dt
    N = vel.shape[0] + 1
    t1 = 0.0
    t2 = (N - 1) * dt

    # 5. Find v_peak
    v_peak = np.max(speed) if len(speed) > 0 else 0.0
    if v_peak == 0:
        raise ValueError("Peak speed is zero, cannot compute dimensionless jerk.")

    # 6. Compute the integral of |d2v/dt2|^2 from t1 to t2
    # Align indices carefully. d2vdt2 corresponds to times:
    # For positions: indices: 0 to N-1
    # For speed: indices: 0 to N-2
    # for dvdt: indices: 0 to N-3
    # for d2vdt2: indices: 0 to N-4
    #
    # We can approximate the integral as sum(|d2v/dt2|^2)*dt.
    integrand = np.abs(d2vdt2) ** 2
    integral = np.sum(integrand) * dt

    # 7. Compute DLJ
    movement_time = t2 - t1
    DLJ = - (movement_time ** 5 / (v_peak ** 2)) * integral

    # 8. Compute LDLJ
    LDLJ = -math.log(abs(DLJ))

    return LDLJ

def compute_sparc_vel_ang(trajectory, dt):
    """
    Compute SPARC (Spectral Arc Length) for a given angular velocity trajectory.

    This implementation:
      - Extracts angular velocity (3D) from the trajectory.
      - Computes the speed (magnitude of angular velocity).
      - Computes the real FFT of the speed.
      - Calculates the arc length in the frequency domain.
      - Returns SPARC = -log(arc_length + epsilon).
      
    Parameters
    ----------
    trajectory : list of dict
        Each element contains 'robot0_eef_vel_ang', which is the angular velocity vector (1x3).
    dt : float
        Time step in seconds.

    Returns
    -------
    sparc : float
        Smoothness value calculated using SPARC (higher values => smoother motion).
    """
    # Extract angular velocity: shape will be (N, 3)
    ang_vel = np.array([step['robot0_eef_vel_ang'][1] for step in trajectory])
    if len(ang_vel) < 2:
        return 0.0

    # Compute speed (magnitude of angular velocity)
    speed = np.linalg.norm(ang_vel, axis=1)  # shape (N,)

    # If there's not enough data, return 0.0
    if len(speed) < 2:
        return 0.0

    # Compute FFT of the speed
    speed_fft = np.fft.rfft(speed)
    freqs = np.fft.rfftfreq(len(speed), d=dt)

    # Compute arc length in the frequency domain
    arc_length = 0.0
    for i in range(len(freqs) - 1):
        df = freqs[i+1] - freqs[i]
        dmag = np.abs(speed_fft[i+1]) - np.abs(speed_fft[i])
        arc_length += np.sqrt(df**2 + dmag**2)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    sparc = -math.log(arc_length + epsilon)

    return sparc

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def plot_trajectory(trajectory, save_path=None):
    # Extract trajectories
    trajectory = trajectory['eef_pos'][:]

    # Determine dimensions
    dimensions = len(trajectory[0])
    timesteps = range(len(trajectory))

    # Dimension names
    dimension_names = ['x', 'y', 'z']

    # Initialize plots
    fig, axes = plt.subplots(dimensions, 1, figsize=(10, 6))
    fig.suptitle('End Effector Trajectories')

    # Plot each dimension
    for dim in range(dimensions):
        axes[dim].plot(timesteps, trajectory[:, dim], label='Trajectory', linestyle='--', marker='o')
        axes[dim].set_xlabel('Timestep')
        axes[dim].set_ylabel(f'{dimension_names[dim]}', rotation=0)
        axes[dim].legend()
        axes[dim].grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

# Sampling rate
sampling_rate = 20.0  # Hz
dt = 1.0 / sampling_rate

file_path = '0111_wiping_action_schedule_20hz_inpaint_FF.hdf5'
file_path = '0111_wiping_action_schedule_20hz_inpaint.hdf5'
file_path = '0111_wiping_action_schedule_40hz_inpaint.hdf5'
file_path = '0111_wiping_seq_20hz.hdf5'
with h5py.File(file_path, 'r') as hdf_file:
    for key, val in hdf_file['data'].items():
        traj = hdf_file['data'][key]['obs']

        sparc_pos = compute_sparc_pos(traj, dt)
        print(f"key: {key}, sparc_pos: {sparc_pos}")
        ldlj_pos = compute_ldlj_pos(traj)
        print(f"key: {key}, ldlj_pos: {ldlj_pos}")

        plot_trajectory(traj, save_path='trajectory_plot.png')