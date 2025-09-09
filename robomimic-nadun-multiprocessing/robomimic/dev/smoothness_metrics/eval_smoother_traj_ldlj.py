# compute.py
import numpy as np
import math
import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def compute_ldlj(trajectory, dt):
    """Compute LDLJ for a given trajectory."""
    ## Extract end-effector positions: shape will be (N, 3) if each 'robot0_eef_pos'[1] is a 3D vector
    # positions = np.array([step['robot0_eef_pos'][1] for step in trajectory])

    # Using actual reference trajectory
    positions = np.array([step['actions'][1][:3] for step in trajectory])

    # If there are not enough points to compute derivatives, return something sensible (e.g., 0 or float('nan'))
    if positions.shape[0] < 3:
        return 0.0

    # Compute velocity: shape (N-1, 3)
    vel = (positions[1:] - positions[:-1]) / dt

    # Compute speed: shape (N-1,)
    speed = np.linalg.norm(vel, axis=1)

    # If there's only one velocity sample, we can't compute second derivatives, etc.
    if speed.shape[0] < 2:
        return 0.0

    # Compute first derivative of speed (dv/dt): shape (N-2,)
    dvdt = (speed[1:] - speed[:-1]) / dt

    # If there's only one dvdt sample, we can't compute second derivatives
    if dvdt.shape[0] < 2:
        return 0.0

    # Compute second derivative of speed (d2v/dt2): shape (N-3,)
    d2vdt2 = (dvdt[1:] - dvdt[:-1]) / dt

    # Find v_peak
    v_peak = np.max(speed) if len(speed) > 0 else 0.0
    if v_peak == 0:
        # Avoid division by zero
        return 0.0

    # Compute the integral of |d2v/dt2|^2
    integrand = np.abs(d2vdt2)**2
    integral = np.sum(integrand) * dt

    # Compute movement time
    movement_time = len(trajectory) * dt

    # Compute dimensionless jerk (DLJ) and log dimensionless jerk (LDLJ)
    DLJ = - (movement_time**5 / (v_peak**2)) * integral

    # Use absolute value inside the log to avoid negative or zero inside log
    LDLJ = -math.log(abs(DLJ)) if DLJ != 0 else float('inf')

    return LDLJ

def compare_trajectories(noisy_trajectory, inpainting_trajectory, dt):
    """Compare LDLJ values for noisy and inpainting trajectories."""
    noisy_ldlj = compute_ldlj(noisy_trajectory, dt)
    inpainting_ldlj = compute_ldlj(inpainting_trajectory, dt)

    smoother = "Noisy" if noisy_ldlj > inpainting_ldlj else "Inpainting"
    return noisy_ldlj, inpainting_ldlj, smoother

def compare(pickle_file, noisy_file, inpainting_file, dt):
    """Plot and compare trajectories from noisy and inpainting pairs."""
    data = load_pickle(pickle_file)
    noisy_data = load_pickle(noisy_file)
    inpainting_data = load_pickle(inpainting_file)

    smoother_counts = {"Noisy": 0, "Inpainting": 0}

    for noisy_key, matches in data.items():
        for match in matches:
            inpainting_key = match['inpainting_key']

            # Extract trajectories
            noisy_trajectory = noisy_data['rollouts'][noisy_key]['obs']
            inpainting_trajectory = inpainting_data['rollouts'][inpainting_key]['obs']

            # Compare LDLJ values
            noisy_ldlj, inpainting_ldlj, smoother = compare_trajectories(noisy_trajectory, inpainting_trajectory, dt)

            smoother_counts[smoother] += 1

            # Debug prints (uncomment for troubleshooting)
            # print(f"Noisy Key: {noisy_key}, Inpainting Key: {inpainting_key}")
            # print(f"  Noisy LDLJ: {noisy_ldlj:.4f}, Inpainting LDLJ: {inpainting_ldlj:.4f}")
            # print(f"  Smoother Trajectory: {smoother}")

    print("\nOverall Smoother Trajectory Counts:")
    print(f"  Noisy: {smoother_counts['Noisy']}")
    print(f"  Inpainting: {smoother_counts['Inpainting']}")

# Example usage
if __name__ == "__main__":
    # pickle_file = 'similar_rollout_pairs.pkl'
    noisy_file = 'noisy.pkl'
    inpainting_file = 'inpainting.pkl'

    # Sampling rate
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    # Compare and plot
    compare(pickle_file, noisy_file, inpainting_file, dt)

