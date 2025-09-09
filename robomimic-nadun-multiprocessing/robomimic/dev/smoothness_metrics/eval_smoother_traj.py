# compute.py
import numpy as np
import math
import pickle
from eval_smoother_traj_ldlj import compute_ldlj

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def compute_sparc(trajectory, dt):
    """
    Compute SPARC (Spectral Arc Length) for a given trajectory.
    This implementation:
      - Extracts 3D end-effector positions from the trajectory.
      - Computes speed (magnitude of velocity).
      - Computes the real FFT of the speed.
      - Calculates the arc length in frequency domain.
      - Returns SPARC = -log(arc_length + epsilon).
    """

    # Extract end-effector positions: shape will be (N, 3)
    positions = np.array([step['robot0_eef_pos'][1] for step in trajectory])
    # Using actual refernce trajectory
    positions = np.array([step['actions'][1][:3] for step in trajectory])
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

def compare_trajectories(traj1, traj2, dt, eval_fn=compute_sparc)  :
    """Compare SPARC values for noisy and inpainting trajectories."""
    traj1_metric = eval_fn(traj1, dt)
    traj2_metric = eval_fn(traj2, dt)

    first_smoother = True if traj1_metric > traj2_metric else False
    return traj1_metric, traj2_metric, first_smoother

def compare(first_file, second_file, dt, filter_success=False):
    """Compare trajectories from two rollouts using SPARC."""
    # data = load_pickle(pickle_file)
    first_data = load_pickle(first_file)
    second_data = load_pickle(second_file)

    smoother_counts = {"First": 0, "Second": 0}
    total_sparc = {"First": 0, "Second": 0}
    total_rollouts = 0

    first_rollouts = first_data['rollouts']
    second_rollouts = second_data['rollouts']

    for rollout_num, rollout in first_rollouts.items():

        total_rollouts += 1

        if filter_success:
            if not (rollout['success'] and second_rollouts[rollout_num]['success']):
                continue

        first_trajectory = rollout['obs']
        second_trajectory = second_rollouts[rollout_num]['obs']

        first_sparc, second_sparc, first_smoother = compare_trajectories(first_trajectory, second_trajectory, dt)

        if first_smoother:
            smoother_counts["First"] += 1
        else:
            smoother_counts["Second"] += 1

        total_sparc["First"] += first_sparc
        total_sparc["Second"] += second_sparc

    # for noisy_key, matches in data.items():
    #     for match in matches:
    #         inpainting_key = match['inpainting_key']
    #
    #         # Extract trajectories
    #         noisy_trajectory = noisy_data['rollouts'][noisy_key]['obs']
    #         inpainting_trajectory = inpainting_data['rollouts'][inpainting_key]['obs']
    #
    #         # Compare SPARC values
    #         noisy_sparc, inpainting_sparc, smoother = compare_trajectories(
    #             noisy_trajectory, inpainting_trajectory, dt
    #         )
    #
    #         smoother_counts[smoother] += 1

            # Debug prints (uncomment for troubleshooting):
            # print(f"Noisy Key: {noisy_key}, Inpainting Key: {inpainting_key}")
            # print(f"  Noisy SPARC: {noisy_sparc:.4f}, Inpainting SPARC: {inpainting_sparc:.4f}")
            # print(f"  Smoother Trajectory: {smoother}")

    print("\nOverall Smoother Trajectory Counts:")
    print(f"  First: {smoother_counts['First']}")
    print(f"  Second: {smoother_counts['Second']}")

    avg_first_sparc = total_sparc["First"] / total_rollouts
    avg_second_sparc = total_sparc["Second"] / total_rollouts

    print("\n Average Sparc:")
    print(f" First: {avg_first_sparc}")
    print(f" Second: {avg_second_sparc}")



# Example usage
if __name__ == "__main__":
    # pickle_file = 'similar_rollout_pairs.pkl'

    first_file = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/SAIL/inpaint_resample_22025-01-17 22:57:47.239902.pkl"
    second_file = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/ablations/without_inpainting_fast_2025-01-20 01:02:23.615246.pkl"

    # first_file = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/inpaint_eval_fast2025-01-05 23:08:58.119782.pkl"
    # second_file = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/normal_eval_fast2025-01-05 23:02:48.106771.pkl"


    # Sampling rate
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    # Compare using SPARC
    compare(first_file, second_file, dt, filter_success=False)

