import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

### Set the constants and data here


# log_fn = "/home/wjung85/Repo/projects/FastIL/logs/inpainting_horizon_400_2024-12-13 15:26:09.904344.pkl"



# Setup some parameters for calculation
# first_demo = data['rollouts']['demo_0']
# kwargs = first_demo['kwargs']
# inf_delay = kwargs['inf_delay']
# execute_n_actions = kwargs['execute_n_actions']

# disregard_inpainted_actions = True
# filter_success = False


def normalize_vectors(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms

def calculate_euclidean_distance(data, filter_success = False, disregard_inpainted_actions = True):
    first_demo = data['rollouts']['demo_0']
    kwargs = first_demo['kwargs']
    inf_delay = kwargs['inf_delay']
    execute_n_actions = kwargs['execute_n_actions']
    
    total_euclidean_distance = 0
    total_num_preds = 0
    total_points = 0
    for demo in data['rollouts']:
        demo = data['rollouts'][demo]
        if filter_success:
            if not demo['rewards'][-1] > 0:
                continue
        # demo = data['rollouts']['demo_0']
        preds = demo['preds']

        curr_prediction_index = execute_n_actions

        for i in range(len(preds) - 1):
            total_num_preds += 1
            if not disregard_inpainted_actions:
                if i > 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                # This is the part of the current prediction that will overlap with the beginning of the next
                curr_pred_overlap = preds[i][curr_prediction_index:]
                # This is part of the next prediction that overlaps with the current
                next_pred_overlap = preds[i + 1][:curr_pred_overlap.shape[0]]


            elif disregard_inpainted_actions: # we only consider the overlapping sections that are not inpainted
                if i == 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                else:
                    curr_prediction_index = execute_n_actions + inf_delay*2

                curr_pred_overlap = preds[i][curr_prediction_index:]
                # Disregard the section at the beginning of the next prediction, which is inpainted
                next_pred_overlap = preds[i+1][inf_delay:inf_delay+curr_pred_overlap.shape[0]]

            # Select only the translation parts
            curr_pred_overlap = curr_pred_overlap[:, :3]
            next_pred_overlap = next_pred_overlap[:, :3]

            total_points += curr_pred_overlap.shape[0]

            # Calculate euclidean distance for overlapping section
            euclidean_distance = np.sum(np.sqrt(np.sum(np.square(curr_pred_overlap - next_pred_overlap), axis=1)))
            total_euclidean_distance += euclidean_distance

    mean_euclidean_distance = total_euclidean_distance / total_num_preds
    mean_diff_per_point = total_euclidean_distance / total_points
    print(f"Mean euclidean distance: {mean_euclidean_distance}")
    print(f"Mean diff_per_point : {mean_diff_per_point}")
    return mean_euclidean_distance, mean_diff_per_point

def calculate_euclidean_distance_fn(data, filter_success = False, disregard_inpainted_actions = True):
    first_demo = data['rollouts']['demo_0']
    kwargs = first_demo['kwargs']
    inf_delay = kwargs['inf_delay']
    execute_n_actions = kwargs['execute_n_actions']
    
    total_euclidean_distance = 0
    total_num_preds = 0
    total_points = 0
    
    distances = []
    for demo in data['rollouts']:
        demo = data['rollouts'][demo]
        if filter_success:
            if not demo['rewards'][-1] > 0:
                continue
        # demo = data['rollouts']['demo_0']
        preds = demo['preds']

        curr_prediction_index = execute_n_actions

        for i in range(len(preds) - 1):
            total_num_preds += 1
            if not disregard_inpainted_actions:
                if i > 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                # This is the part of the current prediction that will overlap with the beginning of the next
                curr_pred_overlap = preds[i][curr_prediction_index:]
                # This is part of the next prediction that overlaps with the current
                next_pred_overlap = preds[i + 1][:curr_pred_overlap.shape[0]]


            elif disregard_inpainted_actions: # we only consider the overlapping sections that are not inpainted
                if i == 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                else:
                    curr_prediction_index = execute_n_actions + inf_delay*2

                curr_pred_overlap = preds[i][curr_prediction_index:]
                # Disregard the section at the beginning of the next prediction, which is inpainted
                next_pred_overlap = preds[i+1][inf_delay:inf_delay+curr_pred_overlap.shape[0]]

            # Select only the translation parts
            curr_pred_overlap = curr_pred_overlap[:, :3]
            next_pred_overlap = next_pred_overlap[:, :3]

            # Calculate euclidean distance for overlapping section
            euclidean_distance = np.sum(np.sqrt(np.sum(np.square(curr_pred_overlap - next_pred_overlap), axis=1)))
            distances.append(euclidean_distance)
        
    return distances

def calculate_euclidean_distance_fn(data, filter_success=False, disregard_inpainted_actions=True):
    """
    Calculate euclidean distances between overlapping predictions.
    Returns a list of distances for each overlapping section.
    
    Args:
        data (dict): Data dictionary containing rollouts
        filter_success (bool): Whether to only process successful trajectories
        disregard_inpainted_actions (bool): Whether to disregard inpainted actions
        
    Returns:
        list: List of euclidean distances for each overlapping section
    """
    first_demo = data['rollouts']['demo_0']
    kwargs = first_demo['kwargs']
    inf_delay = kwargs['inf_delay']
    execute_n_actions = kwargs['execute_n_actions']
    
    distances = []  # Store individual distances
    total_num_preds = 0
    
    for demo_key in data['rollouts']:
        demo = data['rollouts'][demo_key]
        if filter_success and not demo['rewards'][-1] > 0:
            continue
            
        preds = demo['preds']
        curr_prediction_index = execute_n_actions

        for i in range(len(preds) - 1):
            total_num_preds += 1
            if not disregard_inpainted_actions:
                if i > 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                curr_pred_overlap = preds[i][curr_prediction_index:]
                next_pred_overlap = preds[i + 1][:curr_pred_overlap.shape[0]]

            else:  # disregard_inpainted_actions
                if i == 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                else:
                    curr_prediction_index = execute_n_actions + inf_delay*2

                curr_pred_overlap = preds[i][curr_prediction_index:]
                next_pred_overlap = preds[i+1][inf_delay:inf_delay+curr_pred_overlap.shape[0]]

            # Select only the translation parts
            curr_pred_overlap = curr_pred_overlap[:, :3]
            next_pred_overlap = next_pred_overlap[:, :3]

            # Calculate euclidean distance for overlapping section
            euclidean_distance = np.sum(np.sqrt(np.sum(np.square(curr_pred_overlap - next_pred_overlap), axis=1)))
            distances.append(euclidean_distance)
    
    return distances



def calculate_dot_product(data, filter_success = False, disregard_inpainted_actions = True):
    first_demo = data['rollouts']['demo_0']
    kwargs = first_demo['kwargs']
    inf_delay = kwargs['inf_delay']
    execute_n_actions = kwargs['execute_n_actions']

    total_dot_product = 0
    total_num_preds = 0
    total_delta_actions = 0
    for demo in data['rollouts']:
        demo = data['rollouts'][demo]
        if filter_success:
            if not demo['rewards'][-1] > 0:
                continue
        # demo = data['rollouts']['demo_0']
        preds = demo['preds']

        curr_prediction_index = execute_n_actions

        for i in range(len(preds) - 1):
            total_num_preds += 1
            if not disregard_inpainted_actions:
                if i > 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                # This is the part of the current prediction that will overlap with the beginning of the next
                curr_pred_overlap = preds[i][curr_prediction_index:]
                # This is part of the next prediction that overlaps with the current
                next_pred_overlap = preds[i + 1][:curr_pred_overlap.shape[0]]


            elif disregard_inpainted_actions: # we only consider the overlapping sections that are not inpainted
                if i == 0:
                    curr_prediction_index = execute_n_actions + inf_delay
                else:
                    curr_prediction_index = execute_n_actions + inf_delay*2

                curr_pred_overlap = preds[i][curr_prediction_index:]
                # Disregard the section at the beginning of the next prediction, which is inpainted
                next_pred_overlap = preds[i+1][inf_delay:inf_delay+curr_pred_overlap.shape[0]]

            # Select only the translation parts
            curr_pred_overlap = curr_pred_overlap[:, :3]
            next_pred_overlap = next_pred_overlap[:, :3]

            curr_pred_deltas = curr_pred_overlap[1:] - curr_pred_overlap[:-1]
            next_pred_deltas = next_pred_overlap[1:] - next_pred_overlap[:-1]

            curr_pred_deltas = normalize_vectors(curr_pred_deltas)
            next_pred_deltas = normalize_vectors(next_pred_deltas)

            sum_dot_prod = np.sum(curr_pred_deltas * next_pred_deltas)
            total_dot_product += sum_dot_prod
            total_delta_actions += curr_pred_deltas.shape[0]

    dot_product_per_action = total_dot_product / total_delta_actions
    print(f"Dot product per action: {dot_product_per_action}")
    return dot_product_per_action


def calculate_smoothness_from_vel(positions=None, vel=None, print_results=False, return_all_data=False):
    assert not (positions is None and vel is None), "Pass in either positions or velocities"

    # Sampling frequency and time step
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    # Smoothing tolerance factor
    k = 2.0

    # Compute velocity via finite differences
    # vel shape = (N-1, D)
    if vel is None:
        vel = np.diff(positions, axis=0) / dt

    # Compute mean and std of velocity for each dimension
    mean_vel = np.mean(vel, axis=0)
    std_vel = np.std(vel, axis=0)

    # Define boundaries for velocity in each dimension
    upper_bound = mean_vel + k * std_vel
    lower_bound = mean_vel - k * std_vel

    # Check which velocity points fall outside the smoothness boundary
    # A point is considered an outlier if its velocity in any dimension is outside the boundary
    outliers = []
    for i, v in enumerate(vel):
        # Check if any component of v is outside the allowed range
        if np.any(v > upper_bound) or np.any(v < lower_bound):
            outliers.append(i)

    smoothness_score = 1.0 - (len(outliers) / float(len(vel)))

    # Print results
    if print_results:
        print("Total number of frames in trajectory:", positions.shape[0])
        print("Total number of velocity samples:", len(vel))
        print("Mean velocity:", mean_vel)
        print("Std velocity:", std_vel)
        print("Number of suspected outlier points:", len(outliers))
        print("Outlier indices:", outliers)
        print("Smoothness Score (1.0 = perfectly smooth):", smoothness_score)

    if return_all_data:
        return {'smoothness_score': smoothness_score, 'vel': vel, 'outliers': outliers,
                'mean_vel': mean_vel, 'std_vel': std_vel}
    else:
        return smoothness_score

def calculate_LDLJ(positions=None, vel=None, print_results=False):
    assert not (positions is None and vel is None), "Pass in either positions or velocities"

    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

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

    if print_results:
        print("DLJ:", DLJ)
        print("LDLJ:", LDLJ)

    return LDLJ


def calculate_smoothness_for_all_demos(data, smoothness_fn=calculate_smoothness_from_vel, filter_success = False):
    smoothness_scores = []
    for demo in data['rollouts']:
        demo = data['rollouts'][demo]
        if filter_success:
            if not demo['rewards'][-1] > 0:
                continue

        positions = []
        for obs_list in demo['obs']:
            positions.append(obs_list['robot0_eef_pos'][-1])
        positions = np.array(positions)
        score = smoothness_fn(positions=positions)
        smoothness_scores.append(score)

    mean_smoothness = sum(smoothness_scores)/len(smoothness_scores)
    print(f"Mean smoothness : {mean_smoothness}")

# Run calculations
if __name__ == "__main__":
    # Can
    # log_fn = "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_rs_10_2024-12-13 17:09:07.418281.pkl"
    # log_fn = "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_rs_3_2024-12-13 17:05:54.060419.pkl"
    # log_fn = "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_2024-12-13 17:02:44.273838.pkl"
    # log_fn = "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_None_2024-12-13 16:39:38.890267.pkl"

    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/task_1000_gm_inpaint_2024-12-15 23:26:09.117408.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/square_image_action_horizon_16/20241211173329/logs/eval_over_kp_noisy_actions_2024-12-28 18:49:10.260924.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/inpainting+resampling_2024-12-27 20:44:00.288405.pkl"


    with open(log_fn, 'rb') as f:
        data = pickle.load(f)

    data = data[180]
        
    calculate_smoothness_for_all_demos(data, smoothness_fn=calculate_LDLJ)
    calculate_euclidean_distance(data, filter_success=True, disregard_inpainted_actions=True)
    calculate_dot_product(data, filter_success=True, disregard_inpainted_actions=False)
    distances = calculate_euclidean_distance_fn(data)

