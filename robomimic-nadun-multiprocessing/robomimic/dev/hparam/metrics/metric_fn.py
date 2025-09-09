import pickle
import numpy as np

import robomimic.dev.hparam.utils.demo_utils as demo_utils
import robomimic.dev.hparam.metrics.smoothness_metrics as SmoothnessMetrics
# from robomimic.dev.smoothness_metrics.compute_smoothness_metrics import compute_sparc, compute_LDLJ, compute_sparc_torch


"""
Auxilary Function
"""
velocity_dict = {
    "twist_cmd" : lambda demo : demo_utils.get_end_effector_twist(demo, type="cmd"),
    "twist_real" : lambda demo : demo_utils.get_end_effector_twist(demo, type="real"),
    "linvel_cmd" : lambda demo : demo_utils.get_end_effector_velocity(demo, type="cmd"),
    "linvel_real" : lambda demo : demo_utils.get_end_effector_velocity(demo, type="real"),
    "angvel_cmd" : lambda demo : demo_utils.get_end_effector_angular_velocity(demo, type="cmd"),
    "angvel_real" : lambda demo : demo_utils.get_end_effector_angular_velocity(demo, type="real"),
}

"""
Wrapper
"""
def success_wrapper(eval_fn):
    return lambda x : eval_fn(x) if x["success"] > 0 else 0

"""
Performance Evaluation (Stat-level)
"""
def average_success_eval_fn(stat) -> float:
    """ Compare success rate of the result dictionary
    """
    out = np.array(stat["Success_Rate"]).mean()
    return out

def average_horizon_eval_fn(stat) -> float:
    """ Evaluate the average horizon
    """
    out = np.array(stat["Horizon"]).mean()
    return out

"""
Smoothness Evaluation (Demo-level)
""" 
def average_inftime_eval_fn(demo) -> float:
    """ Evaluate the inference time
    """
    out = demo["total_inference_time"] / demo["num_inferences"]
    return out

def horizon_eval_fn(demo) -> float:
    return demo["num_executed_actions"]

def success_eval_fn(demo) -> float:
    return demo["success"]

def sim_time_eval_fn(demo) -> float:
    return demo['total_sim_time_elapsed']

def sparc_cumul_eval_fn(
    demo, 
    movement_type="twist_cmd", 
    padlevel=4, 
    fc=20, 
    amp_th=0.05, 
    time_horizon=None
) -> float:
    """
    Evaluate SPARC (Spectral Arc Length) based on end-effector velocity.
    
    Args:
        demo (dict): Demo dictionary containing trajectory data
        movement_type (str): Type of trajectory to analyze. Valid options: 
            ["twist_cmd", "twist_real", "linvel_cmd", "linvel_real", 
             "angvel_cmd", "angvel_real"]
        padlevel (int): Padding level for SPARC calculation
        fc (int): Frequency cutoff for SPARC calculation
        amp_th (float): Amplitude threshold for SPARC calculation
        time_horizon (int, optional): Number of timesteps to analyze. If None,
            uses the entire trajectory length.

    Returns:
        float: SPARC smoothness metric value. Lower values indicate smoother motion.
    """
    assert movement_type in velocity_dict.keys(), f"Invalid movement_type: {movement_type}"
    
    dt = demo_utils.get_dt(demo)

    if time_horizon is None:
        time_horizon = demo["num_executed_actions"]
    
    # Extract movements
    velocity_fn = velocity_dict[movement_type]
    velocity = velocity_fn(demo)
    speed = np.linalg.norm(velocity, axis=-1)

    speed = speed[:time_horizon]
    
    # Compute SPARC
    sparc = SmoothnessMetrics.sparc(
        movements=speed, 
        dt=dt, 
        padlevel=padlevel, 
        fc=fc, 
        amp_th=amp_th
    )

    return sparc

def sparc_cumul_upto_failure_eval_fn(
    demo, 
    movement_type="twist_cmd", 
    padlevel=4, 
    fc=20, 
    amp_th=0.05, 
) -> float:
    """
    Evaluate SPARC (Spectral Arc Length) based on end-effector velocity.
    
    Args:
        demo (dict): Demo dictionary containing trajectory data
        movement_type (str): Type of trajectory to analyze. Valid options: 
            ["twist_cmd", "twist_real", "linvel_cmd", "linvel_real", 
             "angvel_cmd", "angvel_real"]
        padlevel (int): Padding level for SPARC calculation
        fc (int): Frequency cutoff for SPARC calculation
        amp_th (float): Amplitude threshold for SPARC calculation
        time_horizon (int, optional): Number of timesteps to analyze. If None,
            uses the entire trajectory length.

    Returns:
        float: SPARC smoothness metric value. Lower values indicate smoother motion.
    """
    assert movement_type in velocity_dict.keys(), f"Invalid movement_type: {movement_type}"
    
    dt = demo_utils.get_dt(demo)

    time_horizon = int(failure_time_eval_fn(demo))
    
    # Extract movements
    velocity_fn = velocity_dict[movement_type]
    velocity = velocity_fn(demo)
    speed = np.linalg.norm(velocity, axis=-1)

    speed = speed[:time_horizon]
    
    # Compute SPARC
    sparc = SmoothnessMetrics.sparc(
        movements=speed, 
        dt=dt, 
        padlevel=padlevel, 
        fc=fc, 
        amp_th=amp_th
    )

    return sparc

def switch_delta_eval_fn(demo):
    """
    Evaluate the norm of the action delta at the "knot" of the trajectory
    "knot" indicates the timestamp where the action is switched
    """
    Ta = demo_utils.get_Ta(demo)
    knot_timestamps = demo_utils.get_knot_timestamps(demo)
    
    actions = np.array(demo["executed_actions"])

    delta_actions = np.diff(actions, axis=0)
    
    # Remove the gripper action
    delta_actions = delta_actions[:, :-1]
    
    speed = np.linalg.norm(delta_actions, axis=-1)

    speed_knot = speed[knot_timestamps]

    return speed_knot

def failure_time_eval_fn(demo, threshold = 1.5, freq_cutoff = 2, amp_th = 0.1, padlevel = 0, quantile = 0.25):
    """
    Failure time is defined by solving the inverse transformation of the high-frequency
    spectral energy.

    Args:
        freq_cutoff: the frequency cutoff for the high-frequency spectral energy
        amp_th: the amplitude threshold for the high-frequency spectral energy
        padlevel: the padding level for the fft
        threshold: the threshold for the high-frequency spectral energy
    """
    # parse the demo
    dt = demo_utils.get_dt(demo)
    
    # get the movements
    twist = demo_utils.get_end_effector_twist(demo, type="cmd")
    movements = np.linalg.norm(twist, axis=-1)

    # compute the fft
    nfft = int(pow(2, np.ceil(np.log2(len(movements))) + padlevel))
    f = np.fft.rfftfreq(nfft, d=dt)

    # Perform FFT and normalize amplitudes
    fft_result = np.fft.rfft(movements, nfft)
    Mf = abs(fft_result)
    Mf_normalized = Mf / Mf.max()

    # Create masks
    high_freq_mask = f > freq_cutoff
    mask_sel = (Mf_normalized > amp_th) & high_freq_mask

    # Apply masks and reconstruct signal
    fft_filtered = fft_result.copy()
    fft_filtered[~mask_sel] = 0

    # Filtered frequencies
    # print("Filtered frequencies:", f[mask_sel])

    # Reconstruct signal
    filtered_signal = np.fft.irfft(fft_filtered, nfft)
    filtered_signal = filtered_signal[:len(movements)]
    # filtered_ratio = filtered_signal / movements

    # Identify significant indices
    significant_indices = np.where(np.abs(filtered_signal) > threshold)[0]
    # heuristic: ignore the first 10 indices
    significant_indices = significant_indices[significant_indices > 10]

    if len(significant_indices) == 0:
        return demo["num_executed_actions"]
    else:
        ret = np.quantile(significant_indices, quantile)
        return int(ret)

def sparc_absolute_sliding_window_eval_fn(demo, sliding_window_size=2, movement_type="twist_cmd", padlevel=4, fc=20, amp_th=0.05) -> np.ndarray:
    """
    Evaluate the absolute SPARC within sliding windows.
    
    Computes the difference between SPARC values of consecutive windows to detect 
    changes in movement smoothness.

    Args:
        demo (dict): Demo dictionary containing trajectory data
        sliding_window_size (int): Size of sliding window, number of predictions in each window
        movement_type (str): Type of trajectory to analyze 
            (twist_cmd, twist_real, linvel_cmd, linvel_real, angvel_cmd, angvel_real)
        padlevel (int): Padding level for SPARC calculation (>= 0)
        fc (float): Frequency cutoff for SPARC (Hz)
        amp_th (float): Amplitude threshold for SPARC (0-1)
    
    Returns:
        np.ndarray: Array of SPARC differences within sliding windows
        
    Example:
        >>> demo = load_demo()
        >>> sparc_deltas = sparc_sliding_window_eval_fn(demo, sliding_window_size=3)
    """
    # Input validation
    assert movement_type in velocity_dict, f"Invalid movement_type: {movement_type}"
    assert sliding_window_size > 0, "sliding_window_size must be positive"
    assert padlevel >= 0, "padlevel must be non-negative"
    assert fc > 0, "frequency cutoff must be positive"
    assert 0 < amp_th < 1, "amplitude threshold must be between 0 and 1"
    
    # Get demo parameters
    dt = demo_utils.get_dt(demo)
    num_inf = demo['num_inferences']
    Ta = demo_utils.get_Ta(demo)
    
    # Extract movement data
    velocity_fn = velocity_dict[movement_type]
    velocity = velocity_fn(demo)
    speed = np.linalg.norm(velocity, axis=-1)
    
    def get_window_speed(start_idx, window_size):
        """Helper function to get speed data for a window"""
        window_start = max(0, Ta * start_idx - 1)
        window_end = Ta * (start_idx + window_size) - 1
        return speed[window_start:window_end]
    
    # Compute absolute SPARC
    sparc_vec = []
    for i in range(num_inf - sliding_window_size + 1):
        # Get speeds for current and next window
        speed_i = get_window_speed(i, sliding_window_size)
        
        # Calculate SPARC difference
        sparc_val = SmoothnessMetrics.sparc(speed_i, dt=dt, padlevel=padlevel, fc=fc, amp_th=amp_th)
        sparc_vec.append(sparc_val)

    return np.array(sparc_vec)

def sparc_sliding_window_eval_fn(demo, sliding_window_size=2, movement_type="twist_cmd", padlevel=4, fc=20, amp_th=0.05) -> np.ndarray:
    """
    Evaluate the increase of SPARC within sliding windows.
    
    Computes the difference between SPARC values of consecutive windows to detect 
    changes in movement smoothness.

    Args:
        demo (dict): Demo dictionary containing trajectory data
        sliding_window_size (int): Size of sliding window, number of predictions in each window
        movement_type (str): Type of trajectory to analyze 
            (twist_cmd, twist_real, linvel_cmd, linvel_real, angvel_cmd, angvel_real)
        padlevel (int): Padding level for SPARC calculation (>= 0)
        fc (float): Frequency cutoff for SPARC (Hz)
        amp_th (float): Amplitude threshold for SPARC (0-1)
    
    Returns:
        np.ndarray: Array of SPARC differences within sliding windows
        
    Example:
        >>> demo = load_demo()
        >>> sparc_deltas = sparc_sliding_window_eval_fn(demo, sliding_window_size=3)
    """
    # Input validation
    assert movement_type in velocity_dict, f"Invalid movement_type: {movement_type}"
    assert sliding_window_size > 0, "sliding_window_size must be positive"
    assert padlevel >= 0, "padlevel must be non-negative"
    assert fc > 0, "frequency cutoff must be positive"
    assert 0 < amp_th < 1, "amplitude threshold must be between 0 and 1"
    
    # Get demo parameters
    dt = demo_utils.get_dt(demo)
    num_inf = demo['num_inferences']
    Ta = demo_utils.get_Ta(demo)
    
    # Extract movement data
    velocity_fn = velocity_dict[movement_type]
    velocity = velocity_fn(demo)
    speed = np.linalg.norm(velocity, axis=-1)
    
    def get_window_speed(start_idx, window_size):
        """Helper function to get speed data for a window"""
        window_start = max(0, Ta * start_idx - 1)
        window_end = Ta * (start_idx + window_size) - 1
        return speed[window_start:window_end]
    
    # Compute SPARC differences
    sparc_delta_vec = []
    for i in range(num_inf - sliding_window_size + 1):
        # Get speeds for current and next window
        speed_curr = get_window_speed(i, sliding_window_size - 1)
        speed_next = get_window_speed(i, sliding_window_size)
        
        # Calculate SPARC difference
        sparc_next = SmoothnessMetrics.sparc(speed_next, dt=dt, padlevel=padlevel, fc=fc, amp_th=amp_th)
        sparc_curr = SmoothnessMetrics.sparc(speed_curr, dt=dt, padlevel=padlevel, fc=fc, amp_th=amp_th)
        sparc_delta_vec.append(sparc_next - sparc_curr)

    return np.array(sparc_delta_vec)

def sparc_knot(demo, sliding_window_size=2, movement_type="twist_cmd", padlevel=4, fc=20, amp_th=0.05) -> np.ndarray:
    """
    Evaluate the increase of SPARC within sliding windows

    Args:
        demo (dict): Demo dictionary
        sliding_window_size (int): Size of sliding window, number of predictions in each window (default: 2)
        movement_type (str): Type of trajectory to analyze (twist_cmd, twist_real, linvel_cmd, linvel_real, angvel_cmd, angvel_real)
        padlevel (int): Padding level for SPARC
        fc (int): Frequency cutoff for SPARC
        amp_th (float): Amplitude threshold for SPARC
    
    Returns:
        sparc_delta_vec (np.ndarray): Array of SPARC differences within sliding windows

    TODO:
        - Add post-failure analysis
        - Try sparc(actions[i:i+sliding_window_size]) - \sum sparc(actions[i:i+1])
    """
    assert movement_type in velocity_dict.keys(), f"Invalid type: {movement_type}"
    
    dt       = demo_utils.get_dt(demo)
    num_inf  = demo['num_inferences']
    Ta       = demo_utils.get_Ta(demo)
    
    # Extract movements
    velocity_fn = velocity_dict[movement_type]
    velocity    = velocity_fn(demo)
    speed       = np.linalg.norm(velocity, axis=-1)
    
    # Compute SPARC
    sparc_delta_vec = []
    for i in range(num_inf-sliding_window_size+1):
        # speed_i_full = speed[Ta*i:Ta*(i+sliding_window_size)-1]
        start_i = max(0, Ta*i-1)
        speed_i_full = speed[start_i:Ta*(i+sliding_window_size)-1]
        sparc_i_full = SmoothnessMetrics.sparc(speed_i_full, dt=dt, padlevel=padlevel, fc=fc, amp_th=amp_th)

        sparc_j_segment = []
        for j in range(sliding_window_size):
            start_ij = max(0, Ta*(i+j)-1)
            # speed_j = speed[Ta*(i+j):Ta*(i+j+1)-1]
            speed_j = speed[start_ij:Ta*(i+j+1)-1]
            sparc_j = SmoothnessMetrics.sparc(speed_j, dt=dt, padlevel=padlevel, fc=fc, amp_th=amp_th)
            sparc_j_segment.append(sparc_j)

        sparc_j_segment = np.array(sparc_j_segment)

        sparc_delta_vec.append(sparc_i_full - np.mean(sparc_j_segment))

    return np.array(sparc_delta_vec)


if __name__ == "__main__":
    pkl_path = "/home/wjung85/Repo/projects/FastIL/logs/hparam/square_image_action_horizon_16_cls_f_ts_20_eta_0_nres_0_loss_mse_wg_10_nsmc_1_stdmc_0_iter_0_2025-01-08 11:11:44.394144.pkl"
    with open(pkl_path, "rb") as f:
        data_peek = pickle.load(f)
    
    stat_peek = data_peek["stats"]
    demo_peek = data_peek["rollouts"]["demo_0"]

    # print(sparc_pos_cmd_eval_fn(demo_peek))
    # print(sparc_cumul_eval_fn(demo_peek, movement_type="twist_cmd"))
    # print(sparc_sliding_window_eval_fn(demo_peek, sliding_window_size=2, movement_type="twist_cmd"))
    print(sparc_cumul_upto_failure_eval_fn(demo_peek, movement_type="twist_cmd"))
    # print(sparc_cumul_upto_failure_eval_fn(demo_peek, movement_type="twist_cmd"))

    import robomimic.dev.hparam.utils.plot_utils as plot_utils
    import robomimic.dev.hparam.utils.demo_utils as demo_utils
    import robomimic.dev.hparam.metrics.smoothness_metrics as SmoothnessMetrics
    from robomimic.dev.hparam.eval_suite import DemoEvalSuite
    import robomimic.dev.hparam.utils.pkl_utils as pkl_utils


    result_dir         = "/home/wjung85/Repo/projects/FastIL/logs/hparam_debug"
    baseline_pkls = pkl_utils.get_result_pkl_files(result_dir, filter="bl")
    analyzer = DemoEvalSuite(baseline_pkls, lambda x: sparc_cumul_upto_failure_eval_fn(x), metric_name="sparc_all")