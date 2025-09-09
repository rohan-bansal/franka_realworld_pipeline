"""
Utility functions for demo data
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import robomimic.utils.action_utils as action_utils
import robomimic.dev.hparam.utils.pkl_utils as pkl_utils
from copy import deepcopy

"""
Edit functions for demo data
"""
def get_pred_idx(demo, timestep):
    """
    Get the prediction index for a given timestep
    """
    # Get timing parameters
    Td = demo["kwargs"]["inf_delay"]
    Tne = demo["kwargs"]["execute_n_actions"] 
    Ta = Td + Tne
    Tp = demo["preds"][0].shape[0]
    
    # Find all prediction indices that contain this timestep
    pred_indices = []
    
    # Check predictions starting at i*Ta for i=0,1,2,...
    # Each prediction starts at i*Ta and covers Tp timesteps
    i = 0
    while True:
        if i == 0:
            pred_start = 0
        else:
            pred_start = Tne + (i-1) * Ta
        
        # If prediction starts after timestep, we're done
        if pred_start > timestep:
            break
            
        # If timestep falls within this prediction's range
        if pred_start <= timestep < pred_start + Tp:
            pred_indices.append((i, timestep - pred_start))
            
        i += 1

    ret = {
        "all": pred_indices,
        "cur": timestep // Ta,
    }

    return ret

def crop_demo(demo, start_idx, end_idx):
    """
    Crop demo data

    Args:
        demo (dict): Demo dictionary containing kwargs
        start_idx (int): Start index timestep
        end_idx (int): End index timestep
        
    Returns:
        dict: Cropped demo dictionary
    """
    N_executed_actions = demo["num_executed_actions"]
    assert start_idx < end_idx, f"Start index {start_idx} is greater than end index {end_idx}"
    assert end_idx < N_executed_actions, f"Start index {start_idx} is greater than the number of executed actions {N_executed_actions}"

    # copy the demo deepcopy
    demo_cropped = deepcopy(demo)
    
    # crop the demo non-obs
    demo_cropped["executed_actions"]     = demo["executed_actions"][start_idx:end_idx]
    demo_cropped["obs"]                  = demo["obs"][start_idx:end_idx]
    demo_cropped["num_executed_actions"] = end_idx - start_idx

    # deactivate predictions: do not use predictions for cropped demo
    demo_cropped["preds"] = None
    
    return demo_cropped

"""
Getter Functions for Demo Data
"""
def get_Ta(demo):
    """
    Get the actual `Ta` (total actions per plan)

    The `knot` that switches between plans occurs at i*Ta, where i is an integer.
    
    Args:
        demo (dict): Demo dictionary containing kwargs
        
    Returns:
        int: Total number of actions per plan (Ta = Td + Tne)
    """
    kwargs = demo["kwargs"]
    Td = kwargs["inf_delay"]  # Inference delay
    Tne = kwargs["execute_n_actions"]  # Number of actions to execute
    Ta = Td + Tne
    
    return Ta

def get_dt(demo):
    return 1/demo["kwargs"]["fast_control_freq"]

def get_end_effector_position(demo, type):
    """Get end effector positions
    
    Args:
        demo (dict): Demo dictionary containing observations or actions
        type (str): Either "real" for actual positions or "cmd" for commanded positions
        
    Returns:
        np.ndarray: Array of end effector positions with shape (T, 3)
        
    Raises:
        NotImplementedError: If type is not "real" or "cmd"
    """
    if type == "real":
        positions = np.array([step['robot0_eef_pos'][-1] for step in demo["obs"]])
    elif type == "cmd":
        positions = np.array(demo['executed_actions'])[:, :3]
    else:
        raise NotImplementedError(f"Position type {type} not implemented")
    
    return positions

def get_end_effector_velocity(demo, type):
    """Get end effector linear velocities
    
    Args:
        demo (dict): Demo dictionary containing observations or actions
        type (str): Either "real" for actual velocities or "cmd" for commanded velocities
        
    Returns:
        np.ndarray: Array of end effector velocities
    """
    dt = get_dt(demo)
    if type == "real":
        eef_vel = np.array([step['robot0_eef_vel_lin'][-1] for step in demo["obs"]])
    elif type == "cmd":
        eef_pos = get_end_effector_position(demo, type="cmd")
        eef_vel = (eef_pos[1:] - eef_pos[:-1])/dt
    
    return eef_vel

def get_end_effector_angular_velocity(demo, type):
    """Get end effector angular velocities
    
    Args:
        demo (dict): Demo dictionary containing observations
        type (str): Either "real" for actual velocities or "cmd" for commanded velocities
        
    Returns:
        np.ndarray: Array of end effector angular velocities
        
    Raises:
        NotImplementedError: If type is "cmd" since angular velocity computation not implemented
    """
    if type == "real":
        eef_angvel = np.array([step['robot0_eef_vel_ang'][-1] for step in demo["obs"]])
    elif type == "cmd":
        dt = get_dt(demo)
        eef_axisangles = np.array(demo['executed_actions'])[:, 3:6]
        eef_angvel = action_utils.compute_omega_base_frame(eef_axisangles, dt=dt)
    
    return eef_angvel


def get_timestamps(demo):
    """Get timestamps for each action
    """
    horizon = demo["num_executed_actions"]
    timestamps = np.arange(0, horizon)
    return timestamps

def get_knot_timestamps(demo):
    """Get timestamps for each knot

    A knot is the last point of a prediction before switching to the next plan.
    For a trajectory with N plans, there will be N knots at timestamps (i*Ta - 1)
    where i goes from 1 to N.
    
    Args:
        demo (dict): Demo dictionary containing execution information
        
    Returns:
        np.ndarray: Array of knot timestamps
    """
    Ta = get_Ta(demo)
    horizon = demo["num_executed_actions"]
    
    # Calculate number of complete plans
    n_plans = int(np.ceil(horizon / Ta))
    
    # Generate knot timestamps: Ta-1, 2Ta-1, ..., min((n_plans)*Ta-1, horizon-1)
    knot_timestamps = np.array([min(i*Ta - 1, horizon-2) for i in range(1, n_plans + 1)])
    
    return knot_timestamps

def get_end_effector_twist(demo, type):
    """Get end effector twist (combined linear and angular velocities)
    
    Args:
        demo (dict): Demo dictionary containing observations
        type (str): Either "real" for actual velocities or "cmd" for commanded velocities
        
    Returns:
        np.ndarray: Array of end effector twists with shape (T, 6)
        
    Raises:
        NotImplementedError: If type is "cmd" since angular velocity computation not implemented
    """
    if type == "real":
        eef_vel = np.array([step['robot0_eef_vel_lin'][-1] for step in demo["obs"]])
        eef_velang = np.array([step['robot0_eef_vel_ang'][-1] for step in demo["obs"]])
        
    elif type == "cmd":
        eef_vel    = get_end_effector_velocity(demo, type="cmd")
        eef_velang = get_end_effector_angular_velocity(demo, type="cmd")
    
    return np.concatenate([eef_vel, eef_velang], axis=-1)

if __name__ == "__main__":
    # pkl_file = "/home/wjung85/Repo/projects/FastIL/logs/hparam_debug/square_image_action_horizon_16_bl_iter_0_2025-01-15 18:13:04.989992.pkl"
    pkl_file = "/home/wjung85/Repo/projects/FastIL/logs/hparam_fast/square_image_action_horizon_16_fastinp_iter_0_2025-01-16 16:36:06.146728.pkl"
    demo = pkl_utils.load_result_pkl_file(pkl_file)["rollouts"]["demo_0"]
    # demo = crop_demo(demo, 0, 100)
    # get_pred_idx(demo, 9)
    end_eff_vel = get_end_effector_angular_velocity(demo, type="cmd")
    get_pred_idx(demo, 4)