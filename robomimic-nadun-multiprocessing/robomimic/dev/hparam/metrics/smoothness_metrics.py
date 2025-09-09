import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any, Callable
import matplotlib.pyplot as plt
import torch

import robomimic.dev.hparam.utils.demo_utils as demo_utils
import robomimic.dev.hparam.utils.pkl_utils as pkl_utils
import robomimic.dev.hparam.utils.stat_utils as stat_utils

def sparc(movements: np.ndarray, 
         dt: float, 
         padlevel: int = 4, 
         fc: float = 20.0, 
         amp_th: float = 0.05, 
         return_data: bool = False) -> Union[float, Tuple[float, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Compute SPARC (Spectral Arc Length) smoothness metric.
    
    Args:
        movements: Array of movement values
        dt: Time step between values 
        padlevel: Zero padding level for FFT
        fc: Frequency cutoff in Hz
        amp_th: Amplitude threshold for frequency selection
        return_data: Whether to return FFT data
        
    Returns:
        SPARC value if return_data=False
        Tuple of (SPARC value, (frequencies, magnitudes), (selected frequencies, selected magnitudes)) if return_data=True
    """
    assert movements.ndim == 1, "Movements must be 1D array"
    fs = int(1/dt)

    if len(movements) == 0:
        print("hi")
            
    # Compute padded FFT
    nfft = int(pow(2, np.ceil(np.log2(len(movements))) + padlevel))

    f = np.fft.rfftfreq(nfft) * fs
    Mf = abs(np.fft.rfft(movements, nfft))
    
    # Catch potential divide by zero warning
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        max_val = Mf.max()
        if max_val == 0:
            
            print(f"Warning: Mf.max() is {max_val}, cannot normalize")
        Mf = Mf / max_val
    

    # Apply frequency cutoff
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Apply amplitude threshold
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    if len(inx) < 2:
        return (0.0, (f, Mf), (f_sel, Mf_sel)) if return_data else 0.0
        
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Compute spectral arc length
    sparc = -np.sqrt(
        (np.diff(f_sel) / (f_sel[-1] - f_sel[0])) ** 2 + \
        (np.diff(Mf_sel))** 2
    ).sum()

    if return_data:
        return sparc, (f, Mf), (f_sel, Mf_sel)
    return sparc

def sparc_torch(movements: torch.Tensor, 
                dt: float, 
                padlevel: int = 4, 
                fc: float = 20.0, 
                amp_th: float = 0.05, 
                return_data: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """Compute SPARC (Spectral Arc Length) smoothness metric using PyTorch.
    
    Args:
        movements: Tensor of movement values [batch_size, sequence_length] or [sequence_length]
        dt: Time step between values 
        padlevel: Zero padding level for FFT
        fc: Frequency cutoff in Hz
        amp_th: Amplitude threshold for frequency selection
        return_data: Whether to return FFT data
        
    Returns:
        SPARC value if return_data=False
        Tuple of (SPARC value, (frequencies, magnitudes), (selected frequencies, selected magnitudes)) if return_data=True

    NOTE:
    - We can apply the softmax instead of the max to make it differentiable
    """
    # assert movements.ndim == 1, "Movements must be 1D array"
    if movements.dim() == 1:
        movements = movements.unsqueeze(0)  # Add batch dimension if not present
        
    batch_size = movements.shape[0]
    
    # Compute padded FFT
    nfft = int(pow(2, torch.ceil(torch.log2(torch.tensor(movements.shape[1]))) + padlevel))
    f  = torch.fft.rfftfreq(nfft, d=dt, device=movements.device)
    Mf = torch.abs(torch.fft.rfft(movements, nfft))
    
    # Normalize magnitudes
    max_vals = Mf.max(dim=1, keepdim=True)[0]
    max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)  # Avoid division by zero
    Mf = Mf / max_vals
    
    # Apply frequency cutoff
    fc_mask = (f <= fc)
    f_sel = f[fc_mask]
    Mf_sel = Mf[:, fc_mask]
    
    # Apply amplitude threshold
    amp_mask = (Mf_sel >= amp_th)
    
    # Initialize output tensor
    sparc = torch.zeros(batch_size, device=movements.device)
    
    # Compute SPARC for each sequence in batch
    for i in range(batch_size):
        # Get indices where amplitude is above threshold
        valid_indices = torch.where(amp_mask[i])[0]
        
        if len(valid_indices) < 2:
            sparc[i] = 0.0
            continue
            
        start_idx = valid_indices[0]
        end_idx = valid_indices[-1] + 1
        
        # Select frequency range
        curr_f_sel = f_sel[start_idx:end_idx]
        curr_Mf_sel = Mf_sel[i, start_idx:end_idx]
        
        # Compute normalized frequency differences
        f_diff = torch.diff(curr_f_sel) / (curr_f_sel[-1] - curr_f_sel[0])
        Mf_diff = torch.diff(curr_Mf_sel)
        
        # Compute spectral arc length
        sparc[i] = -torch.sqrt((f_diff ** 2 + Mf_diff ** 2)).sum()
    
    # Remove batch dimension if input was 1D
    if movements.shape[0] == 1:
        sparc = sparc[0]
        Mf = Mf[0]
        Mf_sel = Mf_sel[0]
    
    if return_data:
        return sparc, (f, Mf), (f_sel, Mf_sel)
    return sparc

# Smoothed version of SPARC
# def sparc_torch(movements: torch.Tensor, 
#                 dt: float, 
#                 padlevel: int = 4, 
#                 fc: float = 20.0, 
#                 amp_th: float = 0.05, 
#                 return_data: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
#     """Compute differentiable SPARC metric."""
#     if movements.dim() == 1:
#         movements = movements.unsqueeze(0)
        
#     batch_size = movements.shape[0]
    
#     # Compute FFT - maintains gradients
#     nfft = int(pow(2, torch.ceil(torch.log2(torch.tensor(movements.shape[1]))) + padlevel))
#     f = torch.fft.rfftfreq(nfft, d=dt, device=movements.device)
#     Mf = torch.abs(torch.fft.rfft(movements, nfft))
    
#     # Normalize with safe division - maintains gradients
#     max_vals = Mf.max(dim=1, keepdim=True)[0]
#     max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
#     Mf = Mf / max_vals
    
#     # Frequency cutoff using multiplication instead of indexing
#     fc_mask = (f <= fc).float()
#     Mf_sel = Mf * fc_mask.unsqueeze(0)
    
#     sparc = torch.zeros(batch_size, device=movements.device)
    
#     for i in range(batch_size):
#         # Use soft thresholding instead of hard indexing
#         amp_mask = torch.sigmoid((Mf_sel[i] - amp_th) * 100)  # sharp sigmoid
        
#         # Compute differences with masked values
#         f_diff = torch.diff(f * fc_mask) 
#         Mf_diff = torch.diff(Mf_sel[i] * amp_mask)
        
#         # Normalize frequency differences
#         f_range = torch.sum(fc_mask * f)
#         f_diff = f_diff / (f_range + 1e-8)
        
#         # Compute spectral arc length with masked values
#         arc_length = torch.sqrt((f_diff ** 2 + Mf_diff ** 2).sum() + 1e-8)
#         sparc[i] = -arc_length
    
#     if movements.shape[0] == 1 and not return_data:
#         sparc = sparc[0]
    
#     if return_data:
#         return sparc, (f, Mf), (f * fc_mask, Mf_sel)
#     return sparc

def analyze_trajectory_smoothness(demo_data: dict,
                                  velocity_fn: str = "eef_velocity",
                                  metric_fn: str = "sparc",
                                  dt: float = 0.05,
                                  type: str = "real") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze trajectory smoothness using specified metric.
    
    Args:
        demo_data: Dictionary containing demonstration data
        metric_fn: Smoothness metric to use ("sparc" or "ldlj")
        dt: Time step between values
        type: Type of trajectory to analyze ("real" or "cmd")
        
    Returns:
        Tuple of (success vector, smoothness scores, trajectory indices)
    """
    success_vec = demo_data["stats"]["Success_Rate"]
    smoothness_scores = []
    traj_indices = []
    
    velocity_fn_dict = {
        "eef_velocity": demo_utils.get_end_effector_velocity,
        "eef_angular_velocity": demo_utils.get_end_effector_angular_velocity,
        "eef_twist": demo_utils.get_end_effector_twist,
    }
    
    for i, demo in enumerate(demo_data["rollouts"].values()):
        velocity = velocity_fn_dict[velocity_fn](demo, type=type)
        speed    = np.linalg.norm(velocity, axis=-1)
        
        if metric_fn == "sparc":
            score = sparc(speed, dt)
        else:
            raise ValueError(f"Unknown metric function: {metric_fn}")
            
        smoothness_scores.append(score)
        traj_indices.append(i)
        
    return np.array(success_vec), np.array(smoothness_scores), np.array(traj_indices) 


def analyze_sparc_progression(demo_data: dict,
                              velocity_fn: str = "eef_velocity",
                              dt: float = 0.05,
                              type: str = "real",
                              n_skip: int = 2,
                              amp_th: float = 0.05,
                              padlevel: int = 4,
                              t_end: int = None) -> Dict[str, List]:
    """Analyze smoothness progression over time for each trajectory.
    
    Args:
        demo_data: Dictionary containing demonstration data
        velocity_fn: Type of velocity to analyze ("eef_velocity", "eef_angular_velocity", "eef_twist")
        metric_fn: Smoothness metric to use ("sparc" or "ldlj")
        dt: Time step between values
        type: Type of trajectory to analyze ("real" or "cmd")
        n_skip: Number of timesteps to skip between analyses
        amp_th: Amplitude threshold for SPARC
        padlevel: Zero padding level for FFT
        
    Returns:
        Dictionary containing:
            - success_vec: Success/failure for each trajectory
            - raw_ffts: List of (freqs, magnitudes) for each trajectory
            - filtered_ffts: List of (filtered_freqs, filtered_magnitudes) for each trajectory
            - sparc_progression: List of SPARC values over time for each trajectory
            - time_points: Timesteps where SPARC was evaluated
    """
    velocity_fn_dict = {
        "eef_velocity": demo_utils.get_end_effector_velocity,
        "eef_angular_velocity": demo_utils.get_end_effector_angular_velocity,
        "eef_twist": demo_utils.get_end_effector_twist,
    }
    
    # Get velocity and speed
    velocity = velocity_fn_dict[velocity_fn](demo_data, type=type)
    if t_end is not None:
        velocity = velocity[:t_end]
    speed = np.linalg.norm(velocity, axis=-1)
    
    # Get full trajectory SPARC with FFT data
    full_sparc, raw_fft, filtered_fft = sparc(
        speed, dt, 
        padlevel=padlevel,
        amp_th=amp_th,
        return_data=True
    )
    
    # Calculate SPARC progression
    t_eval = np.arange(n_skip, speed.shape[0], n_skip)
    sparc_progression = []
    
    for j in t_eval:
        sparc_val = sparc(
            speed[:j], dt,
            padlevel=padlevel,
            amp_th=amp_th
        )
        sparc_progression.append(sparc_val)
        
    result = {
        "full_sparc": (full_sparc, raw_fft, filtered_fft),
        "sparc_progression": sparc_progression,
        "time_points": t_eval,
    }
        
    return result