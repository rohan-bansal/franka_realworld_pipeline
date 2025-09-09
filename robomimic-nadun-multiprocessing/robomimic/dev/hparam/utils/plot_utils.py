import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional, Callable
from robomimic.dev.hparam.utils import pkl_utils
from robomimic.dev.hparam.utils import demo_utils
import imageio
import os

import functools
from contextlib import contextmanager

def plot_decorator(func: Callable):
    """
    Decorator to enforce consistent plotting behavior across all plotting functions.
    Handles common plotting parameters like figsize, xlim, ylim, label, new_fig, and grid.
    
    Args:
        func: The plotting function to wrap
    
    The wrapped function should accept at least these kwargs:
        figsize (tuple): Figure size
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
        label (str): Legend label
        new_fig (bool): Whether to create new figure
        grid (bool): Whether to show grid
    """
    @functools.wraps(func)
    def wrapper(*args, 
                figsize=(9, 3),
                xlim=None,
                ylim=None,
                label=None,
                new_fig=True,
                grid=False,
                **kwargs):
        
        # Create new figure if requested
        if new_fig:
            plt.figure(figsize=figsize)
            
        # Call the original plotting function
        result = func(*args, 
                     **kwargs)
        
        # Apply common plotting settings
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        plt.grid(grid)
        
        if label:
            plt.legend()
            
        return result
    
    return wrapper

def plot_3d_decorator(func: Callable):
    """
    Decorator to enforce consistent 3D plotting behavior across all plotting functions.
    Handles common plotting parameters like figsize, xlim, ylim, zlim, labels, new_fig, and grid.
    
    Args:
        func: The plotting function to wrap
    
    The wrapped function should accept at least these kwargs:
        figsize (tuple): Figure size
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
        zlim (tuple): Z-axis limits
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        zlabel (str): Z-axis label
        new_fig (bool): Whether to create new figure
        grid (bool): Whether to show grid
        view_init (tuple): Initial viewing angle (elevation, azimuth)
    """
    @functools.wraps(func)
    def wrapper(*args,
                figsize=(10, 10),
                xlim=None,
                ylim=None,
                zlim=None,
                xlabel='X',
                ylabel='Y',
                zlabel='Z',
                labels=None,
                new_fig=True,
                grid=True,
                view_init=(90, 0),
                # video_writer=None,
                **kwargs):
        
        # Create new figure if requested
        if new_fig:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
            
        # Call the original plotting function
        result = func(*args, ax=ax, **kwargs)
        
        # Apply common plotting settings
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        
        if labels:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust the position of the legend
        
        ax.grid(grid)
        ax.set_box_aspect([1, 1, 1])
        
        if view_init is not None:
            if view_init == "xy":
                ax.view_init(elev=90, azim=0)
            elif view_init == "xz":
                ax.view_init(elev=0, azim=90)
            elif view_init == "yz":
                ax.view_init(elev=90, azim=90)
            else:
                ax.view_init(elev=view_init[0], azim=view_init[1])

        # if video_writer is not None:
        #     fig.canvas.draw()
        #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #     video_writer.append_data(image)

        #     return result, video_writer
        if new_fig:
            return result, fig
        else:
            return result
    
    return wrapper

@plot_3d_decorator
def plot_predictions_3d(preds, colormap = "Blues", ax=None, alpha=0.7):
    """
    Args:
        preds (list): (B, T, 3)
        ax (Axes3D, optional): The 3D axes to plot on

    Draw B lines of (T, 3) 3D trajectory with different colors
    """
    # Create different colors for each trajectory
    colors = plt.cm.get_cmap(colormap)(np.linspace(0.4, 1, len(preds)))
    
    # Plot each trajectory
    for pred, color in zip(preds, colors):
        # Plot the line
        ax.plot3D(pred[:, 0], pred[:, 1], pred[:, 2], color=color, alpha=alpha)
        
        # Plot start point as circle
        ax.scatter(pred[0, 0], pred[0, 1], pred[0, 2], color=color, marker='o')
        
        # Plot end point as star
        ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], color=color, marker='*', s=200)
    
    return ax

@plot_decorator
def plot_predictions(demo, action_idx, pred_key = "preds", alpha_pred = 0.6, alpha_inpaint = 1.0):
    """
    Plot the predictions of a demo
    Args:
        demo (dict): The demo to plot
        action_idx (int): The index of the action to plot
    """
    Tne = demo['kwargs']['execute_n_actions']
    Td  = demo['kwargs']['inf_delay']
    Ta  = Tne + Td

    preds = demo[pred_key]

    timesteps = []
    actions = []

    for i, pred in enumerate(preds):
        if i == 0:
            start_timestep = 0
        else:
            start_timestep = (i - 1) * Ta + Tne
        timesteps.append(np.arange(start_timestep, start_timestep + pred.shape[0]))
        actions.append(pred[:, action_idx])

    # viridis is a perceptually uniform colormap that's also colorblind-friendly
    # other good options: plasma, magma, inferno, cividis
    color_codes = np.linspace(0, 1, len(timesteps))
    # np.random.shuffle(color_codes)
    colors = plt.cm.tab20(color_codes)
    
    for i, timestep in enumerate(timesteps):
        if i == 0:
            plt.plot(timestep[:Tne+1], actions[i][:Tne+1], '-', color=colors[i], alpha=alpha_pred, linewidth=3)
            plt.plot(timestep[Tne:], actions[i][Tne:], 'o-', color=colors[i], alpha=alpha_pred, linewidth=3)
        else:
            plt.plot(timestep[:Td+1], actions[i][:Td+1], 'x-', color=colors[i], alpha=alpha_inpaint, linewidth=3)
            plt.plot(timestep[Td:Ta+1], actions[i][Td:Ta+1], '-', color=colors[i], alpha=alpha_pred, linewidth=3) 
            plt.plot(timestep[Ta:], actions[i][Ta:], 'o-', color=colors[i], alpha=alpha_pred, linewidth=3)

    return timesteps, actions

def plot_smoothness_analysis(results: List[Dict], title: str = "", figsize: Tuple[int, int] = (18, 6), ylim_smoothness: Tuple[float, float] = (0, 1)) -> None:
    """Plot smoothness analysis results.
    
    Args:
        results: List of analysis results from analyze_smoothness_progression
        title: Title for the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    for result in results:
        color = "green" if result["success"] else "red"
        alpha = 0.2 if result["success"] else 0.5
        linestyle = "-." if result["success"] else "-"
        
        # Plot raw FFT
        freqs, mags = result["full_sparc"][1]
        ax1.plot(freqs, mags, color=color, alpha=alpha, linestyle=linestyle)
        
        # Plot filtered FFT
        freqs_sel, mags_sel = result["full_sparc"][2]
        ax2.plot(freqs_sel, mags_sel, color=color, alpha=alpha, linestyle=linestyle)
        
        # Plot SPARC progression
        ax3.plot(result["time_points"], result["sparc_progression"], 
                color=color, alpha=alpha, linestyle=linestyle)
    
    ax1.set_title("Raw FFT")
    ax2.set_title("Filtered FFT")
    ax3.set_title("SPARC Progression")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    ax3.set_xlim(0, 500)
    ax3.set_ylim(ylim_smoothness)
    
    for ax in (ax1, ax2, ax3):
        ax.grid(True)
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()

def plot_smoothness_video(results_list: List[List[Dict]], save_path: str = "smoothness_analysis.mp4", fps: int = 20) -> None:
    """
    Create a video from the results list using plot_smoothness_analysis.

    Args:
        results_list: List of results for each frame.
        save_path: Path to save the generated video.
        fps: Frames per second for the video.
    """
    # Ensure the directory for saving exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create a video writer
    writer = imageio.get_writer(save_path, fps=fps)

    # Create a single figure for all frames with margins
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    for frame_idx, results in enumerate(results_list):
        # Clear the axes for the next frame
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Plot the smoothness analysis for this frame
        for result in results:
            color = "green" if result["success"] else "red"
            alpha = 0.7 if result["success"] else 0.8
            linestyle = "-." if result["success"] else "-"
            linewidth = 3
            
            # Plot raw FFT
            freqs, mags = result["full_sparc"][1]
            ax1.plot(freqs, mags, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            
            # Plot filtered FFT
            freqs_sel, mags_sel = result["full_sparc"][2]
            ax2.plot(freqs_sel, mags_sel, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            
            # Plot SPARC progression
            ax3.plot(result["time_points"], result["sparc_progression"], 
                     color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        
        ax1.set_title("Raw FFT")
        ax2.set_title("Filtered FFT")
        ax3.set_title("SPARC Progression")
        ax1.set_xlim(0, 20)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 20)
        ax2.set_ylim(0, 1)
        ax3.set_xlim(0, 400)
        ax3.set_ylim(-15, 0)
        for ax in (ax1, ax2, ax3):
            ax.grid(True)
        
        # Add title for the frame
        fig.suptitle(f"Frame {frame_idx}")
        plt.tight_layout()

        # Capture the current frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)

    # Close the writer and figure
    plt.close(fig)
    writer.close()

    print(f"Video saved to {save_path}")

def plot_success_rate_comparison(speed_to_pkl_files, figsize=(12, 6), save_path=None):
    """
    Plot success rate comparison across different speeds and methods.
    
    Args:
        speed_to_pkl_files (dict): Dictionary mapping speeds to pkl files for each method
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the figure
        
    Returns:
        fig: matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from robomimic.dev.hparam.eval_suite import DemoEvalSuite
    import robomimic.dev.hparam.metrics.metric_fn as Metrics

    # Method labels and their corresponding keys in pkl files
    labels = [
        'Uncond. Baseline',
        'Uncond. + Inpainting',
        'Cond. + CFG',
        'Cond. + Inpainting',
        'Cond. + No Guidance'
    ]
    
    method_keys = [
        'uncond_no_guide',
        'uncond_inp',
        'cond_cfg',
        'cond_inp',
        'cond_no_guide'
    ]

    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Position calculations
    x = np.arange(len(labels))
    width = 0.25  # Width of bars
    
    # Colors for different speeds
    colors = {
        '1x': '#2ecc71',
        '2x': '#3498db',
        '3x': '#e74c3c',
        '4x': '#9b59b6',
        '5x': '#f1c40f'
    }
    
    # Plot bars for each speed
    speeds = sorted(speed_to_pkl_files.keys())
    num_speeds = len(speeds)
    offsets = np.linspace(-width, width, num_speeds)
    
    for speed, offset in zip(speeds, offsets):
        means = []
        yerr_minus = []
        yerr_plus = []
        
        for method_key in method_keys:
            # Get results for this method and speed
            results = speed_to_pkl_files[speed][method_key]
            if not results:  # Skip if no results
                means.append(np.nan)
                yerr_minus.append(np.nan)
                yerr_plus.append(np.nan)
                continue
                
            # Analyze success rates
            analyzer = DemoEvalSuite(results, Metrics.success_eval_fn, metric_name="success")
            success_rates = analyzer.df.groupby(by='result_idx')['success'].mean()
            
            mean = success_rates.mean()
            min_val = success_rates.min()
            max_val = success_rates.max()
            
            means.append(mean)
            yerr_minus.append(mean - min_val)
            yerr_plus.append(max_val - mean)
        
        plt.bar(x + offset, means, width, 
               label=f'{speed} Speed', 
               color=colors[speed], 
               alpha=0.7)
        plt.errorbar(x + offset, means, 
                    yerr=[yerr_minus, yerr_plus], 
                    fmt='none', 
                    color='black', 
                    capsize=5)

    # Customize plot
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison Across Policy Classes and Speeds')
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend()

    # Set y-axis limits
    plt.ylim(0, 1.0)

    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

if __name__ == "__main__":
    import robomimic.dev.hparam.metrics.smoothness_metrics as SmoothnessMetrics
    import robomimic.dev.hparam.utils.plot_utils as plot_utils
    
    
    amp_th = 0.05
    dt = 0.05
    n_skip = 3
    type = "cmd"
    result_idx = 0
    demo_idx   = 5

    # pkl_path = "/home/wjung85/Repo/projects/FastIL/logs/hparam/square_image_action_horizon_16_inp_t_ts_40_eta_1_nres_1_iter_0_2025-01-08 14:01:21.391115.pkl"
    # data_picked = pkl_utils.load_result_pkl_file(pkl_path)
    
    result_dir         = "/home/wjung85/Repo/projects/FastIL/logs/hparam_debug"
    pkl_files_inp      = pkl_utils.get_result_pkl_files(result_dir, filter="inp")
    data_picked = pkl_utils.load_result_pkl_file(pkl_files_inp[result_idx])
    demo_data   = data_picked["rollouts"][f"demo_{demo_idx}"]

    results_to_animate = []
    N_timestep = demo_data["num_executed_actions"]
    t_end_vec = np.arange(10, N_timestep, 5)
    for t_end in t_end_vec:
        results_eef = SmoothnessMetrics.analyze_sparc_progression(
            demo_data,
            velocity_fn="eef_velocity",
            dt=dt,
            type=type,
            n_skip=n_skip,
            amp_th=amp_th,
            t_end=t_end if t_end < N_timestep else None
        )
        results_eef["success"] = data_picked["stats"]["Success_Rate"][demo_idx]
        
        results_to_animate.append([results_eef])
        
    plot_smoothness_video(results_to_animate, save_path=os.path.join(os.path.dirname(__file__), "smoothness_analysis.mp4"), fps=20)
    

    