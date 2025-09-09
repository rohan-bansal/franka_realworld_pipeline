import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from robomimic.dev.eval.eval_inconsistency import calculate_LDLJ, calculate_dot_product, calculate_euclidean_distance_fn, calculate_smoothness_from_vel

# palette
# basic_colors = [
#     'red', 'blue', 'green', 'yellow', 'black', 'white',
#     'cyan', 'magenta', 'gray', 'purple', 'orange', 'brown',
#     'pink', 'olive', 'navy', 'teal', 'violet'
# ]

# pythonCopylight_colors = [
#     'lightblue', 'lightgreen', 'lightgray', 'lightcoral',
#     'lightyellow', 'lightpink', 'lightsalmon', 'lightseagreen',
#     'lightcyan', 'lightsteelblue'
# ]

# pythonCopydark_colors = [
#     'darkblue', 'darkgreen', 'darkred', 'darkcyan',
#     'darkmagenta', 'darkorange', 'darkviolet', 'darkgray',
#     'darkkhaki', 'darkolivegreen', 'darkseagreen', 'darkslateblue'
# ]


# pythonCopynamed_shades = [
#     'cornflowerblue', 'forestgreen', 'goldenrod',
#     'indianred', 'maroon', 'midnightblue',
#     'orangered', 'royalblue', 'sandybrown',
#     'seagreen', 'slateblue', 'steelblue',
#     'tomato', 'turquoise', 'crimson'
# ]

def plot_violin_chart_smoothness(data_files, smoothness_fn, labels=None, filter_success=False, colors=None):
    """
    Create multiple violin plots comparing smoothness metrics across different data files.
    
    Args:
        data_files (list): List of file paths to pickle files containing experiment data
        smoothness_fn (callable): Function to calculate smoothness (e.g., calculate_LDLJ or calculate_smoothness_from_vel)
        labels (list): Optional list of labels for each plot. If None, uses filenames
        filter_success (bool): Whether to only include successful trajectories
        colors (list): Optional list of colors for violin plots. If None, uses default colors
    """
    # Create a figure with multiple subplots
    n_plots = len(data_files)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    
    # If only one plot, wrap axes in a list for consistent indexing
    if n_plots == 1:
        axes = [axes]
    
    # Use default colors if none provided
    if colors is None:
        colors = sns.color_palette("husl", n_plots)
    
    # Use filenames as labels if none provided
    if labels is None:
        labels = [Path(file_path).stem for file_path in data_files]
    
    # Track min and max values for axis synchronization
    all_scores = []
    
    # First pass: collect all scores to determine axis limits
    for file_path in data_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        scores = []
        for demo_key in data['rollouts']:
            demo = data['rollouts'][demo_key]
            
            if filter_success and not demo['rewards'][-1] > 0:
                continue
                
            positions = []
            for obs_list in demo['obs']:
                positions.append(obs_list['robot0_eef_pos'][-1])
            positions = np.array(positions)
            
            score = smoothness_fn(positions=positions)
            scores.append(score)
        
        all_scores.extend(scores)
    
    # Calculate global axis limits
    global_min = np.min(all_scores)
    global_max = np.max(all_scores)
    y_margin = (global_max - global_min) * 0.3  # Add 10% margin
    
    # Second pass: create plots
    for idx, (file_path, label, color) in enumerate(zip(data_files, labels, colors)):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        smoothness_scores = []
        
        for demo_key in data['rollouts']:
            demo = data['rollouts'][demo_key]
            
            if filter_success and not demo['rewards'][-1] > 0:
                continue
                
            positions = []
            for obs_list in demo['obs']:
                positions.append(obs_list['robot0_eef_pos'][-1])
            positions = np.array(positions)
            
            score = smoothness_fn(positions=positions)
            smoothness_scores.append(score)
        
        # Create violin plot for this dataset
        sns.violinplot(data=smoothness_scores, ax=axes[idx], color=color)
        
        # Add individual points
        sns.swarmplot(data=smoothness_scores, color='red', size=4, alpha=0.5, ax=axes[idx])
        
        # Customize subplot
        axes[idx].set_title(label)
        axes[idx].set_ylabel('Smoothness Score' if idx == 0 else '')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
        # Set consistent y-axis limits
        axes[idx].set_ylim(global_min - y_margin, global_max + y_margin)
        
        # Calculate and display statistics
        mean_score = np.mean(smoothness_scores)
        std_score = np.std(smoothness_scores)
        axes[idx].axhline(y=mean_score, color='r', linestyle='--', alpha=0.5)
        axes[idx].text(0.05, 0.95, f'Mean: {mean_score:.2f}\nStd: {std_score:.2f}', 
                      transform=axes[idx].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig


def plot_violin_chart_euclidean(data_files, labels=None, filter_success=False, disregard_inpainted_actions=True, colors=None):
    """
    Create multiple violin plots comparing smoothness metrics across different data files.
    
    Args:
        data_files (list): List of file paths to pickle files containing experiment data
        smoothness_fn (callable): Function to calculate smoothness (e.g., calculate_LDLJ or calculate_smoothness_from_vel)
        labels (list): Optional list of labels for each plot. If None, uses filenames
        filter_success (bool): Whether to only include successful trajectories
        colors (list): Optional list of colors for violin plots. If None, uses default colors
    """
    # Create a figure with multiple subplots
    n_plots = len(data_files)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    
    # If only one plot, wrap axes in a list for consistent indexing
    if n_plots == 1:
        axes = [axes]
    
    # Use default colors if none provided
    if colors is None:
        colors = sns.color_palette("husl", n_plots)
    
    # Use filenames as labels if none provided
    if labels is None:
        labels = [Path(file_path).stem for file_path in data_files]
    
    # Track min and max values for axis synchronization
    all_scores = []
    
    # First pass: collect all scores to determine axis limits
    for file_path in data_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        distances = calculate_euclidean_distance_fn(data, filter_success=filter_success, disregard_inpainted_actions=disregard_inpainted_actions)

        all_scores.extend(distances)
    
    # Calculate global axis limits
    global_min = np.min(all_scores)
    global_max = np.max(all_scores)
    y_margin = (global_max - global_min) * 0.3  # Add 10% margin
    
    # Second pass: create plots
    for idx, (file_path, label, color) in enumerate(zip(data_files, labels, colors)):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        smoothness_scores = []
        
        distances = calculate_euclidean_distance_fn(data, filter_success=filter_success, disregard_inpainted_actions=disregard_inpainted_actions)
        smoothness_scores.extend(distances)
        
        # Create violin plot for this dataset
        sns.violinplot(data=smoothness_scores, ax=axes[idx], color=color)
        
        # Add individual points
        sns.swarmplot(data=smoothness_scores, color='red', size=4, alpha=0.5, ax=axes[idx])
        
        # Customize subplot
        axes[idx].set_title(label)
        axes[idx].set_ylabel('Smoothness Score' if idx == 0 else '')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
        # Set consistent y-axis limits
        axes[idx].set_ylim(global_min - y_margin, global_max + y_margin)
        
        # Calculate and display statistics
        mean_score = np.mean(smoothness_scores)
        std_score = np.std(smoothness_scores)
        axes[idx].axhline(y=mean_score, color='r', linestyle='--', alpha=0.5)
        axes[idx].text(0.05, 0.95, f'Mean: {mean_score:.2f}\nStd: {std_score:.2f}', 
                      transform=axes[idx].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Define your data files
    # square
    square_data_files = [
        "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_None_2024-12-13 16:42:46.866927.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_2024-12-13 16:59:20.285876.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_rs_3_2024-12-13 17:17:23.382599.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_rs_10_2024-12-13 17:12:53.182660.pkl"
    ]
    
    # can
    can_data_files = [
        "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_None_2024-12-13 16:39:38.890267.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_2024-12-13 17:02:44.273838.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_rs_3_2024-12-13 17:05:54.060419.pkl",
        "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_rs_10_2024-12-13 17:09:07.418281.pkl",
    ]
    
    labels = ["none", "rs_0", "rs_3", "rs_10"]
    colors = ["lightblue", "lightgreen", "lightsalmon", "lightcoral"]

    square_plot_dict = {
        "none":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_None_2024-12-13 16:42:46.866927.pkl",
            "color": "lightblue",
        },
        "rs_0":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_2024-12-13 16:59:20.285876.pkl",
            "color": "lightgreen",
        },
        "rs_3":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_rs_3_2024-12-13 17:17:23.382599.pkl",
            "color": "darkkhaki",
        },
        "rs_10":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_inpaint_rs_10_2024-12-13 17:12:53.182660.pkl",
            "color": "lightsalmon",
        },
        "cg_old":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_consistency_old_2024-12-13 19:03:37.732423.pkl",
            "color": "lightcoral",
        },
        "cg_new":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_consistency_10_2024-12-14 01:58:24.402815.pkl",
            "color": "darkblue",
        },
        "cg_resample":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_square_gm_consistency_50_rs10_2024-12-14 20:05:51.268035.pkl",
            "color": "olive"
        }
    }

    can_plot_dict = {
        "none":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_None_2024-12-13 16:39:38.890267.pkl",
            "color": "lightblue",
        },
        "rs_0":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_2024-12-13 17:02:44.273838.pkl",
            "color": "lightgreen",
        },
        "rs_10":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_inpaint_rs_10_2024-12-13 17:09:07.418281.pkl",
            "color": "lightsalmon",
        },
        "cg_old":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_consistency_old10_2024-12-13 19:21:56.746352.pkl",
            "color": "lightcoral",
        },
        "cg_new":{
            "path": "/home/wjung85/Repo/projects/FastIL/logs/task_can_gm_consistency_10_2024-12-14 01:41:53.518638.pkl",
            "color": "darkblue",
        }
    }
    
    # Create plot
    # fig = plot_violin_chart_smoothness(
    #     can_data_files, 
    #     calculate_smoothness_from_vel,
    #     labels=labels,
    #     colors=colors)

    plot_dict = can_plot_dict
    plot_dict = square_plot_dict


    labels     = plot_dict.keys()
    data_files = [v["path"] for (k, v) in plot_dict.items()]
    colors     = [v["color"] for (k, v) in plot_dict.items()]

    fig = plot_violin_chart_smoothness(
        data_files, 
        calculate_LDLJ,
        labels=labels,
        colors=colors)
    
    fig.savefig("violin_chart_smoothness_LDLJ.png", dpi=300, bbox_inches='tight')
    
    # fig = plot_violin_chart_euclidean(can_data_files, 
    #                                   labels=labels, 
    #                                   colors=colors)
    plt.show()