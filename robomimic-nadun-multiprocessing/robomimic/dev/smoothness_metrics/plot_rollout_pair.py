import pickle
import matplotlib.pyplot as plt

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def plot_trajectories(noisy_data, inpainting_data, noisy_key, inpainting_key):
    """Plot trajectories for a pair of keys."""
    noisy_trajectory = noisy_data['rollouts'][noisy_key]['obs']
    inpainting_trajectory = inpainting_data['rollouts'][inpainting_key]['obs']

    # Determine dimensions
    dimensions = len(noisy_trajectory[0]['robot0_eef_pos'][1])
    noisy_timesteps = range(len(noisy_trajectory))
    inpainting_timesteps = range(len(inpainting_trajectory))

    # Dimension names
    dimension_names = ['x', 'y', 'z']

    # Initialize plots
    fig, axes = plt.subplots(dimensions, 1, figsize=(10, 6))
    fig.suptitle(f'Trajectories: Noisy ({noisy_key}) vs Inpainting ({inpainting_key})')

    # Plot each dimension
    for dim in range(dimensions):
        noisy_dim = [step['robot0_eef_pos'][1][dim] for step in noisy_trajectory]
        inpainting_dim = [step['robot0_eef_pos'][1][dim] for step in inpainting_trajectory]

        axes[dim].plot(noisy_timesteps, noisy_dim, label='Noisy Trajectory', linestyle='--', marker='o')
        axes[dim].plot(inpainting_timesteps, inpainting_dim, label='Inpainting Trajectory', linestyle='-', marker='s')
        axes[dim].set_xlabel('Timestep')
        axes[dim].set_ylabel(f'{dimension_names[dim]}', rotation=0)
        axes[dim].legend()
        axes[dim].grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pairs(pickle_file, noisy_file, inpainting_file):
    """Print and plot the pairs."""
    # Load data
    data = load_pickle(pickle_file)
    noisy_data = load_pickle(noisy_file)
    inpainting_data = load_pickle(inpainting_file)
    
    for noisy_key, matches in data.items():
        print(f"Noisy Key: {noisy_key}")
        for match in matches:
            inpainting_key = match['inpainting_key']
            distance = match['distance']
            print(f"  Inpainting Key: {inpainting_key}, Distance: {distance}")

            # Plot the trajectories
            plot_trajectories(noisy_data, inpainting_data, noisy_key, inpainting_key)


# File paths
pickle_file = 'similar_rollout_pairs.pkl'
noisy_file = 'noisy.pkl'
inpainting_file = 'inpainting.pkl'

# Print and plot the pairs
plot_pairs(pickle_file, noisy_file, inpainting_file)

