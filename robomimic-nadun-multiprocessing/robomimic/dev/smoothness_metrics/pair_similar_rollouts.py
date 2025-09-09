import pickle
import numpy as np
from scipy.spatial.distance import cdist

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def find_top_k_similar(noisy_file, inpainting_file, output_file, k=3):
    # Load data from pickle files
    noisy_data = load_pickle(noisy_file)
    inpainting_data = load_pickle(inpainting_file)

    # Extract rollouts and initial states
    noisy_rollouts = noisy_data['rollouts']
    inpainting_rollouts = inpainting_data['rollouts'] 

    noisy_initial_states = [
        (key, np.array(rollout['initial_state_dict']['states']))
        for key, rollout in noisy_rollouts.items()
    ]
    
    inpainting_initial_states = [
        (key, np.array(rollout['initial_state_dict']['states']))
        for key, rollout in inpainting_rollouts.items()
    ]

    # Prepare arrays for comparison
    noisy_keys, noisy_states = zip(*noisy_initial_states)
    inpainting_keys, inpainting_states = zip(*inpainting_initial_states)

    noisy_states = np.array(noisy_states)
    inpainting_states = np.array(inpainting_states)

    # Calculate pairwise distances
    distances = cdist(noisy_states, inpainting_states, metric='euclidean')

    # Find top-k similar rollouts for each noisy rollout
    similar_rollouts = {}
    for i, noisy_key in enumerate(noisy_keys):
        top_k_indices = np.argsort(distances[i])[:k]
        similar_rollouts[noisy_key] = [
            {
                'inpainting_key': inpainting_keys[idx],
                'distance': distances[i][idx]
            }
            for idx in top_k_indices
        ]

    # Save the results
    save_pickle(similar_rollouts, output_file)

# File paths
noisy_file = 'noisy.pkl'
inpainting_file = 'inpainting.pkl'
output_file = 'similar_rollout_pairs.pkl'

# Top k similar rollouts
k = 3

# Find and save similar rollouts
find_top_k_similar(noisy_file, inpainting_file, output_file, k)

