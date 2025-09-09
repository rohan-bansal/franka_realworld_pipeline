import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from fastdtw import fastdtw

# Example input tensor
# B = batch size, T = trajectory length, D = trajectory dimension

# Measure 1: Variance Across Trajectories
def compute_variance(trajectories):
    return np.mean(np.var(trajectories, axis=0))  # Variance over B

# Measure 2: Average Pairwise Distance
def compute_pairwise_distance(trajectories):
    B, T, D = trajectories.shape
    distances = []
    for i in range(B):
        for j in range(i + 1, B):
            dist = np.mean(np.linalg.norm(trajectories[i] - trajectories[j], axis=-1))  # Average trajectory distance
            distances.append(dist)
    return np.mean(distances)

# Measure 3: Dynamic Time Warping (DTW)
def compute_dtw_distance(trajectories):
    B, T, D = trajectories.shape
    distances = []
    for i in range(B):
        for j in range(i + 1, B):
            dist, _ = fastdtw(trajectories[i], trajectories[j], dist=lambda x, y: np.linalg.norm(x - y))
            distances.append(dist)
    return np.mean(distances)

# Measure 4: Kullback-Leibler (KL) Divergence
def compute_kl_divergence(trajectories):
    B, T, D = trajectories.shape

    flattened = trajectories.reshape(B, -1)  # Flatten trajectories into B x (T*D)
    histograms = np.array([np.histogram(traj, bins=20, range=(flattened.min(), flattened.max()), density=True)[0] for traj in flattened])
    kl_divergences = []
    for i in range(B):
        for j in range(i + 1, B):
            kl = entropy(histograms[i] + 1e-8, histograms[j] + 1e-8)  # Add epsilon for numerical stability
            kl_divergences.append(kl)
    return np.mean(kl_divergences)

# Measure 5: Frechet Distance
def compute_frechet_distance(trajectories):
    B, T, D = trajectories.shape

    def frechet(traj1, traj2):
        dist_matrix = cdist(traj1, traj2)
        return np.max(np.min(dist_matrix, axis=0))

    distances = []
    for i in range(B):
        for j in range(i + 1, B):
            dist = frechet(trajectories[i], trajectories[j])
            distances.append(dist)
    return np.mean(distances)



def main(trajectories):
    return {
        "variance": compute_variance(trajectories),
        "pairwise_distance": compute_pairwise_distance(trajectories),
        "dtw_distance": compute_dtw_distance(trajectories),
        "kl_divergence": compute_kl_divergence(trajectories),
        "frechet_distance": compute_frechet_distance(trajectories),
    }

if __name__ == "__main__":
    # Example Usage
    trajectories = np.random.rand(100, 50, 3)  # Example (B=100, T=50, D=3)

    print(main(trajectories))
