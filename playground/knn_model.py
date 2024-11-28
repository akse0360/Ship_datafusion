import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors


## Functions
# --------------------------- Matching Functions ---------------------------- #
def knn_matching(p1, p2, k=1):
    """
    Apply KNN to match p1 and p2 using haversine distance.
    """
    n1, n2 = p1.shape[1], p2.shape[1]
    distances = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            distances[i, j] = haversine_distance(p1[0, i], p1[1, i], p2[0, j], p2[1, j])
    knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
    knn.fit(distances)
    distances, indices = knn.kneighbors(distances, return_distance=True)
    return indices, distances

# ---------------------------- Utility Functions ---------------------------- #
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# --------------------------- Plotting Functions ---------------------------- #

def plot_tracks(ax, track1, track2, title="Track Matching Visualization"):
    """
    Plot two sets of tracks on the given axis.
    """
    ax.scatter(track1[0], track1[1], c='b', marker='o', label="Track 1")
    ax.scatter(track2[0], track2[1], c='r', marker='x', label="Track 2")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

def plot_connection(ax, track1, track2, assignment, **kwargs):
    """
    Plot lines connecting matched points between two tracks on the given axis.
    """   
    for i, j in enumerate(assignment):
        t1 = track1[:, i]
        t2 = track2[:, j]
        ax.plot([t1[0], t2[0]], [t1[1], t2[1]], **kwargs)


def plot_rank_percentage(ax, ground_truth, indices, ranks_to_plot):
    """
    Plots the average percentage of correct matches within specified ranks.
    """
    percentages = []
    for rank in ranks_to_plot:
        within_rank = np.any(indices[:, :rank] == ground_truth[:, None], axis=1)
        percentage = np.mean(within_rank) * 100  # Convert to percentage
        percentages.append(percentage)

    ax.scatter(ranks_to_plot, percentages, marker='.', color='b')
    ax.set_title("Average Percentage of Being Within the Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Average Percentage (%)")
    ax.set_xticks(ranks_to_plot)
    ax.set_ylim(0, 100)
    ax.grid(True)


def plot_knn_matching(p1, p2, ground_truth, indices):
    """
    Plot track matching visualization with KNN matching results.
    
    Args:
    - p1: numpy array, data points in p1.
    - p2: numpy array, data points in p2.
    - ground_truth: numpy array, ground truth indices.
    - indices: numpy array, predicted indices from KNN.
    """
    points1 = p1.T
    points2 = p2.T  
    # Create subplot grid
    fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi = 100, constrained_layout=True)
    # Plot track matching visualization in [0, 0]
    plot_tracks(axes,points1, points2, title="Track Matching Visualization, KNN Matching")
    # Plot a single dummy line for Ground Truth and KNN Match to include in the legend
    axes.plot([], [], color='g', linestyle='-', label="Ground Truth")
    axes.plot([], [], color='r', linestyle='--', label="KNN Match")

    plot_connection(axes, points1, points2, ground_truth, color='g', linestyle='-', label="Ground Match")
    plot_connection(axes, points1, points2, indices[:, 0], color='r', linestyle='--', label="KNN Match")

    axes.legend( ['Sensor 1','Sensor 2', 'Ground Truth', 'KNN match'], loc='upper left', bbox_to_anchor=(1.009, 1), fancybox=True, shadow=True)

    plt.show()
    return fig, axes

