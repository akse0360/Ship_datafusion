import os
import numpy as np
import torch

from datetime import datetime

# Model
from model.mlp import MyModel
# Data generation
from scripts.generation import Generator
# Plots
from scripts.plotter import Plotter
import matplotlib.pyplot as plt
# Movie generator:
import imageio.v2 as imageio


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



# Functions:
# Updated Plotter class
class Plotter:
    ## VISUALIZATION
    # Plot points
    def plot_tracks(track1, track2):
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(track1[0], track1[1], c='b', marker='o', label="Track 1")
        ax.scatter(track2[0], track2[1], c='r', marker='x', label="Track 2")
        ax.legend()
        return fig, ax

    # Plot connections with dynamic line width
    def plot_connection(ax, track1, track2, assignment, **kwargs):
        for i, j in enumerate(assignment):
            t1 = track1[i, :2]
            t2 = track2[j, :2]
            ax.plot([t1[0], t2[0]], [t1[1], t2[1]], **kwargs)

    ## EVALUATION PARAMETERS
    # Plot loss curve
    def plot_loss(losses):
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title("Training Loss Over Iterations")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        plt.show()
        return fig, ax

    # Plot Rank-N accuracy curve
    def plot_rank_accuracy(rank_accuracies, rank=5):
        fig, ax = plt.subplots()
        ax.plot(rank_accuracies)
        ax.set_title(f"Rank-{rank} Accuracy Over Iterations")
        ax.set_xlabel("Iterations")
        ax.set_ylabel(f"Rank-{rank} Accuracy")
        plt.show()
        return fig, ax

    def plot_iteration_with_loss_and_accuracy(p1, p2, diag_matrix, losses, rank_accuracies, loss, iteration, max_iter):
        """
        Creates a subplot with the track plot on top, the loss curve in the bottom left,
        and the rank-N accuracy curve in the bottom right.
        """
        # Create a figure and a GridSpec layout for custom positioning
        fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.3)

        # Top plot: Track connections (spanning both columns)
        ax1 = fig.add_subplot(gs[0, :])

        # Ensure only the first two columns are used for plotting
        ax1.scatter(p1[:, 0], p1[:, 1], c='b', marker='o', label="Track 1")
        ax1.scatter(p2[:, 0], p2[:, 1], c='r', marker='x', label="Track 2")

        # Ensure valid indices for assignment
        green_indices = np.arange(min(len(p1), len(p2)))  # Valid assignment with correct bounds
        Plotter.plot_connection(ax1, p1, p2, green_indices, color='g')

        # MLP connections
        d = torch.argmin(diag_matrix, dim=1).detach().numpy()
        d = np.clip(d, 0, len(p2) - 1)  # Clip to valid indices
        Plotter.plot_connection(ax1, p1, p2, d, color='r', linestyle='--')

        # Add title and legend
        ax1.set_title(f"Matching [pos] visualization, Iteration: {iteration}, Loss: {loss:.4f}")
        ax1.legend(loc='upper left', bbox_to_anchor=(1.009, 1), fancybox=True, shadow=True)

        # Bottom left plot: Loss curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(losses['train'], label="Training Loss", color='b')
        ax2.plot(losses['val'], label="Validation Loss", color='orange', linestyle='--')
        ax2.set_xlim(0, max_iter + 1)
        ax2.set_title("Training Loss Over Iterations")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Loss")
        ax2.legend(loc='upper right', fancybox=True, shadow=True)

        # Bottom right plot: Rank accuracy curves
        ax3 = fig.add_subplot(gs[1, 1])
        for rank, accuracies in rank_accuracies.items():
            ax3.plot(range(len(accuracies)), accuracies, label=f'Rank {rank}', marker='o', linestyle='dashed')
        ax3.set_xlim(0, (max_iter + 1) / 10)
        ax3.set_ylim(0, 1)
        ax3.set_title("Rank-N Accuracy Over Iterations")
        ax3.set_xlabel("Iterations, [i/10]")
        ax3.set_ylabel("Accuracy")
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fancybox=True, shadow=True)

        return fig, (ax1, ax2, ax3)
    
# Model folders
def clear_folder(currpath=None, models=None, clear_movies=False) -> dict:
    """
    Clear folder for frames or movies. If folders do not exist, function creates them.

    Args:
    - currpath (str): current path
    - models (str or list of str): folder(s) to clear
    - clear_movies (bool): if True, clear folder for movies, otherwise clear folder for frames

    Returns: dictionary with paths for frames and movies folders, keys: model, and a sub directory with keys: image, movies, values are paths to folders
    """

    if models is None:
        raise ValueError("No folder specified")

    if isinstance(models, str):
        models = [models]

    # Create result folder
    results_path = os.path.join(currpath, 'results')
    os.makedirs(results_path, exist_ok=True)

    paths = {}

    for model in models:
        model_folder = os.path.join(results_path, model)
        os.makedirs(model_folder, exist_ok=True)

        # Create paths to figure and movie folders
        figure_storage = os.path.join(model_folder, 'figures')
        video_storage = os.path.join(model_folder, 'movies')
        csv_storage = os.path.join(model_folder, 'csv')
        model_storage = os.path.join(model_folder, 'models')

        # Create folders, if they do not exist
        os.makedirs(figure_storage, exist_ok=True)
        os.makedirs(video_storage, exist_ok=True)
        os.makedirs(csv_storage, exist_ok=True)
        os.makedirs(model_storage, exist_ok=True)

        # Directory for saving frames, and cleaning it every run
        for item in os.listdir(figure_storage):
            os.remove(os.path.join(figure_storage, item))

        if clear_movies: # If movie is true, clean movies folder
            for item in os.listdir(video_storage):
                os.remove(os.path.join(video_storage, item))
        
        paths[model] = {'images': figure_storage, 'movies': video_storage, 'csv': csv_storage, 'models': model_storage}

    return paths

# Clear one folder
def clear_one_folder(folder = None) -> None:
    """
    Clear one folder.

    Args:
    - folder (str): folder to clear

    Returns: None
    """
    if folder is None:
        raise ValueError("No folder specified")
    else:
        os.makedirs(folder, exist_ok=True)
        # Directory for saving frames, and cleaning it every run
        for item in os.listdir(folder):
            os.remove(os.path.join(folder, item))

# Get time
def get_time() -> str:
    """
    Get current time in format: YYYYMMDDHHMMSS

    Returns: string with current time
    """
    # Get current time
    current_time = datetime.now()
    # Format the time as a string
    return current_time.strftime('%Y%m%dT%H%M%S')


def forward(t1, t2, model):
    e1 = model(torch.tensor(t1, dtype=torch.float32).T)
    e2 = model(torch.tensor(t2, dtype=torch.float32).T)
    return torch.cdist(e1,e2)

def forward_heading(t1, t2, model):
    e1 = model(torch.tensor(t1, dtype=torch.float32))
    e2 = model(torch.tensor(t2, dtype=torch.float32))
    return torch.cdist(e1,e2)


models = ['knn', 'pos', 'head']

# Set up the file paths
curpath = r'C:\Users\abelt\OneDrive\Dokumenter\GitHub\Ship_datafusion\playground'
os.makedirs(curpath, exist_ok=True)

paths = clear_folder(currpath= curpath, models=models, clear_movies=False)
formatted_time = get_time()

# Set up model and optimizer
np.random.seed(1)

maxiter = 2000
N = 2000
N_test = N//5
sigma_distance = 0.005
sigma_heading = 0.005
encode = False
missing_percentage = 0.0

ranks = [1, 3, 5]

# Generate validation data
# Position data:
p1, p2 = Generator.generate_data(N_test, sigma=sigma_distance)
ground_truth = np.arange(p1.shape[1])

# Heading data:
t1, t2 = Generator.calculate_heading(p1,p2, encoding=encode, missing_percentage=missing_percentage)

# Mask to simulate SAR and AIS data ratio
# mask = np.random.rand(len(p1)) > missing_percentage
# remove points in mask for p2

mlp_params = {
    'hidden_dim': 1024,
    'output_dim': 254,
    'n_layers': 5,
    'dropout': 0.2,
    'learning_rate': 0.00001,
}

# Initialize variables
frames = []
losses = {'train': [], 'val': []}

clear_one_folder(paths['pos']['images'])

# Create empty lists to store accuracies
rank_accuracies = {rank: [] for rank in ranks}

model = MyModel(input_dim=2, hidden_dim=mlp_params['hidden_dim'], 
            output_dim=mlp_params['output_dim'], depth=mlp_params['n_layers'], 
            drop_prob=mlp_params['dropout'])
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_params['learning_rate'])
# Training loop
for i in range(maxiter + 1):
    model.train()
    optimizer.zero_grad()
    p1t, p2t = Generator.generate_data(N, sigma=sigma_distance)
    
    diag_matrix = forward(p1t, p1t, model)
    
    # Calculate training loss and backpropagate
    diag = torch.diag(diag_matrix)
    train_loss = torch.sum(diag)
    train_loss.backward()
    losses['train'].append(train_loss.item())
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        diag_matrix_val = forward(p1, p2, model)

        diag_val = torch.diag(diag_matrix_val)
        val_loss = torch.sum(diag_val)
        losses['val'].append(val_loss.item())

    # Plot and save frame every 10 iterations
    if i % 10 == 0 or i == 1: # and i > 0
        # Rank-N evaluation
        for rank in ranks:
            accuracy = Generator.evaluate_rankN(diag_matrix_val, rank)
            rank_accuracies[rank].append(accuracy)

        # Plotting
        fig, (ax1, ax2, ax3) = Plotter.plot_iteration_with_loss_and_accuracy(
            p1.T, p2.T, diag_matrix_val, losses, rank_accuracies, train_loss.item(), i, maxiter
        )
        frame_path = os.path.join(paths['pos']['images'], f'frame_{i}.png')
        fig.savefig(frame_path)
        frames.append(imageio.imread(frame_path))
        plt.close(fig)

# Save the video
imageio.mimsave(os.path.join(paths['pos']['movies'], f'{formatted_time}_training_pos.mp4'), frames, fps=3)