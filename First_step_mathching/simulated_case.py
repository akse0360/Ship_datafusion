# TODO MAKE The connections to all but with a size differnce

import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio.v2 as imageio
import os
from datetime import datetime
from machine.mlp import MyModel
import matplotlib.animation as animation

# Set up the file paths
curpath = r'C:\Users\abelt\OneDrive\Dokumenter\GitHub\Ship_datafusion\First_step_mathching'
folderpath = os.path.join(curpath, 'frames')
os.makedirs(folderpath, exist_ok=True)

# Normalize the data (optional)
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Generate synthetic data
def generate_data(N, sigma): # Position only
    g = np.random.Generator(np.random.PCG64())
    x = g.uniform(0, 1, N)
    y = g.uniform(0, 1, N)
    x1 = np.clip(g.normal(x, sigma), a_min=0, a_max=1)  
    y1 = np.clip(g.normal(y, sigma), a_min=0, a_max=1)
    p1 = np.stack((x, y))
    p2 = np.stack((x1, y1))
    return p1, p2

def calculate_heading(p1, p2, degrees=False, sigma=0.1):
    headings = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    if degrees:
        headings = np.degrees(headings)
    
    headings1 = headings + np.random.normal(0, sigma, len(headings))
    headings2= headings + np.random.normal(0, sigma, len(headings))
    return headings1, headings2


# Clear folder for frames
def clear_folder(folder = folderpath):
    os.makedirs(folderpath, exist_ok=True)
    # Directory for saving frames, and cleaning it every run
    for item in os.listdir(folder):
        os.remove(os.path.join(folderpath, item))

def evaluate_rankN(dist_matrix, n=5):
    # Ranking evaluation function
    def rankN(scores, labels, n=5):
        sort = torch.argsort(scores, dim=1, descending=True)
        sorted_labels = labels.gather(1, sort)
        sorted_labels = sorted_labels[torch.any(sorted_labels, dim=1)]
        correct = torch.any(sorted_labels[:, 0:n], dim=1)
        return correct.float().mean()
    
    distance_matrix = dist_matrix 
    labels = torch.eye(distance_matrix.size(0), dtype=torch.float32)
    scores = -distance_matrix  

    rank_accuracy = rankN(scores, labels, n)
    return rank_accuracy

# Forward pass
def forward(p1,p2,model):
    t1 = torch.tensor(p1.T, dtype=torch.float32)
    t2 = torch.tensor(p2.T, dtype=torch.float32)
    e1 = model(t1)
    e2 = model(t2)
    #diag_matrix = 
    return torch.cdist(e1, e2)

## Plot functions
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
        t1 = track1.T[i]
        t2 = track2.T[j]
        #dist = np.linalg.norm(t1 - t2)
        ax.plot([t1[0], t2[0]], [t1[1], t2[1]],  **kwargs) #,linewidth=2.5/(dist + 1.3)

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

# Plot the frames only with loss:
def plot_iteration_with_loss(p1, p2, diag_matrix, green_indices, losses, loss, iteration, max_iter) -> tuple:
    """
    Creates a subplot with the track plot on top and the loss curve below.

    Parameters:
        p1 (ndarray): Points for the first track, shape (2, N).
        p2 (ndarray): Points for the second track, shape (2, N).
        diag_matrix (ndarray): Diagonal matrix of the distance matrix.
        green_indices (array-like): Indices for green connections (e.g., correct matches).
        losses (list of float): List of loss values over training iterations.
        loss (float): Loss value for the current iteration.
        iteration (int): Current iteration number.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        fig (Figure): The matplotlib figure object containing both subplots.
        (ax1, ax2): Tuple of axes objects for the track and loss plots.
    """
    # Create a 2-row subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top plot: Track connections
    ax1.scatter(p1[0], p1[1], c='b', marker='o', label="Track 1")
    ax1.scatter(p2[0], p2[1], c='r', marker='x', label="Track 2")
    
    plot_connection(ax1, p1, p2, green_indices, color='g')

    d = torch.argmin(diag_matrix, dim=1).detach().numpy()
    plot_connection(ax1, p1, p2, d, color='r')
    ax1.set_title(f"Iteration: {iteration}, Loss: {loss:.4f}")
    
    # Adjust the axis position for legend
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=2)

    # Bottom plot: Loss curve
    ax2.plot(losses, label="Training Loss", color='b')
    ax2.set_xlim(0, max_iter+1) 
    ax2.set_title("Training Loss Over Iterations")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    ax2.legend(fancybox=True, shadow=True)

    return fig, (ax1, ax2)

# Plot the frames
def plot_iteration_with_loss_and_accuracy(p1, p2, diag_matrix, green_indices, losses, rank_accuracies, loss, iteration, max_iter):
    """
    Creates a subplot with the track plot on top, the loss curve in the bottom left, and the rank-N accuracy curve in the bottom right.

    Parameters:
        p1 (ndarray): Points for the first track, shape (2, N).
        p2 (ndarray): Points for the second track, shape (2, N).
        diag_matrix (ndarray): Diagonal matrix of the distance matrix.
        green_indices (array-like): Indices for green connections (e.g., correct matches).
        losses (list of float): List of loss values over training iterations.
        rank_accuracies (dict of lists): Dictionary where keys are rank values and values are lists of accuracy values over iterations.
        loss (float): Loss value for the current iteration.
        iteration (int): Current iteration number.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        fig (Figure): The matplotlib figure object containing all three subplots.
        (ax1, ax2, ax3): Tuple of axes objects for the track, loss, and accuracy plots.
    """
    # Create a figure and a GridSpec layout for custom positioning
    fig = plt.figure(figsize=(12, 8), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.3)
    
    # Top plot: Track connections (spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(p1[0], p1[1], c='b', marker='o', label="Track 1")
    ax1.scatter(p2[0], p2[1], c='r', marker='x', label="Track 2")
    plot_connection(ax1, p1, p2, green_indices, color='g')
    
    d = torch.argmin(diag_matrix, dim=1).detach().numpy()
    plot_connection(ax1, p1, p2, d, color='r')
    ax1.set_title(f"Iteration: {iteration}, Loss: {loss:.4f}")
    
    # Adjust the legend position outside the plot
    ax1.legend(loc='upper left', bbox_to_anchor=(1.009, 1), fancybox=True, shadow=True)

    # Bottom left plot: Loss curve
    ax2 = fig.add_subplot(gs[1, 0])
    #ax2.plot(losses, label="Training Loss", color='b')
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
        ax3.plot(range(len(accuracies)), accuracies, label=f'Rank {rank}',  marker='o', linestyle='dashed')
    ax3.set_xlim(0, (max_iter + 1) / 10)
    ax3.set_ylim(0, 1)
    ax3.set_title("Rank-N Accuracy Over Iterations")
    ax3.set_xlabel("Iterations, [i/10]")
    ax3.set_ylabel("Accuracy")
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fancybox=True, shadow=True)
    
    return fig, (ax1, ax2, ax3)


if __name__ == '__main__':
    # Set up model and optimizer
    maxiter = 200
    N = 300
    N_test = N // 4
    sigma = 0.075

    ranks = [1, 3, 5]


    mlp_params = {
        'input_dim': 2,
        'hidden_dim': 1024,
        'output_dim': 254,
        'n_layers': 5,
        'dropout': 0.1,
        'learning_rate': 0.0001,
    }
    # Directory for saving frames, and cleaning it every run
    curpath = r'C:\Users\abelt\OneDrive\Dokumenter\GitHub\Ship_datafusion\First_step_mathching'
    folderpath = os.path.join(curpath, 'frames')
    
    clear_folder(folder=folderpath)

    # Initialize variables
    frames = []
    losses = {'train': [], 'val': []}

    # Create empty lists to store accuracies
    rank_accuracies = {rank: [] for rank in ranks}

    # Generate validation data
    p1, p2 = generate_data(N_test, sigma)

    model = MyModel(input_dim=mlp_params['input_dim'], hidden_dim=mlp_params['hidden_dim'], 
                output_dim=mlp_params['output_dim'], depth=mlp_params['n_layers'], 
                drop_prob=mlp_params['dropout'])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_params['learning_rate'])
    # Training loop
    for i in range(maxiter + 1):
        model.train()
        optimizer.zero_grad()
        p1t, p2t = generate_data(N, sigma)
        diag_matrix = forward(p1t, p2t, model)
        
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
                accuracy = evaluate_rankN(diag_matrix, rank)
                rank_accuracies[rank].append(accuracy.item())

            d = torch.argmin(diag_matrix_val, dim=1).detach().numpy()

            # Plotting
            fig, (ax1, ax2, ax3) = plot_iteration_with_loss_and_accuracy(
                p1, p2, diag_matrix_val, np.arange(N_test), losses, rank_accuracies, train_loss.item(), i, maxiter
            )
            frame_path = os.path.join(folderpath, f'frame_{i}.png')
            fig.savefig(frame_path)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)

    # Save the video
    imageio.mimsave(os.path.join(folderpath, '0_training_epochs.mp4'), frames, fps=4)