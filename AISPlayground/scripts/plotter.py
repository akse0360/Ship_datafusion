import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
# Plotter
class Plotter:
    ## VISUALIZATION
    # Plot points
    def plot_tracks(ax, track1, track2, title="Track Matching Visualization"): 
        ax.scatter(track1[0], track1[1], c='b', marker='o', label="Track 1")
        ax.scatter(track2[0], track2[1], c='r', marker='x', label="Track 2")
        ax.set_title(title)
        ax.set_xlabel("Normalized x")
        ax.set_ylabel("Normalized y")
        ax.legend()
        return ax

    # # Plot connections with dynamic line width
    # def plot_connection(ax, track1, track2, assignment, **kwargs):
    #     for i, j in enumerate(assignment):
    #         t1 = track1[i, :2]
    #         t2 = track2[j, :2]
    #         ax.plot([t1[0], t2[0]], [t1[1], t2[1]], **kwargs)


    def plot_connection(ax, track1, track2, assignment, **kwargs):
        """
        Plots connections between points in track1 and track2 based on assignment.

        Parameters:
            ax: The matplotlib axis to plot on.
            track1: DataFrame or numpy array containing points from track 1.
            track2: DataFrame or numpy array containing points from track 2.
            assignment: List or array of indices indicating connections between track1 and track2.
        """
        for i, j in enumerate(assignment):
            # Use .iloc if track1 and track2 are DataFrames
            if isinstance(track1, pd.DataFrame):
                t1 = track1.iloc[i][['latitude_norm', 'longitude_norm']].values
            else:
                t1 = track1[i, :2]
            
            if isinstance(track2, pd.DataFrame):
                t2 = track2.iloc[j][['latitude_norm', 'longitude_norm']].values
            else:
                t2 = track2[j, :2]

            ax.plot([t1[0], t2[0]], [t1[1], t2[1]], **kwargs)

    # Plot heading vector
    def plot_direction_vectors(ax, p1, p2, magnitude=0.05, color1='blue', color2='red', width=0.003):
        """
        Plots direction vectors for two sets of points (p1, p2) on a given axis.
        
        Parameters:
            ax (matplotlib.axes.Axes): The axis to plot on.
            p1 (np.ndarray): Array of shape (N, 3) where each row is (x, y, heading in radians).
            p2 (np.ndarray): Array of shape (N, 3) where each row is (x, y, heading in radians).
            magnitude (float): The length of the direction vectors.
            color1 (str): Color for p1 vectors.
            color2 (str): Color for p2 vectors.
            width (float): Width of the quiver arrows.
        """
        
        
        # Extract headings in radians from the third column
        heading1 = p1[:, 2]
        heading2 = p2[:, 2]

        # Calculate dx and dy based on headings and magnitude
        dx1 = magnitude * np.cos(heading1)
        dy1 = magnitude * np.sin(heading1)
        
        dx2 = magnitude * np.cos(heading2)
        dy2 = magnitude * np.sin(heading2)

        # Plot quiver arrows for p1 and p2
        ax.quiver(p1[:, 0], p1[:, 1], dx1, dy1, angles='xy', scale_units='xy', scale=1, color=color1, width=width, alpha = 0.3)
        ax.quiver(p2[:, 0], p2[:, 1], dx2, dy2, angles='xy', scale_units='xy', scale=1, color=color2, width=width, alpha = 0.3)


    ## EVALUATION PARAMETERS
    # Plot loss curve
    def plot_loss(losses):
        fig, ax = plt.subplots()
        ax.semilogy(losses)
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
        fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.3)

        # Top plot: Track connections
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(p1['latitude_norm'].values, p1['longitude_norm'].values, c='b', marker='o', label="Track 1")
        ax1.scatter(p2['latitude_norm'].values, p2['longitude_norm'].values, c='r', marker='x', label="Track 2")

        green_indices = np.arange(min(len(p1), len(p2)))
        Plotter.plot_connection(ax1, p1[['latitude_norm', 'longitude_norm']], p2[['latitude_norm', 'longitude_norm']], green_indices, color='g')

        d = torch.argmin(diag_matrix, dim=1).detach().cpu().numpy()
        Plotter.plot_connection(ax1, p1[['latitude_norm', 'longitude_norm']], p2[['latitude_norm', 'longitude_norm']], d, color='r', linestyle='--')

        ax1.set_title(f"Matching visualization, Iteration: {iteration}, Loss: {loss:.4f}")
        ax1.legend(loc='upper left', bbox_to_anchor=(1.009, 1), fancybox=True, shadow=True)

        # Bottom left plot: Loss curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(losses['train'], label="Training Loss", color='b')
        ax2.semilogy(losses['val'], label="Validation Loss", color='orange', linestyle='--')
        ax2.set_xlim(0, max_iter + 1)
        ax2.set_title("Training Loss Over Iterations")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Loss")
        ax2.legend(loc='upper right', fancybox=True, shadow=True)

        # Bottom right plot: Rank accuracy curves
        ax3 = fig.add_subplot(gs[1, 1])
        for rank, accuracies in rank_accuracies.items():
            ax3.plot(range(len(accuracies)), accuracies, label=f'Rank {rank}', marker='o', linestyle='dashed')
        ax3.set_xlim(0, (max_iter + 1))
        ax3.set_ylim(0, 1)
        ax3.set_title("Rank-N Accuracy Over Iterations")
        ax3.set_xlabel("Iterations, [i]")
        ax3.set_ylabel("Accuracy")
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fancybox=True, shadow=True)

        return fig, (ax1, ax2, ax3)



    # Plot the frames
    def plot_iteration_with_loss_and_accuracy_playground(p1, p2, diag_matrix, losses, rank_accuracies, loss, iteration, max_iter):
        """
        Creates a subplot with the track plot on top, the loss curve in the bottom left, and the rank-N accuracy curve in the bottom right.
        """
        # Create a figure and a GridSpec layout for custom positioning
        fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.3)
        
        # Top plot: Track connections (spanning both columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(p1[:, 0], p1[:, 1], c='b', marker='o', label="Track 1")
        ax1.scatter(p2[:, 0], p2[:, 1], c='r', marker='x', label="Track 2")

        # Ensure valid indices for assignment
        green_indices = np.arange(min(len(p1), len(p2)))  # Example of a valid assignment
        Plotter.plot_connection(ax1, p1, p2, green_indices, color='g')

        # MLP connections
        d = torch.argmin(diag_matrix, dim=1).detach().cpu().numpy()
        Plotter.plot_connection(ax1, p1, p2, d, color='r', linestyle='--')

        # Remove direction vector plotting
        ax1.set_title(f"Matching visualization, Iteration: {iteration}, Loss: {loss:.4f}") #(Keep title if needed)

        ax1.legend(loc='upper left', bbox_to_anchor=(1.009, 1), fancybox=True, shadow=True)

        # Bottom left plot: Loss curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(losses['train'], label="Training Loss", color='b')
        ax2.semilogy(losses['val'], label="Validation Loss", color='orange', linestyle='--')
        ax2.set_xlim(0, max_iter + 1)
        ax2.set_title("Training Loss Over Iterations")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Loss")
        ax2.legend(loc='upper right', fancybox=True, shadow=True)

        # Bottom right plot: Rank accuracy curves
        ax3 = fig.add_subplot(gs[1, 1])
        for rank, accuracies in rank_accuracies.items():
            ax3.plot(range(len(accuracies)), accuracies, label=f'Rank {rank}', marker='o', linestyle='dashed')
        ax3.set_xlim(0, (max_iter + 1))
        ax3.set_ylim(0, 1)
        ax3.set_title("Rank-N Accuracy Over Iterations")
        ax3.set_xlabel("Iterations, [i]")
        ax3.set_ylabel("Accuracy")
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fancybox=True, shadow=True)
        
        return fig, (ax1, ax2, ax3)