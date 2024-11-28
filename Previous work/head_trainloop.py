# -*- coding: utf-8 -*-
# ---------------- #
# Import libaries
import os
import numpy as np
import torch
# Plot
import matplotlib.pyplot as plt
# Movie generator:
import imageio.v2 as imageio
# ---------------- #
# Managers:
## folder manager
from scripts.startUp import StartUp
## Training manager
from scripts.trainingManager import TrainingManager
# ---------------- #
# MLP
### Model
from model.mlp import MyModel
### Data generation
from scripts.generation import Generator
### Metrics
from scripts.metrics import Metrics
### Loss functions, optimizers
from model.loss_functions import LossFunctions

# Visualisation
## Plots
from scripts.plotter import Plotter

# ---------------- #
def forward(t1, t2, model):
    e1 = model(torch.tensor(t1, dtype=torch.float32))
    e2 = model(torch.tensor(t2, dtype=torch.float32))
    return torch.cdist(e1,e2)

# ---------------- #
if __name__ == '__main__':
    # Set up paths
    models = ['head']

    # Set up the file paths
    curpath = r'C:\Users\abelt\OneDrive\Dokumenter\GitHub\Ship_datafusion\playground'
    os.makedirs(curpath, exist_ok=True)

    paths = StartUp.clear_folder(currpath= curpath, models=models, clear_movies=False)
    formatted_time = StartUp.get_time()
    
    mlp_params = {
        'hidden_dim': 1024,
        'output_dim': 254,
        'n_layers': 5,
        'dropout': 0.2,
        'learning_rate': 0.00001,
    }

    # Set up model and optimizer
    np.random.seed(1)

    maxiter = 750
    N = 2000
    N_test = N//10
    sigma_distance = 0.05
    sigma_heading = 0.015

    ranks = [1, 3, 5]
    # Generate validation data
    # Position data:
    p1, p2 = Generator.generate_data(N_test, sigma=sigma_distance)

    ground_truth = np.arange(p1.shape[0])
    # Heading data:
    t1, t2 = Generator.calculate_heading(p1,p2, sigma=sigma_heading)

    # Initialize TrainingManager for 'head' context
    manager = TrainingManager('head', paths, ranks)

    # Initialize model and optimizer
    model = MyModel(
        input_dim=3, 
        hidden_dim=mlp_params['hidden_dim'], 
        output_dim=mlp_params['output_dim'], 
        depth=mlp_params['n_layers'], 
        drop_prob=mlp_params['dropout']
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_params['learning_rate'])

    # Training loop
    for i in range(maxiter + 1):
        model.train()
        optimizer.zero_grad()

        # Generate training data
        p1t, p2t = Generator.generate_data(N, sigma=sigma_distance)
        t1p, t2p = Generator.calculate_heading(p1t, p2t)
        diag_matrix = forward(t1p, t2p, model)

        # Compute training loss
        train_loss = LossFunctions.get_loss_diagsum(diag_matrix)
        manager.losses['train'].append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # Forward pass
            diag_matrix_val = forward(t1, t2, model)

            # Compute validation loss
            val_loss = LossFunctions.get_loss_diagsum(diag_matrix_val)
            manager.losses['val'].append(val_loss.item())

        # Rank-N evaluation
        true_indices = np.arange(t1.shape[0])
        y_pred = np.argmin(diag_matrix_val, axis=1)

        for rank in ranks:
            neg_distances = -diag_matrix_val
            accuracy = Metrics.get_top_k_accuracy_score(true_indices, neg_distances, k=rank)
            manager.rank_accuracies[rank].append(accuracy)

        # Calculate metrics
        metrics = Metrics(true_indices, y_pred)
        metrics_data = {
            'Accuracy': metrics.get_accuracy(),
            'Recall': metrics.get_recall(),
            'Precision': metrics.get_precision(),
            'MAP': metrics.get_mean_average_precision(),
            'F1': metrics.get_f1(),
        }
        tpfptnfn = metrics.get_tp_fp_tn_fn()

        # Save metrics to CSV
        manager.save_metrics_to_csv(i, train_loss.item(), val_loss.item(), metrics_data, manager.rank_accuracies[3][-1], tpfptnfn)

        # Save best model
        rank_1_accuracy = manager.rank_accuracies[1][-1]
        if val_loss.item() < manager.best_val_loss and rank_1_accuracy > manager.best_rank_accuracy:
            manager.best_val_loss = val_loss.item()
            manager.best_rank_accuracy = rank_1_accuracy
            torch.save(model.state_dict(), manager.best_model_path)
            print(f"Iteration {i}: Best model saved with validation loss {manager.best_val_loss:.4f}, Rank-1 Accuracy: {manager.best_rank_accuracy:.4f}")

        # Plot and save frame every 10 iterations
        if i % 10 == 0 or i == 1:
            for rank in ranks:
                # accuracy10 = Generator.evaluate_rankN(diag_matrix_val, rank)
                # manager.rank_accuracies10[rank].append(accuracy10)
                accuracy10 = Metrics.get_top_k_accuracy_score(true_indices, -diag_matrix_val, k=rank)
                manager.rank_accuracies10[rank].append(accuracy10)
                
            fig, (ax1, ax2, ax3) = Plotter.plot_iteration_with_loss_and_accuracy(
                t1, t2, diag_matrix_val, manager.losses, manager.rank_accuracies10, train_loss.item(), i, maxiter
            )
            frame_path = os.path.join(manager.paths['images'], f'frame_{i}.png')
            fig.savefig(frame_path)
            manager.frames.append(imageio.imread(frame_path))
            plt.close(fig)

    # Save the training video
    video_path = os.path.join(manager.paths['movies'], f'{formatted_time}_training_head.mp4')
    imageio.mimsave(video_path, manager.frames, fps=3)
    print(f"Training video saved at {video_path}")
