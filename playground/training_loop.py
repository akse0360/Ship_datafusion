import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from tqdm.notebook import tqdm

from scripts.startUp import StartUp
from scripts.trainingManager import TrainingManager
from model.mlp import MyModel
from scripts.generation import Generator
from scripts.metrics import Metrics
from model.loss_functions import LossFunctions
from scripts.plotter import Plotter

def train_model_loss_static_val(
    model_type,  # 'pos' or 'head'
    input_dim,  # Input feature dimension
    p1, p2,
    hidden_dim=1024,
    output_dim=254,
    n_layers=5,
    dropout=0.2,
    learning_rate=0.00001,
    max_iter=750,
    N=2000,
    sigma_distance=0.05,
    sigma_heading=0.015,
    ranks=[1, 3, 5],
    loss_function=LossFunctions.softmax_matching_loss,  # Default loss function
    save_path='.\\',
):
    """
    Trains an MLP model for positional or heading embedding tasks.

    Args:
        model_type (str): Type of training ('pos' for position, 'head' for heading).
        input_dim (int): Input dimension of the model.
        hidden_dim (int): Hidden layer size.
        output_dim (int): Output embedding dimension.
        n_layers (int): Number of hidden layers.
        dropout (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.
        max_iter (int): Number of training iterations.
        N (int): Number of training samples per iteration.
        sigma_distance (float): Noise level for positional data.
        sigma_heading (float): Noise level for heading data.
        ranks (list): List of Rank-N values for evaluation.
        save_path (str): Path to save training results.

    Returns:
        model: Trained PyTorch model.
        training_manager: TrainingManager with recorded metrics.
    """
    

    def forward(t1, t2, model):
        t1 = torch.tensor(t1, dtype=torch.float32).to(device)
        t2 = torch.tensor(t2, dtype=torch.float32).to(device)
        e1 = model(t1)
        e2 = model(t2)
        return torch.cdist(e1, e2)

    
    loss_function_name = loss_function.__name__ if hasattr(loss_function, '__name__') else str(loss_function)
    print(loss_function_name)
    # Set up paths
    os.makedirs(save_path, exist_ok=True)
    paths = StartUp.clear_folder(currpath=save_path, models=[model_type], clear_movies=False)

    # Initialize TrainingManager
    manager = TrainingManager(model_type, paths, ranks)

    # Define the model and optimizer
    model = MyModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        depth=n_layers,
        drop_prob=dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Generate validation data
    t1, t2 = p1, p2

    # Training loop
    for i in tqdm(range(max_iter + 1), desc=f'Training {model_type}-model'):
        model.train()
        optimizer.zero_grad()

        # Generate training data
        p1t, p2t = Generator.generate_data(N, sigma=sigma_distance)
        if 'head' in model_type:
            t1p, t2p = Generator.calculate_heading(p1t, p2t, sigma=sigma_heading)
        else:
            t1p, t2p = p1t, p2t

        # Compute distances and loss
        diag_matrix = forward(t1p, t2p, model)
        train_loss = loss_function(diag_matrix)
        manager.losses['train'].append(train_loss.item())

        # Backpropagation
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            diag_matrix_val = forward(t1, t2, model)
            val_loss = loss_function(diag_matrix_val)
            manager.losses['val'].append(val_loss.item())

            # Rank evaluation
            true_indices = np.arange(t1.shape[0])
            y_pred = np.argmin(diag_matrix_val.cpu().numpy(), axis=1)

            for rank in ranks:
                accuracy = Generator.evaluate_rankN(diag_matrix_val, rank)
                manager.rank_accuracies[rank].append(accuracy)

            # Metrics
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
            manager.save_metrics_to_csv(i, train_loss.item(), val_loss.item(), metrics_data, manager.rank_accuracies[1][-1], tpfptnfn)

        # Save best model
        rank_1_accuracy = manager.rank_accuracies[1][-1]
        if val_loss.item() < manager.best_val_loss and rank_1_accuracy > manager.best_rank_accuracy:
            manager.best_val_loss = val_loss.item()
            manager.best_rank_accuracy = rank_1_accuracy
            torch.save(model.state_dict(), manager.best_model_path)

        # Plot progress every 10 iterations
        if i % 10 == 0 or i == 1:
            fig, _ = Plotter.plot_iteration_with_loss_and_accuracy(
                t1, t2, diag_matrix_val, manager.losses, manager.rank_accuracies, train_loss.item(), i, max_iter
            )
            frame_path = os.path.join(manager.paths['images'], f'frame_{i}.png')
            fig.savefig(frame_path)
            manager.frames.append(imageio.imread(frame_path))
            plt.close(fig)

    # Save the training video
    video_path = os.path.join(manager.paths['movies'], f'{StartUp.get_time()}_training_{model_type}.mp4')
    imageio.mimsave(video_path, manager.frames, fps=3)

    print(f"Training video for {model_type}, saved at {video_path}")
    return model, manager
