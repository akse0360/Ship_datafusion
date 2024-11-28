import os
import csv
import shutil
from scripts.startUp import StartUp

class TrainingManager:
    def __init__(self, context, paths, ranks):
        """
        Initialize Training Manager for a specific context.

        Args:
        - context: The training context (e.g., 'pos', 'head', 'sog', 'all').
        - paths: Dictionary containing paths for the current context.
        - ranks: List of ranks for evaluation.
        """
        self.context = context
        self.paths = paths[context]
        self.ranks = ranks

        # Initialize variables
        self.frames = []
        self.losses = {'train': [], 'val': []}
        self.rank_accuracies = {rank: [] for rank in ranks}
        self.rank_accuracies10 = {rank: [] for rank in ranks}
        self.best_val_loss = float('inf')
        self.best_rank_accuracy = 0.0
        
        self.best_model_path = os.path.join(self.paths['models'], f'{StartUp.get_time()}_best_model.pth')

        # Initialize CSV
        self.csv_file_path = os.path.join(self.paths['csv'], f'{StartUp.get_time()}_training_metrics.csv')
        self._initialize_csv()

        # Clear images folder
        self._clear_folder(self.paths['images'])

    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'iteration', 'train_loss', 'val_loss', 'accuracy', 'recall',
                'precision', 'mean_average_precision', 'f1', 'accuracy_rank1', 'confusion'
            ])

    def _clear_folder(self, folder_path):
        """Clear all files from a folder."""
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def save_metrics_to_csv(self, iteration, train_loss, val_loss, metrics, rank1_accuracy, tpfptnfn):
        """Save metrics to the CSV file."""
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                iteration, train_loss, val_loss, metrics['Accuracy'], metrics['Recall'],
                metrics['Precision'], metrics['MAP'], metrics['F1'], rank1_accuracy, tpfptnfn
            ])
