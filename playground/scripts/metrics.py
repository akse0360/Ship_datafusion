import numpy as np
from sklearn.metrics import confusion_matrix, top_k_accuracy_score, accuracy_score, recall_score, precision_score, f1_score, average_precision_score
# TODO from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


class Metrics:
    def __init__(self, y_true, y_pred):
        """
        Initialize Metrics with true and predicted indices.
        
        Args:
        - y_true: Ground truth indices.
        - y_pred: Predicted indices.
        """
        self.y_true = np.array(y_true, dtype=object)
        self.y_pred = np.array(y_pred)
        if self.y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError("Mismatch between y_true and y_pred lengths.")

        # Filter out None values for valid ground truth indices
        self.valid_mask = self.y_true != -1
        self.valid_true = self.y_true[self.valid_mask]
        self.valid_pred = self.y_pred[self.valid_mask]
        self.valid_true = np.array(self.valid_true, dtype=int)  # Convert to integers
        self.valid_pred = np.array(self.valid_pred, dtype=int)
        self.valid_true = self.valid_true[self.valid_true != -1]
        self.valid_pred = self.valid_pred[self.valid_true != -1]

        # Map predictions to valid classes
        valid_classes = set(self.valid_true)
        self.valid_pred = np.array([p if p in valid_classes else -1 for p in self.valid_pred])

        if len(self.valid_true) != len(self.valid_pred):
            raise ValueError("Filtered y_true and y_pred lengths are inconsistent.")

    @staticmethod
    def _to_numpy(data, allow_none=False):
        """
        Convert input data to NumPy array, with optional support for None values.
        
        Args:
        - data: Input data (list, tensor, etc.).
        - allow_none: Whether to allow None values in the resulting array.

        Returns:
        - NumPy array with appropriate type.
        """
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, "detach"):  # Handle PyTorch tensors
            return data.detach().cpu().numpy()
        elif allow_none:
            return np.array(data, dtype=object)  # Allows None
        else:
            return np.array(data)

    def get_accuracy(self):
        """
        Calculate accuracy using sklearn.
        Accuracy is the proportion of true matches that were predicted correctly.
        """
        self.valid_true = np.array(self.valid_true, dtype=int)
        self.valid_pred = np.array(self.valid_pred, dtype=int)
        if len(self.valid_true) == 0:
            print("Warning: No valid entries in y_true; accuracy is undefined.")
            return 0.0  # Return 0.0 if no valid entries
        return accuracy_score(self.valid_true, self.valid_pred)

    def get_recall(self):
        """
        Calculate recall using sklearn.
        Recall is the proportion of true matches that were predicted correctly.
        """
        return recall_score(self.valid_true, self.valid_pred, average="micro")

    def get_precision(self):
        """
        Calculate precision using sklearn.
        Precision is the proportion of predicted matches that are correct.
        """
        return precision_score(self.valid_true, self.valid_pred, average="micro")

    def get_f1(self):
        """
        Calculate F1-score using sklearn.
        F1-score is the harmonic mean of precision and recall.
        """
        return f1_score(self.valid_true, self.valid_pred, average="micro")

    def get_mean_average_precision(self):
        """
        Compute Mean Average Precision (MAP).
        """
        max_label = int(max(self.y_true[~np.equal(self.y_true, None)]))  # Exclude None
        y_true_binary = (self.y_true[:, None] == np.arange(max_label + 1)).astype(int)
        y_pred_binary = (self.y_pred[:, None] == np.arange(max_label + 1)).astype(int)
        return average_precision_score(y_true_binary, y_pred_binary, average="micro")

    def top_k_recall(self, k):
        """
        Check if true matches are within the top k predictions.
        
        Args:
        - k: Number of top predictions to consider.
        
        Returns:
        - Recall within top k.
        """
        correct_matches = 0
        for true_idx, pred_idx in zip(self.valid_true, self.valid_pred):
            if true_idx in pred_idx[:k]:
                correct_matches += 1
        return correct_matches / len(self.valid_true) if len(self.valid_true) > 0 else 0.0
    
    def get_confusion_matrix(self):
        """Get the confusion matrix."""
        return confusion_matrix(self.valid_true, self.valid_pred)

    def get_tp_fp_tn_fn(self):
        """
        Calculate True Positives (TP), False Positives (FP),
        True Negatives (TN), and False Negatives (FN).

        Returns:
        - [TP, FP, TN, FN]: A list containing the counts.
        """
        # Compute confusion matrix
        confusion_matrix = self.get_confusion_matrix()

        tp = np.diag(confusion_matrix).sum()  # Sum of diagonal entries
        fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  # Column sum minus diagonal
        fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)  # Row sum minus diagonal
        tn = confusion_matrix.sum() - (tp + fp.sum() + fn.sum())  # Total - (TP + FP + FN)

        return [int(tp), int(fp.sum()), int(tn), int(fn.sum())]
    
    @staticmethod
    def get_top_k_accuracy_score(true_indices, neg_distances, k):
        """
        Calculate the top-k accuracy score.

        Args:
        - true_indices: Array of true indices.
        - neg_distances: Array of negative distances.
        - k: The value of k for top-k accuracy.

        Returns:
        - Top-k accuracy score.
        """
        return top_k_accuracy_score(true_indices, neg_distances, k=k)

#     @staticmethod
#     def cross_val_score(model, X, y, cv=5, scoring='accuracy'):
#         """
#         Perform cross-validation and return the scores.

#         Args:
#         - model: The model to evaluate.
#         - X: Features.
#         - y: Labels.
#         - cv: Number of cross-validation folds.
#         - scoring: Scoring metric.

#         Returns:
#         - scores: Cross-validation scores.
#         """
#         return cross_val_score(model, X, y, cv=cv, scoring=scoring)