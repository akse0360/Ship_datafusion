import numpy as np
import torch
import torch.nn.functional as F


class LossFunctions:
    @staticmethod
    def diagonal_sum_loss(diag_matrix):
        """
        Calculate the loss as the sum of the diagonal elements of the distance matrix.

        Args:
        - diag_matrix: Diagonal matrix of distances.

        Returns:
        - loss: Sum of the diagonal elements.
        """
        return torch.sum(torch.diag(diag_matrix))/diag_matrix.shape[1]

    @staticmethod
    def diagonal_softmax_loss(diag_matrix):
        """
        Calculate the loss as the softmax of the diagonal elements of the distance matrix.

        Args:
        - diag_matrix: Diagonal matrix of distances.

        Returns:
        - loss: Softmax of the diagonal elements.
        """
        return torch.sum(torch.diag(F.softmax(diag_matrix, dim=1)))
    
    @staticmethod
    def cross_entropy_matching_loss(diag_matrix):
        """
        Calculate the cross entropy loss.

        Args:
        - output: Output of the model.
        - target: Ground truth labels.

        Returns:
        - loss: Cross entropy loss.
        """
        target = torch.arange(diag_matrix.size(0), device=diag_matrix.device)
        return F.cross_entropy(F.softmax(diag_matrix, dim=1), target)
    
    @staticmethod
    def softmax_diagonal_loss(diag_matrix):
        softmax_matrix = torch.softmax(diag_matrix, dim=1)
        diag = torch.diag(softmax_matrix)

        row_max, _ = torch.max(softmax_matrix, dim=1)
        
        loss = torch.sum((diag - row_max) ** 2) / diag_matrix.size(0)
        return loss

    @staticmethod
    def softmax_matching_loss(distance_matrix):
        """
        Row-wise softmax with cross-entropy loss for matching.

        Args:
            distance_matrix (torch.Tensor): Distance matrix (N x N).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply softmax to negative distances (convert to similarity)
        similarity_matrix = -distance_matrix
        log_probs = torch.log_softmax(similarity_matrix, dim=1)

        # Ground truth indices (diagonal should be the target)
        target_indices = torch.arange(distance_matrix.size(0), device=distance_matrix.device)

        # Cross-entropy loss
        return torch.nn.functional.nll_loss(log_probs, target_indices)
    
    @staticmethod
    def contrastive_distance_loss(distance_matrix, margin=1.0):
        """
        Contrastive loss for matching P1 with P2.

        Args:
            distance_matrix (torch.Tensor): Distance matrix (N x N).
            margin (float): Minimum margin between correct and incorrect matches.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Extract diagonal (correct matches)
        diag_distances = torch.diagonal(distance_matrix, offset=0)

        # Expand diagonal to compare with entire matrix
        diag_matrix = diag_distances.unsqueeze(1).expand_as(distance_matrix)

        # Margin loss: max(0, margin + d_correct - d_incorrect)
        loss_matrix = torch.clamp(margin + diag_matrix - distance_matrix, min=0.0)

        # Exclude diagonal from loss (no self-comparison)
        mask = torch.eye(distance_matrix.size(0), device=distance_matrix.device).bool()
        loss_matrix = loss_matrix.masked_fill(mask, 0.0)

        # Mean loss
        return loss_matrix.sum() / distance_matrix.size(0)

    @staticmethod
    def triplet_margin_loss(distance_matrix, margin=1.0):
        positive_distances = torch.diagonal(distance_matrix)
        min_negative_distances, _ = torch.min(distance_matrix, dim=1)
        loss = torch.clamp(positive_distances - min_negative_distances + margin, min=0.0)
        return loss.mean()