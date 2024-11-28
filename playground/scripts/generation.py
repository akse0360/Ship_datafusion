import numpy as np
import torch

class Generator:
    
    # Normalize the data (optional)
    def normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

    # Encode heading as cos and sin
    def encode_heading_cos_sin(headings):
        cos_heading = np.cos(headings) #np.cos(np.radians(headings))
        sin_heading = np.sin(headings) #np.sin(np.radians(headings))
        
        return cos_heading, sin_heading

    # Position Generate synthetic data
    def generate_data(N, sigma): # Position only
        g = np.random.Generator(np.random.PCG64())
        x = g.uniform(0, 1, N)
        y = g.uniform(0, 1, N)
        x1 = np.clip(g.normal(x, sigma), a_min=0, a_max=1)  
        y1 = np.clip(g.normal(y, sigma), a_min=0, a_max=1)
        p1 = np.stack((x, y)).T
        p2 = np.stack((x1, y1)).T
        return p1, p2

    # Ensure calculate_heading outputs (N, 3) arrays
    def calculate_heading(p1, p2, sigma=0.05):
        g = np.random.Generator(np.random.PCG64())

        headings = np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])

        headings1 = headings + g.normal(0, (2 * np.pi) * sigma, len(headings))
        headings2 = headings + g.normal(0, (2 * np.pi) * sigma, len(headings))

        # Wrap from 0 to 2pi
        headings1 = np.mod(headings1, 2 * np.pi)
        headings2 = np.mod(headings2, 2 * np.pi)

        t1 = np.column_stack((p1, headings1))
        t2 = np.column_stack((p2, headings2))

        return t1, t2


    def mask_data_points(p1, p2, missing_percentage_data):
        """
        Masks data points in p2 to simulate less data points compared to p1.
        
        Args:
        - p1: numpy array, data points in p1.
        - p2: numpy array, data points in p2.
        - missing_percentage_data: float, percentage of data points to mask in p2.
        
        Returns:
        - p2_masked: numpy array, masked data points in p2.
        - ground_truth_masked: numpy array, ground truth indices reflecting the masked p2.
        """
        # Ground truth
        ground_truth = np.arange(p1.shape[0])
        
        # Mask to simulate SAR and AIS data ratio
        mask = np.random.rand(p2.shape[0]) > missing_percentage_data
        p2_masked = p2[mask, :]
        
        # Map ground truth to masked p2
        remaining_indices = np.where(mask)[0]  # Indices retained in p2_masked

        ground_truth_masked = np.array([
            np.where(remaining_indices == idx)[0][0] if idx in remaining_indices else -1
            for idx in ground_truth
        ])
        # Keep ground truth for valid matches (-1 indicates no match in p2_masked)
        valid_ground_truth_masked = np.where(ground_truth_masked != -1, ground_truth_masked, -1)

        return p2_masked, valid_ground_truth_masked

    def evaluate_rankN(dist_matrix, n=5):
        """
        Evaluate Rank-N accuracy for a given distance matrix.

        Args:
        - dist_matrix (torch.Tensor): Distance matrix of shape (N, N).
        - n (int): Rank to evaluate.

        Returns:
        - rank_accuracy (float): The Rank-N accuracy as a float.
        """
        # Labels as an identity matrix (ground truth: i-th row matches i-th column)
        labels = torch.arange(dist_matrix.size(0), device= dist_matrix.device)
        
        # Get indices of the sorted distances (ascending order, closest first)
        sorted_indices = torch.argsort(dist_matrix, dim=1, descending=False)
        
        # Check if the ground truth index is within the top-N predictions
        correct = (sorted_indices[:, :n] == labels.unsqueeze(1)).any(dim=1)
        
        # Compute the mean accuracy
        rank_accuracy = correct.float().mean().item()
        return rank_accuracy


    # # Determine rank
    # def evaluate_rankN(dist_matrix, n=5):
    #     """
    #     Args:
    #     - dist_matrix (torch.Tensor): Distance matrix of shape (N, N).
    #     - n (int): Rank to evaluate.

    #     """
    #     # Ranking evaluation function
    #     def rankN(scores, labels, n=5):
    #         sort = torch.argsort(scores, dim=1, descending=True)
    #         sorted_labels = labels.gather(1, sort)
    #         sorted_labels = sorted_labels[torch.any(sorted_labels, dim=1)]
    #         correct = torch.any(sorted_labels[:, 0:n], dim=1)
    #         return correct.float().mean()
        
    #     distance_matrix = dist_matrix 
    #     labels = torch.eye(distance_matrix.size(0), dtype=torch.float32)
    #     scores = -distance_matrix  

    #     rank_accuracy = rankN(scores, labels, n)
    #     return rank_accuracy