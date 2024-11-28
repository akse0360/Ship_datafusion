import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)



class Layer(torch.nn.Module):
    def __init__(self, hidden_dim, drop_prob=0.2):
        super(Layer, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(drop_prob)
        
    def forward(self, y):
        x = self.linear(y)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + y  # Residual connection

class DoubleFourierSphereLayer(nn.Module):
    def __init__(self, num_frequencies=10):
        super(DoubleFourierSphereLayer, self).__init__()
        self.num_frequencies = num_frequencies
        # Initialize random frequencies for latitude and longitude encoding
        self.freq_lat = nn.Parameter(torch.randn(num_frequencies) * 2 * np.pi, requires_grad=False)
        self.freq_lon = nn.Parameter(torch.randn(num_frequencies) * 2 * np.pi, requires_grad=False)
    
    def forward(self, lat, lon):
        lat_rad = lat * np.pi / 180.0
        lon_rad = lon * np.pi / 180.0
        encoding_lat = torch.cat([torch.sin(f * lat_rad).unsqueeze(-1) for f in self.freq_lat], dim=-1)
        encoding_lon = torch.cat([torch.sin(f * lon_rad).unsqueeze(-1) for f in self.freq_lon], dim=-1)
        
        # Combine latitude and longitude encodings
        embedding = torch.cat([encoding_lat, encoding_lon], dim=-1)
        return embedding

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, depth=1, drop_prob=0.2, num_frequencies=10):
        super(MyModel, self).__init__()
        self.output_dim = output_dim
        self.dfs_layer = DoubleFourierSphereLayer(num_frequencies=num_frequencies)
        
        # DFS layer provides 2 * num_frequencies for each of latitude and longitude
        self.base_dim = 2 * num_frequencies
        self.extra_features_dim = input_dim - 2  # Additional features if present
        
        # Total input dimension includes DFS encoding + any extra features
        total_input_dim = self.base_dim + self.extra_features_dim
        self.layers = nn.ModuleList()
        
        # Input layer with `total_input_dim`
        self.layers.append(nn.Linear(total_input_dim, hidden_dim))
        
        # Hidden layers with residual connections
        for _ in range(depth):
            self.layers.append(Layer(hidden_dim, drop_prob=drop_prob))
        
        # Output layer with fixed output_dim (default 2)
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def distance_matrix(self, emb1, emb2):
         # Compute distance matrix using torch.cdist for actual distances
         return torch.matmul(emb1, emb2.t())
    
    def forward(self, x):
        lat, lon = x[:, 0], x[:, 1]
        dfs_encoded = self.dfs_layer(lat, lon)
        
        # Concatenate DFS encoding with any extra features if available
        if self.extra_features_dim > 0:
            extra_features = x[:, 2:]  # Assuming lat/lon are first two columns
            x = torch.cat([dfs_encoded, extra_features], dim=-1)
        else:
            x = dfs_encoded
        
        # Pass through each layer
        for layer in self.layers:
            x = layer(x)
        return x

