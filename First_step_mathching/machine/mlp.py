import torch
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)

class Layer(torch.nn.Module):
    def __init__(self, hidden_dim, drop_prob,norm=None):
        super(Layer, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm = norm(hidden_dim) if norm else torch.nn.Identity()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(drop_prob)
        
    def forward(self, y):
        x = self.linear(y)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + y  # Residual connection

class MyModel(torch.nn.Module):
    """
    Model for the task of learning embeddings. The model consists of an input layer, a number of hidden layers with
    residual connections, and an output layer. The output layer produces embeddings of a fixed dimensionality.

    The output of the model is a tensor of shape (batch_size, output_dim) containing the embeddings of the input data.
    This model only makes one embedding per pass.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1, drop_prob=0.2,norm=None):
        super(MyModel, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        # Hidden layers with residual connections
        for _ in range(depth):
            self.layers.append(Layer(hidden_dim, drop_prob,norm=norm))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Passing through layers
        for layer in self.layers:
            x = layer(x)
        return x
    
    # def distance_matrix(self, emb1, emb2):
    #     # Compute distance matrix using inner products
    #     return torch.matmul(emb1, emb2.t()) # torch.cdist(emb1, emb2) 

