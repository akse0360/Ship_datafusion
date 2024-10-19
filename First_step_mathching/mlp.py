import torch
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    
    idx1, idx2 = linear_sum_assignment(d_np)
    
    return torch.tensor(idx1), torch.tensor(idx2)


class MyModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1):
        super(MyModel, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        for _ in range(depth):
            self.layers.append(Layer(hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # X er B, D
        # B er batch size eller N antal punkter vi vil matche med en anden vector x'
        # D er antal features i input e.g. cog, sog, location
        for layer in self.layers:
            x = layer(x)
        return x
    
    def distance_matrix(self, x, y):
        # Vi vil matche x med y ved at lave en afstands matrix
        emb1 = self.forward(x)
        emb2 = self.forward(y)


        # Alternativ metode med torch.einsum
        dist_matrix = torch.einsum('ik,jk->ij', emb1, emb2)

        return dist_matrix
        dist_matrix = torch.matmul(emb1, emb2.t())
        return dist_matrix
        # TODO make it go faster by implementing below
        emb1, emb2 = self.mlp(batch.view(-1, self.output_dim)) \
                         .view(batch.size(0), batch.size(1), self.output_dim) \
                         .chunk(2, dim=1) 
        print(emb1.size(), emb2.size())
        dist_matrix = torch.matmul(emb1.squeeze(1), emb2.squeeze(1).t())
        
        return dist_matrix

class Layer(torch.nn.Module):
    def __init__(self, hidden_dim,drop_prob=0.2):
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
        return x + y 
