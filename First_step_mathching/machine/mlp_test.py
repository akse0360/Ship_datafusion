import torch
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)

class Layer(torch.nn.Module):
    def __init__(self, hidden_dim, drop_prob):
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

class MyModel(torch.nn.Module):
    """
    Model for the task of learning embeddings. The model consists of an input layer, a number of hidden layers with
    residual connections, and an output layer. The output layer produces embeddings of a fixed dimensionality.

    The output of the model is a tensor of shape (batch_size, output_dim) containing the embeddings of the input data.
    This model only makes one embedding pr. pass 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1, drop_prob=0.2):
        super(MyModel, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        # Hidden layers with residual connections
        for _ in range(depth):
            self.layers.append(Layer(hidden_dim, drop_prob))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Passing through layers
        for layer in self.layers:
            x = layer(x)
        return x
    
    def distance_matrix(self, x, y):
        # Get embeddings
        emb1 = self.forward(x)
        emb2 = self.forward(y)

        # Compute distance matrix using torch.cdist for actual distances
        return torch.matmul(emb1, emb2)

##############################
import torch
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)


class Layer(torch.nn.Module):
    def __init__(self, hidden_dim, drop_prob):
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
class MyModel(torch.nn.Module):
    """
    Model for the task of learning embeddings. The model consists of an input layer, a number of hidden layers with
    residual connections, and an output layer. The output layer produces embeddings of a fixed dimensionality.

    The output of the model is a tensor of shape (batch_size, output_dim) containing the embeddings of the input data.
    This model only makes one embedding pr. pass 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1, drop_prob=0.2):
        super(MyModel, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        # Hidden layers with residual connections
        for _ in range(depth):
            self.layers.append(Layer(hidden_dim, drop_prob))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Passing through layers
        for layer in self.layers:
            x = layer(x)
        return x
    
    def distance_matrix(self, x, y):
        # Get embeddings
        emb1 = self.forward(x)
        emb2 = self.forward(y)

        # Compute distance matrix using torch.cdist for actual distances
        return torch.cdist(emb1, emb2)



             #     # Forward pass
            #     combined_batch = torch.cat([ais1, ais2], dim=0)
            #     emb1, emb2 = model(combined_batch).chunk(2, dim=0)
            #     dist_matrix = model.distance_matrix(emb1, emb2)
                
            #     # Compute loss
            #     loss = loss_fn(dist_matrix, temperature=TEMPERATURE)  
            # scaler.scale(loss.mean()).backward()  

            # scaler.step(optimizer)
            # scaler.update()

            # with torch.no_grad():
            #     # Use autocast for validation too
            #     with torch.amp.autocast('cuda', dtype=torch.float32):
            #         # Forward pass and distance matrix calculation
            #         combined_batch = torch.cat([ais1, ais2], dim=0)
            #         emb1, emb2 = model(combined_batch).chunk(2, dim=0)
            #         dist_matrix = model.distance_matrix(emb1, emb2)
                    
            #         # Compute validation loss
            #         loss = loss_fn(dist_matrix)  # Calculate validation loss (returns tensor)

            # val_loss += loss.sum().item()  # Or loss.sum().item()

            # val_loader_tqdm.set_postfix({'val_loss': val_loss / ((i + 1) * batch_size)})

            # if i * batch_size >= val_iterations:
            #     break

class MyModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1, drop_prob=0.2):
        super(MyModel, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        # Hidden layers with residual connections
        for _ in range(depth):
            self.layers.append(Layer(hidden_dim, drop_prob = drop_prob))

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
    
    def distance_matrix(self, emb1, emb2):
        # Compute distance matrix using torch.cdist for actual distances
        return torch.matmul(emb1, emb2.t())
    
    def forward(self, batch):
        emb1, emb2 = self.layers(batch.view(-1, self.output_dim)).view(batch.size(0), batch.size(1), self.output_dim).chunk(2, dim=1)
        return emb1, emb2

    # def forward(self, batch):

    #     # Passing through layers
    #     for layer in self.layers:
    #         batch = layer(batch)

    #     # Reshape and split into emb1 and emb2 for matrix computation
    #     emb1, emb2 = batch.view(-1, self.output_dim) \
    #                      .view(batch.size(0), batch.size(1), self.output_dim) \
    #                      .chunk(2, dim=1)
    #     print(emb1.size(), emb2.size())

    #     # Calculate distance matrix via matrix multiplication     
    #     return torch.matmul(emb1.squeeze(1), emb2.squeeze(1).t())


######################################################
import torch
from scipy.optimize import linear_sum_assignment


def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)


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

class MyModel_(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=1):
        super(MyModel_, self).__init__()
        self.output_dim = output_dim  # Dimension of the embeddings
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        for _ in range(depth):
            self.layers.append(Layer(hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # X er B, D
        # B is batch size or  N number of points we wish to match to vector x'
        # D  is the number fo features in the input e.g. cog, sog, location
        for layer in self.layers:
            x = layer(x)
        return x
    
    def distance_matrix(self, x, y):
        # We wish to find the distance between x and y
        emb1 = self.forward(x)
        emb2 = self.forward(y)

        # Compute distance matrix with different methods
        #dist_matrix = torch.cdist(emb1, emb2)
        dist_matrix = torch.matmul(emb1, emb2.t())
        #dist_matrix = torch.einsum('ik,jk->ij', emb1, emb2)

        return dist_matrix
        
        # TODO make it go faster by implementing below
        emb1, emb2 = self.forward(batch.view(-1, self.output_dim)) \
                         .view(batch.size(0), batch.size(1), self.output_dim) \
                         .chunk(2, dim=1) 
        print(emb1.size(), emb2.size())
        dist_matrix = torch.matmul(emb1.squeeze(1), emb2.squeeze(1).t())
        
        return dist_matrix


    
# class MyModel3(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, depth=1, drop_prob=0.2):
#         super(MyModel, self).__init__()
#         self.output_dim = output_dim  # Dimension of the embeddings
#         self.layers = torch.nn.ModuleList()

#         # Input layer
#         self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

#         # Hidden layers with residual connections
#         for _ in range(depth):
#             self.layers.append(Layer(hidden_dim, drop_prob = drop_prob))

#         # Output layer
#         self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
    
#     def distance_matrix(self, emb1, emb2):
#         # Compute distance matrix using torch.cdist for actual distances
#         return torch.matmul(emb1, emb2.t())
    
#     def forward(self, batch):
#         # Pass the batch through each layer
#         for layer in self.layers:
#             batch = layer(batch)

#         # Reshape to get embeddings for emb1 and emb2
#         # Assuming batch originally had shape [batch_size, 2, input_dim]
#         emb1, emb2 = batch.view(batch.size(0) // 2, 2, self.output_dim).chunk(2, dim=1)
#         emb1, emb2 = emb1.squeeze(1), emb2.squeeze(1)  # Remove the extra dimension

#         return emb1, emb2

#     def forward2(self, batch):
#         # Passing through layers
#         for layer in self.layers:
#             batch = layer(batch)

#         # Reshape and split into emb1 and emb2 for matrix computation
#         emb1, emb2 = batch.view(-1, self.output_dim) \
#                          .view(batch.size(0), batch.size(1), self.output_dim) \
#                          .chunk(2, dim=1)
#         print(emb1.size(), emb2.size())

#         # Calculate distance matrix via matrix multiplication     
#         return torch.matmul(emb1.squeeze(1), emb2.squeeze(1).t())