import torch
from torch import nn
from torch.optim import Adam
from mlp import MLP
from First_step_mathching.machine.ai_loader import ais_dataset
from scripts.data_loader import DataLoader

# Example of Hungarian function (to be implemented or imported)
from scipy.optimize import linear_sum_assignment

def hungarian(d):
    d_np = d.cpu().detach().numpy()
    idx1, idx2 = linear_sum_assignment(d_np)
    return torch.tensor(idx1), torch.tensor(idx2)

def infer(points1, points2, model):
    # Forward pass to compute embeddings
    emb1 = model(points1)
    emb2 = model(points2)
    
    # Compute distance matrix
    d = torch.matmul(emb1, emb2.t())
    
    # Hungarian algorithm for matching
    idx1, idx2 = hungarian(d)
    
    # Return the matching results
    return points1[idx1], points2[idx2]

def data_loaded():
    # Define date and time filter
    date_key = '03-11-2022'

    # PATHS, dataframe and shpfile #
    base_path = "C:\\Users\\abelt\\OneDrive\\Desktop\\Kandidat\\"
    
    # AIS file paths
    ais_files = {
        '02-11-2022': 'ais\\ais_110215.csv',
        '03-11-2022': 'ais\\ais_110315.csv',
        '05-11-2022': 'ais\\ais_1105.csv'
    }

    # Loading data using the DataLoader class
    data_loader = DataLoader(base_path=base_path, ais_files=ais_files, date_key=date_key)
    ais_loader = data_loader.load_data()

    # Return a copy of the specified AIS data
    return ais_loader.dfs_ais[date_key].copy()

def train():
    # Define model architecture and parameters
    EMBED_DIM = 4
    model = nn.Sequential(
        MLP(4, 128, 100),  # First MLP
        MLP(100, 128, EMBED_DIM)  # Second MLP
    )
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create dataset and dataloader
    dataset = ais_dataset(None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)
    
    for batch in dataloader:
        ais1, ais2 = batch  # Get the batched pairs from the dataset

        # Forward pass: get similarity matrix from the model
        emb1 = model(ais1)  # Embed ais1 points
        emb2 = model(ais2)  # Embed ais2 points
        
        # Compute distance (or similarity) matrix using dot product
        d = torch.matmul(emb1, emb2.t())
        d = torch.softmax(d, dim=1)  # Apply softmax to convert to probabilities
        
        # Calculate loss (negative log likelihood of correct match)
        diag = torch.diag(d)  # Get diagonal elements (i.e., matching pairs)
        loss = -torch.log(diag).mean()  # Calculate the loss as mean of log probabilities
        print(f'Loss: {loss.item()}')
        
        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

        # Stop after the first batch for demonstration
        break

if __name__ == '__main__':
    train()
