import os
import random
import numpy as np
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from mlp import MyModel, hungarian
from data_loader import ais_dataset
from scripts.data_loader import DataLoader as DL

def data_loaded():
    # Define date and time filter
    date_key = '03-11-2022'

    # PATHS, dataframe and shpfile #
    # Define paths
    base_path = "C:\\Users\\abelt\\OneDrive\\Desktop\\Kandidat\\"
    ## File names ##
    # AIS
    ais_files = {
        '02-11-2022': 'ais\\ais_110215.csv',
        '03-11-2022': 'ais\\ais_110315.csv',
        '05-11-2022': 'ais\\ais_1105.csv'
    }

    # LOADING #
    data_loader = DL(base_path = base_path, ais_files = ais_files, date_key = date_key)
    ais_loader, sar, norsat = data_loader.load_data()
    return ais_loader.dfs_ais[date_key].copy()

def split_dataset(dataset, split_ratio=0.8, seed=1,batch_size=3):
    indices = list(range(len(dataset)))

    num_train = int(split_ratio * len(indices))
    random.seed(seed)
    indices = sorted(indices, key=lambda x: random.random())
    train_indices = indices[:num_train]
    valid_indices = indices[num_train:]

    #Create samplers for training and validation
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    #Create data loaders using the samplers
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, pin_memory=True)
    return train_loader, val_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_losses, file_name="models/checkpoint.pth"):
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_losses': val_losses
    }
    torch.save(checkpoint, file_name)

def load_checkpoint(model, optimizer, file_name="models/checkpoint.pth"):
    
    if os.path.isfile(file_name):
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_losses = checkpoint['val_losses']
        return epoch, train_loss, val_loss, val_losses
    else:
        return 0, 0, 0, []

def infer(points1, points2, model_path="model.pth"):
    # Load the trained model
    input_dim = 2 
    hidden_dim = 256
    output_dim = 2 

    model = MyModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  

    # Convert input points to tensors
    points1 = torch.tensor(points1, dtype=torch.float32)
    points2 = torch.tensor(points2, dtype=torch.float32)

    
    d1 = model.forward(torch.stack([points1, points2], dim=0))

    similarity_matrix = torch.matmul(d1, d2.t())

    idx1, idx2 = hungarian(similarity_matrix)

    return idx1, idx2

def loss_fn(dist_matrix):
    d1 = torch.softmax(dist_matrix, dim=1) # TODO temperature scaling
    diag = torch.diag(d1)
    return -torch.log(diag)

def step(batch, model, loss_fn, optimizer=None):
        if optimizer:
            optimizer.zero_grad() 
        ais1, ais2 = batch
        dist_matrix = model.distance_matrix(ais1, ais2)
        loss = loss_fn(dist_matrix)
        if optimizer:
            loss.mean().backward()  
            optimizer.step()
        return loss.sum().item()

def train_epoch_tqdm(resume_training=False, checkpoint_path="models/checkpoint.pth"):
    EMBED_DIM = 2 # Number of features to study e.g. lat, lon, cog, sog
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=256, output_dim=EMBED_DIM, depth=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    dataset = ais_dataset(data_loaded())
    train_loader, val_loader = map(lambda x: cycle(x), split_dataset(dataset, batch_size=batch_size))

    train_iterations = 1000
    val_iterations = 100
    early_stop = 250

    if resume_training:
        start_epoch, train_loss, val_loss, val_losses = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []

    j = start_epoch

    while True:
        saved = False
        model.train()
        train_loss = 0

        # Use tqdm for the training loop
        train_loader_tqdm = tqdm(train_loader, total=train_iterations // batch_size, desc=f'Epoch {j+1} [Training]', leave=False)
        for i, batch in enumerate(train_loader_tqdm):
            train_loss += step(batch, model, loss_fn, optimizer)
            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * batch_size)})
            if i * batch_size >= train_iterations:
                break

        model.eval()
        val_loss = 0

        # Use tqdm for the validation loop
        val_loader_tqdm = tqdm(val_loader, total=val_iterations // batch_size, desc=f'Epoch {j+1} [Validation]', leave=False)
        for i, batch in enumerate(val_loader_tqdm):
            val_loss += step(batch, model, loss_fn)
            val_loader_tqdm.set_postfix({'val_loss': val_loss / ((i + 1) * batch_size)})
            if i * batch_size >= val_iterations:
                break

        val_losses.append(val_loss)

        if val_losses[j] <= min(val_losses):
            saved = True
            torch.save(model.state_dict(), "models/model_epoch.pth")

        save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)

        # Progress feedback after each epoch
        tqdm.write(f'Epoch {j+1}: train_loss={train_loss/train_iterations:.4f}, val_loss={val_loss/val_iterations:.4f}, model_saved={saved}')

        if j > early_stop and all(
            [val_losses[j] >= vl for vl in val_losses[j-early_stop:j]]
        ):
            break
        elif math.isnan(val_losses[j]):
            break

        j += 1

if __name__ == '__main__':
    #train_epoch()
    train_epoch_tqdm()

    #infer(points1, points2, model_path="model.pth")