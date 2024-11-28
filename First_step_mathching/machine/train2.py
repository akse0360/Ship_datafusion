import os

from tqdm import tqdm
import torch

from machine.mlp import MyModel, hungarian
from First_step_mathching.machine.ai_loader import ais_dataset
from machine.helperFunctiontions import Checkpoint

import datetime 
from datetime import datetime 

def cycle(iterable):
    while True:
        yield from iterable

def loss_fn(dist_matrix, temperature=1):
    scaled_dist_matrix = dist_matrix / temperature
    diag = torch.diag(torch.softmax(scaled_dist_matrix, dim=1))
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

def train_epoch_tqdm(model_params: dict = None, resume_training: bool = False, checkpoint_dir: str = "models"):
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_path_{datetime_str}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    EMBED_DIM = model_params.get('input_dim', 2)
    HIDDEN_DIM = model_params.get('hidden_dim', 256)
    DEPTH = model_params.get('depth', 2)
    LEARNIGN_RATE = model_params.get('learning rate', 0.0005)
    DROP_PROB = model_params.get('dropout probability', 0.2)  
    TEMPERATURE = model_params.get('temperature', 1)
    
    batch_size = model_params.get('batch size', 64)
    train_iterations = model_params.get('train iterations', 1000)
    val_iterations = model_params.get('validation iterations', 100)
    early_stop = model_params.get('early stop', 150)

    # Initialize the model, optimizer and scaler
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, output_dim=EMBED_DIM, depth=DEPTH, drop_prob=DROP_PROB).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNIGN_RATE,weight_decay=0.01)
    scaler = torch.amp.GradScaler()

    dataset = ais_dataset(ais_dataset.import_data_fn(), grouping_id = 'track_id')
    train_loader, val_loader = map(cycle, ais_dataset.split_dataset(dataset, batch_size=batch_size))

    if resume_training:
        start_epoch, train_loss, val_loss, val_losses = Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []

    j = start_epoch
    while True:
        saved, train_loss = False, 0
        model.train()
        train_loader_tqdm = tqdm(train_loader, total=train_iterations // batch_size, desc=f'Epoch {j+1} [Training]', leave=False)
        
        for i, batch in enumerate(train_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float32):
                combined_batch = torch.cat([ais1, ais2], dim=0)
                emb1, emb2 = model(combined_batch).chunk(2, dim=0)
                dist_matrix = model.distance_matrix(emb1, emb2)
                loss = loss_fn(dist_matrix, temperature=TEMPERATURE)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
            #scaler.step(optimizer)
            #scaler.update()

            train_loss += loss.sum()
            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * batch_size)})
            if i * batch_size >= train_iterations:
                break

        model.eval()
        val_loss = 0
        val_loader_tqdm = tqdm(val_loader, total=val_iterations // batch_size, desc=f'Epoch {j+1} [Validation]', leave=False)
        
        for i, batch in enumerate(val_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
                combined_batch = torch.cat([ais1, ais2], dim=0)
                emb1, emb2 = model(combined_batch).chunk(2, dim=0)
                dist_matrix = model.distance_matrix(emb1, emb2)
                loss = loss_fn(dist_matrix, temperature=TEMPERATURE)

            val_loss += loss.sum()
            val_loader_tqdm.set_postfix({'val_loss': val_loss / ((i + 1) * batch_size)})
            if i * batch_size >= val_iterations:
                break

        val_losses.append(val_loss)

        if val_losses[j] <= min(val_losses):
            saved = True
            torch.save(model.state_dict(), f"models/model_{datetime_str}.pth")

        Checkpoint.save_checkpoint(model, optimizer, j, train_loss.sum(), val_loss.sum(), val_losses, checkpoint_path)
        Checkpoint.log_training(epoch=j+1, train_loss=train_loss.sum(), train_iterations=train_iterations, val_loss=val_loss.sum(), val_iterations=val_iterations, saved=saved, file_name=f"training_log_{datetime_str}.csv")
        
        tqdm.write(f'Epoch {j+1}: train_loss={train_loss/train_iterations:.4f}, val_loss={val_loss/val_iterations:.4f}, model_saved={saved}')

        if j > early_stop and all(
            [val_losses[j] >= vl for vl in val_losses[j-early_stop:j]]
            ):
            break
        j += 1

if __name__ == '__main__':
    num_features = 2
    model_params = {'input_dim': num_features, 'hidden_dim': 512, 'output_dim': num_features, 'depth': 3, 'batch size': 32*10}
    train_epoch_tqdm(model_params=model_params)
