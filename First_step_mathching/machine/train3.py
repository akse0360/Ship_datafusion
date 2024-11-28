import os
from tqdm import tqdm
import torch
from datetime import datetime
from machine.mlp import MyModel
from First_step_mathching.machine.ai_loader import ais_dataset
from machine.helperFunctiontions import Checkpoint

def cycle(iterable):
    while True:
        yield from iterable


def initialize_model_optimizer(model_params, device):
    model = MyModel(
        input_dim=model_params.get('input_dim', 2),
        hidden_dim=model_params.get('hidden_dim', 256),
        output_dim=model_params.get('input_dim', 2),
        depth=model_params.get('depth', 2),
        drop_prob=model_params.get('dropout probability', 0.2)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_params.get('learning rate', 0.0005), weight_decay=0.01)
    return model, optimizer

def load_data(batch_size):
    dataset = ais_dataset(ais_dataset.import_data_fn(), grouping_id='track_id')
    train_loader, val_loader = map(cycle, ais_dataset.split_dataset(dataset, batch_size=batch_size))
    return train_loader, val_loader

def train_step(batch, model, loss_fn, optimizer, device):
    ais1, ais2 = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    combined_batch = torch.cat([ais1, ais2], dim=0)
    emb1, emb2 = model(combined_batch).chunk(2, dim=0)
    dist_matrix = model.distance_matrix(emb1, emb2)
    loss = loss_fn(dist_matrix, temperature=temperature)
    loss.sum().backward()
    optimizer.step()

    return loss.sum()

def validation_step(batch, model, loss_fn, device):
    ais1, ais2 = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        combined_batch = torch.cat([ais1, ais2], dim=0)
        emb1, emb2 = model(combined_batch).chunk(2, dim=0)
        loss = loss_fn(emb1, emb2)
        
    return loss.sum()

def train_epoch(train_loader, model, loss_fn, optimizer, device, train_iterations, temperature):
    model.train()
    train_loss = 0
    for i, batch in enumerate(tqdm(train_loader, total=train_iterations, desc='Training', leave=False)):
        train_loss += train_step(batch, model, loss_fn, optimizer, device)
        if i >= train_iterations:
            break
    return train_loss / train_iterations

def validate_epoch(val_loader, model, loss_fn, device, val_iterations, temperature, batch_size):
    model.eval()
    val_loss = 0
    val_loader_tqdm = tqdm(val_loader, total=val_iterations, desc='Validation', leave=False)
    for i, batch in enumerate(val_loader_tqdm):
        #val_loss += validation_step(batch, model, loss_fn, device, temperature)
        val_loss += validation_step(batch, model, loss_fn, device)
        if i >= val_iterations:
            break
    return val_loss / (val_iterations * batch_size)

def is_correctly_matched(model, emb1, emb2):
    # Compute the distance matrix
    dist_matrix = model.distance_matrix(emb1, emb2)
    
    # Find the indices of the minimum values along each row
    min_indices = torch.argmin(dist_matrix, dim=1)
    
    # Check if these indices correspond to the diagonal elements
    is_min_in_diagonal = torch.all(min_indices == torch.arange(len(min_indices)).to(min_indices.device))
    
    # Print the result
    print(f"Distance matrix:\n{dist_matrix}")
    print(f"Indices of minimum values along each row: {min_indices}")
    print(f"Is the entire diagonal the smallest values? {'Yes' if is_min_in_diagonal else 'No'}")
    
    # Return whether the match is correct
    return is_min_in_diagonal

def train(model_params, resume_training=False, checkpoint_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_path_{datetime_str}.pth")
    
    model, optimizer = initialize_model_optimizer(model_params, device)
    train_loader, val_loader = load_data(model_params.get('batch size', 64))
    
    # Modified loss function to minimize distances along the diagonal
    def loss_fn(emb1, emb2, margin=1.0):
        dist_matrix = model.distance_matrix(emb1, emb2)
        batch_size = dist_matrix.size(0)
        diagonal_loss = torch.diag(dist_matrix)
        off_diagonal_loss = dist_matrix + torch.eye(batch_size).to(dist_matrix.device) * 1e9
        off_diagonal_loss = torch.min(off_diagonal_loss, dim=1)[0]
        return torch.mean(diagonal_loss) + torch.mean(torch.relu(diagonal_loss - off_diagonal_loss + margin))
    
    temperature = model_params.get('temperature', 1)
    
    start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []
    if resume_training:
        start_epoch, train_loss, val_loss, val_losses = Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)

    j = start_epoch
    early_stop = model_params.get('early stop', 10)  # Number of epochs to wait for improvement
    while True:
        saved, train_loss = False, 0
        model.train()
        train_loader_tqdm = tqdm(train_loader, total=model_params.get('train iterations', 1000) // model_params.get('batch size', 64), desc=f'Epoch {j+1} [Training]', leave=False)

        # Training Loop
        for i, batch in enumerate(train_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)
            combined_batch = torch.cat([ais1, ais2], dim=0)
            emb1, emb2 = model(combined_batch).chunk(2, dim=0)
            loss = loss_fn(emb1, emb2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * model_params.get('batch size', 64))})
            if i >= model_params.get('train iterations', 1000) // model_params.get('batch size', 64):
                break

        train_loss /= model_params.get('train iterations', 1000)
        
        # Validation Loop
        val_loss = validate_epoch(val_loader, model, loss_fn, device, model_params.get('validation iterations', 100), temperature, model_params.get('batch size', 64))

        # Checkpointing and Logging
        if val_loss <= min(val_losses, default=val_loss):
            saved = True
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{datetime_str}.pth"))
        
        val_losses.append(val_loss)
        Checkpoint.save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)
        Checkpoint.log_training(epoch=j+1, train_loss=train_loss, train_iterations=model_params.get('train iterations', 1000), val_loss=val_loss, val_iterations=model_params.get('validation iterations', 100), saved=saved, file_name=f"training_log_{datetime_str}.csv")
        
        tqdm.write(f'Epoch {j+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, model_saved={saved}')
        
        # Every 5 epochs, check if the smallest distances are along the diagonal
        if (j + 1) % 5 == 0:
            for batch in train_loader:
                ais1, ais2 = batch[0].to(device), batch[1].to(device)
                combined_batch = torch.cat([ais1, ais2], dim=0)
                emb1, emb2 = model(combined_batch).chunk(2, dim=0)
                is_correctly_matched(model, emb1, emb2)
                break  # Check only the first batch for simplicity
        
        # Early Stopping Condition: Check if no improvement in `early_stop` epochs
        if j > early_stop and all(val_losses[j] >= val_losses[j - k - 1] for k in range(early_stop)):
            tqdm.write(f"Early stopping after {early_stop} epochs with no improvement.")
            break
        
        j += 1

# def train_original(model_params, resume_training=False, checkpoint_dir="models"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
#     checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_path_{datetime_str}.pth")
    
#     model, optimizer = initialize_model_optimizer(model_params, device)
#     train_loader, val_loader = load_data(model_params.get('batch size', 64))
#     loss_fn = lambda dist_matrix, temperature: -torch.log(torch.diag(torch.softmax(dist_matrix / temperature, dim=1)))
#     temperature = model_params.get('temperature', 1)
    
#     start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []
#     if resume_training:
#         start_epoch, train_loss, val_loss, val_losses = Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)

#     for epoch in range(start_epoch, model_params.get('epochs', 100)):
#         train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, model_params.get('train iterations', 1000), temperature)
#         val_loss = validate_epoch(val_loader, model, loss_fn, device, model_params.get('validation iterations', 100), temperature, model_params.get('batch size', 64))
        
#         saved = False
#         if val_loss <= min(val_losses, default=val_loss):
#             saved = True
#             torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{datetime_str}.pth"))
        
#         val_losses.append(val_loss)
#         Checkpoint.save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_losses, checkpoint_path)
#         Checkpoint.log_training(epoch=epoch+1, train_loss=train_loss, train_iterations=model_params.get('train iterations', 1000), val_loss=val_loss, val_iterations=model_params.get('validation iterations', 100), saved=saved, file_name=f"training_log_{datetime_str}.csv")
        
#         tqdm.write(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, model_saved={saved}')
        
#         if epoch > model_params.get('early stop', 30) and all(vl >= val_loss for vl in val_losses[epoch-model_params.get('early stop', 30):]):
#             break

# def train4(model_params, resume_training=False, checkpoint_dir="models"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
#     checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_path_{datetime_str}.pth")
    
#     model, optimizer = initialize_model_optimizer(model_params, device)
#     train_loader, val_loader = load_data(model_params.get('batch size', 64))
    
#     loss_fn = lambda dist_matrix, temperature: -torch.log(torch.diag(torch.softmax(dist_matrix / temperature, dim=1)))
#     temperature = model_params.get('temperature', 1)
    
#     start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []
#     if resume_training:
#         start_epoch, train_loss, val_loss, val_losses = Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)

#     j = start_epoch
#     early_stop = model_params.get('early stop', 10)  # Number of epochs to wait for improvement
#     while True:
#         saved, train_loss = False, 0
#         model.train()
#         train_loader_tqdm = tqdm(train_loader, total=model_params.get('train iterations', 1000) // model_params.get('batch size', 64), desc=f'Epoch {j+1} [Training]', leave=False)

#         # Training Loop
#         for i, batch in enumerate(train_loader_tqdm):
#             train_loss += train_step(batch, model, loss_fn, optimizer, device, temperature)
#             train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * model_params.get('batch size', 64))})
#             if i >= model_params.get('train iterations', 1000) // model_params.get('batch size', 64):
#                 break

#         train_loss /= model_params.get('train iterations', 1000)
        
#         # Validation Loop
#         val_loss = validate_epoch(val_loader, model, loss_fn, device, model_params.get('validation iterations', 100), temperature, model_params.get('batch size', 64))

#         # Checkpointing and Logging
#         if val_loss <= min(val_losses, default=val_loss):
#             saved = True
#             torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{datetime_str}.pth"))
        
#         val_losses.append(val_loss)
#         Checkpoint.save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)
#         Checkpoint.log_training(epoch=j+1, train_loss=train_loss, train_iterations=model_params.get('train iterations', 1000), val_loss=val_loss, val_iterations=model_params.get('validation iterations', 100), saved=saved, file_name=f"training_log_{datetime_str}.csv")
        
#         tqdm.write(f'Epoch {j+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, model_saved={saved}')
        
#         # Early Stopping Condition: Check if no improvement in `early_stop` epochs
#         if j > early_stop and all(val_losses[j] >= val_losses[j - k - 1] for k in range(early_stop)):
#             tqdm.write(f"Early stopping after {early_stop} epochs with no improvement.")
#             break
        
#         j += 1


if __name__ == '__main__':
    num_features = 2
    model_params = {'input_dim': num_features, 'hidden_dim': 256, 'output_dim': num_features, 'depth': 3, 'batch size': 5*32}
    train(model_params=model_params)