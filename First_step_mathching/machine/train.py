import torch
from tqdm import tqdm
from datetime import datetime
from machine.mlp import MyModel
from machine.ai_loader import ais_dataset
from machine.helperFunctiontions import Checkpoint

import multiprocessing as mp
from multiprocessing import Manager, Barrier

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def loss_fn(dist_matrix, temperature=1):
    scaled_dist_matrix = dist_matrix / temperature
    diag = torch.diag(torch.softmax(scaled_dist_matrix, dim=1))
    return -diag**2
    return -torch.clamp(torch.log(diag),-100)


def forward(batch, model, device):
    ais1, ais2 = batch[0].to(device), batch[1].to(device)
    # Check the shape of ais1 and ais2
    # print(f"ais1 shape: {ais1.shape}, ais2 shape: {ais2.shape}")
    combined_batch = torch.cat([ais1, ais2], dim=0)
    # Check the shape of combined_batch before passing to the model
    # print(f"combined_batch shape: {combined_batch.shape}")    
    emb1, emb2 = model(combined_batch).chunk(2, dim=0)
    # Check the shape of emb1 and emb2
    #print(f"emb1 shape: {emb1}, emb2 shape: {emb2.shape}")
    #emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    #emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    diag_matrix = torch.cdist(emb1, emb2) # torch.matmul(emb1, emb2)

    return diag_matrix

def metrics(dist_matrix, device):
    correct = (torch.argmax(dist_matrix, dim=1) == torch.arange(dist_matrix.size(0), device=device)).type(torch.float).mean().item()
    dist = torch.diag(dist_matrix).mean().item()

    return correct, dist

def train_step(batch, model, loss_fn, device, optimizer, temperature):
    optimizer.zero_grad()
    dist_matrix = forward(batch, model, device)
    loss = loss_fn(dist_matrix, temperature).sum()
    correct, mean_dist = metrics(dist_matrix, device)
    
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), correct, mean_dist
    
def val_step(batch, model, loss_fn, device, temperature):
    dist_matrix = forward(batch, model, device)
    # print(dist_matrix)
    loss = loss_fn(dist_matrix, temperature).sum()
    correct, mean_dist = metrics(dist_matrix, device)
        
    return loss.item(), correct, mean_dist

def train_epoch_tqdm(model_params: dict = None, resume_training: bool = False, checkpoint_path: str = "models/checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Model parameters
    EMBED_DIM = model_params.get('input_dim', 2)
    HIDDEN_DIM = model_params.get('hidden_dim', 1024)
    OUTPUT_DIM = model_params.get('output_dim', 128)
    DEPTH = model_params.get('depth', 2)
    LEARNING_RATE = model_params.get('learning rate', 0.0001)
    DROP_PROB = model_params.get('dropout probability', 0.2)
    TEMPERATURE = model_params.get('temperature', 1)

    batch_size = model_params.get('batch size', 64)
    train_iterations = model_params.get('train iterations', 100000)
    val_iterations = model_params.get('validation iterations', int(train_iterations * .2))
    early_stop = model_params.get('early stop', 300)
    
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, depth=DEPTH, drop_prob=DROP_PROB).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    dataset = ais_dataset(ais_dataset.import_data_fn(), grouping_id='track_id')
    train_loader, val_loader = map(lambda x: cycle(x), ais_dataset.split_dataset(dataset, batch_size=batch_size))
    
    start_epoch, train_loss, val_loss, val_losses, train_corrects, val_corrects = (0, 0, 0, [], [], []) if not resume_training else Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)

    epoch = start_epoch

    while True:
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_mean_dist = 0

        train_loader_tqdm = tqdm(train_loader, total=train_iterations // batch_size, desc=f'Epoch {epoch+1} [Training]', leave=False)
        for i, batch in enumerate(train_loader_tqdm):
            loss, correct, mean_dist = train_step(batch, model, loss_fn, device, optimizer, TEMPERATURE)
            train_loss += loss
            train_correct += correct
            train_mean_dist += mean_dist

            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * batch_size), 'correct': int(100 * train_correct / (i + 1)), 'mean_dist': train_mean_dist})
            
            if i * batch_size >= train_iterations:
                break   
        
        avg_train_correct = train_correct / (i + 1)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_mean_dist = 0

        val_loader_tqdm = tqdm(val_loader, total=val_iterations // batch_size, desc=f'Epoch {epoch+1} [Validation]', leave=False)
        for j, batch in enumerate(val_loader_tqdm):
            loss, correct, mean_dist = val_step(batch, model, loss_fn, device, TEMPERATURE)
            val_loss += loss
            val_correct += correct
            val_mean_dist += mean_dist

            val_loader_tqdm.set_postfix({'val_loss': val_loss / ((j + 1) * batch_size), 'correct': int(100 * val_correct / (j + 1)), 'mean_dist': val_mean_dist})
            
            if j * batch_size >= val_iterations:
                break
        
        avg_val_correct = val_correct / (j + 1)
        
        # Adjust model saving condition
        val_losses.append(val_loss)
        train_corrects.append(avg_train_correct)
        val_corrects.append(avg_val_correct)
        saved = False  # Initialize saved as False

        # Save model only if validation loss improves within the early_stop window
        if val_loss <= min(val_losses[-early_stop:]):
            torch.save(model.state_dict(), f"models/model_best_{datetime_str}.pth")
            saved = True  # Set saved to True only if model was saved

        # Log training results and save checkpoint
        Checkpoint.save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_losses, train_corrects, val_corrects, checkpoint_path)
        Checkpoint.log_training(
            epoch=epoch,
            train_loss=train_loss / train_iterations,
            val_loss=val_loss / val_iterations,
            avg_train_correct=avg_train_correct,
            avg_val_correct=avg_val_correct,
            saved=saved,
            file_name=f"training_log_{datetime_str}.csv"
        )

        tqdm.write(f'Epoch {epoch + 1}: train_loss={train_loss / train_iterations:.4f}, val_loss={val_loss / val_iterations:.4f}, train_correct={avg_train_correct:.4f}, val_correct={avg_val_correct:.4f}, model_saved={saved}')

        # Early stopping logic
        if len(val_losses) > early_stop and all(
                [val_losses[epoch] >= vl for vl in val_losses[-early_stop:]]
            ):
            print("No improvement stop triggered.")
            break

        epoch += 1

if __name__ == '__main__':
    model_params = {'input_dim': 2,             # Number of features: lat, lon ect.
                    'hidden_dim': 1024,         # Number of hidden units
                    'output_dim': 128,          # Size of latent space
                    'depth': 10,                # Number of hidden layers
                    'batch size': 1024,         # Number of samples per batch
                    'learning rate' : 0.0005,   # Learning rate
                    'temperature' : 1}        # Temperature 
    
    train_epoch_tqdm(model_params=model_params, resume_training = False) #, checkpoint_path = "models/checkpoint.pth"
