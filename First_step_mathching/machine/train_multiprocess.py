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

def loss_fn(dist_matrix, temperature=1, eps=1e-8):
    scaled_dist_matrix = dist_matrix / temperature
    diag = torch.diag(torch.softmax(scaled_dist_matrix, dim=1))
    return -torch.log(diag + eps)

def forward(batch, model, device):
    ais1, ais2 = batch[0].to(device), batch[1].to(device)
    combined_batch = torch.cat([ais1, ais2], dim=0)
    emb1, emb2 = model(combined_batch).chunk(2, dim=0)
    return model.distance_matrix(emb1, emb2)


def metrics(dist_matrix, device):
    correct = (torch.argmax(dist_matrix, dim=1) == torch.arange(dist_matrix.size(0), device=device)).type(torch.float).mean().item()
    dist = torch.max(dist_matrix, dim=1)[0].mean().item()
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
    loss = loss_fn(dist_matrix, temperature).sum()
    correct, mean_dist = metrics(dist_matrix, device)
        
    return loss.item(), correct, mean_dist

# Modified train function to accept shared data and barrier
def train(model, rank, datetime_str, resume_training, model_params, shared_dict, barrier):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = model_params.get('input_dim', 2)
    HIDDEN_DIM = model_params.get('hidden_dim', 1024)
    DEPTH = model_params.get('depth', 2)
    LEARNING_RATE = model_params.get('learning rate', 0.0001)
    DROP_PROB = model_params.get('dropout probability', 0.2)
    TEMPERATURE = model_params.get('temperature', 1)

    batch_size = model_params.get('batch size', 64)
    train_iterations = model_params.get('train iterations', 100000)
    val_iterations = model_params.get('validation iterations', int(train_iterations * .2))
    early_stop = model_params.get('early stop', 300)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dataset = ais_dataset(ais_dataset.import_data_fn(), grouping_id='track_id')
    train_loader, val_loader = map(lambda x: cycle(x), ais_dataset.split_dataset(dataset, batch_size=batch_size))

    # Track the best validation loss
    best_val_loss = float('inf')

    start_epoch, train_loss, val_loss, val_losses, train_corrects, val_corrects = (0, 0, 0, [], [], []) if not resume_training else Checkpoint.load_checkpoint(model, optimizer, checkpoint_path)

    epoch = start_epoch

    while True:
        # Training loop
        model.train()
        train_loss = 0
        train_correct = 0
        
        for i, batch in enumerate(train_loader):
            loss, correct = train_step(batch, model, loss_fn, device, optimizer, TEMPERATURE)
            train_loss += loss
            train_correct += correct
            
            if i * batch_size >= train_iterations:
                break

        avg_train_correct = train_correct / (i + 1)

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        
        for j, batch in enumerate(val_loader):
            loss, correct = val_step(batch, model, loss_fn, device, TEMPERATURE)
            val_loss += loss
            val_correct += correct

            if j * batch_size >= val_iterations:
                break

        avg_val_correct = val_correct / (j + 1)

        # Store metrics in the shared dictionary
        shared_dict[rank] = {'epoch': epoch, 'train_loss': train_loss / train_iterations, 'val_loss': val_loss / val_iterations, 'train_correct': avg_train_correct, 'val_correct': avg_val_correct}

        # Barrier: Wait until all processes finish the epoch
        barrier.wait()

        # Only rank 0 evaluates all results from all processes
        if rank == 0:
            # Aggregate or compare the results from all processes
            all_metrics = shared_dict.values()
            print(f'Epoch {epoch + 1} Results Across Processes:')
            for metrics in all_metrics:
                print(metrics)
                

        # Barrier: Wait before starting the next epoch
        barrier.wait()

        epoch += 1

if __name__ == '__main__':
    num_cores = mp.cpu_count()
    print(num_cores)
    num_processes = num_cores - 7
    model_params = {
        'input_dim': 2, 
        'hidden_dim': 1024, 
        'output_dim': 2, 
        'depth': 10, 
        'batch size': 1024,
        'temperature': 1
    }
    
    model = MyModel(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=model_params['output_dim'],
        depth=model_params['depth'],
        drop_prob=0.2
    )
    
    # Share memory between processes
    model.share_memory()

    # Create a Manager to handle shared data between processes
    manager = Manager()
    shared_dict = manager.dict()  # Shared dictionary to store metrics

    # Create a barrier to synchronize processes after each epoch
    barrier = Barrier(num_processes)

    # Generate datetime_str once
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    
    processes = []
    
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model, rank, datetime_str, model_params, shared_dict, barrier))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()