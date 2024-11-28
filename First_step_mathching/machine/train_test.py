def train_epoch_tqdm(resume_training=False, checkpoint_path="models/checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    EMBED_DIM = 2  # Number of features to study e.g., lat, lon, cog, sog
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=1024, output_dim=EMBED_DIM, depth=5).to(device)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Initialize GradScaler for automatic mixed precision
    scaler = torch.amp.GradScaler()

    batch_size = 64
    dataset = ais_dataset(data_loaded())
    train_loader, val_loader = map(lambda x: cycle(x), split_dataset(dataset, batch_size=batch_size))

    train_iterations = 1000
    val_iterations = 100
    early_stop = 300

    if resume_training:
        start_epoch, train_loss, val_loss, val_losses = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []

    j = start_epoch

    while True:
        saved = False
        model.train()
        train_loss = 0

        train_loader_tqdm = tqdm(train_loader, total=train_iterations // batch_size, desc=f'Epoch {j+1} [Training]', leave=False)
        for i, batch in enumerate(train_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast('cuda'):
                # Forward pass
                combined_batch = torch.cat([ais1, ais2], dim=0)
                embeddings = model(combined_batch)
                
                # Split embeddings for ais1 and ais2 and compute the distance matrix
                emb1, emb2 = embeddings.chunk(2, dim=0)
                dist_matrix = torch.matmul(emb1, emb2.T)
                
                # Compute loss
                loss = loss_fn(dist_matrix)  
            scaler.scale(loss.mean()).backward()  

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.sum().item()  # Or loss.mean().item(),

            # Update the progress bar
            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * batch_size)})

            if i * batch_size >= train_iterations:
                break

        model.eval()
        val_loss = 0

        # Use tqdm for the validation loop
        val_loader_tqdm = tqdm(val_loader, total=val_iterations // batch_size, desc=f'Epoch {j+1} [Validation]', leave=False)
        for i, batch in enumerate(val_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)  # Move data to device

            with torch.no_grad():
                # Use autocast for validation too
                with torch.amp.autocast('cuda'):
                    # Forward pass and distance matrix calculation
                    combined_batch = torch.cat([ais1, ais2], dim=0)
                    embeddings = model(combined_batch)
                    
                    emb1, emb2 = embeddings.chunk(2, dim=0)
                    dist_matrix = torch.matmul(emb1, emb2.T)
                    
                    # Compute validation loss
                    loss = loss_fn(dist_matrix)  # Calculate validation loss (returns tensor)

            val_loss += loss.mean().item()  # Or loss.sum().item()

            val_loader_tqdm.set_postfix({'val_loss': val_loss / ((i + 1) * batch_size)})

            if i * batch_size >= val_iterations:
                break

        val_losses.append(val_loss)

        # Save the model if validation loss improved
        if val_losses[j] <= min(val_losses):
            saved = True
            torch.save(model.state_dict(), "models/model_s.pth")

        save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)

        # Progress feedback after each epoch
        log_training(epoch=j+1, train_loss=train_loss, train_iterations=train_iterations, 
                     val_loss=val_loss, val_iterations=val_iterations, saved=saved)

        tqdm.write(f'Epoch {j+1}: train_loss={train_loss/train_iterations:.4f}, val_loss={val_loss/val_iterations:.4f}, model_saved={saved}')

        # Early stopping logic
        if j > early_stop and all(
            [val_losses[j] >= vl for vl in val_losses[j-early_stop:j]]
            ):
            break
        elif math.isnan(val_losses[j]):
            break

        j += 1



#######################################################################
def train_epoch_tqdm3(resume_training=False, checkpoint_path="models/checkpoint.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    EMBED_DIM = 2  # Number of features to study e.g. lat, lon, cog, sog
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=1024, output_dim=EMBED_DIM, depth=5).to(device)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Initialize GradScaler for automatic mixed precision
    scaler = torch.amp.GradScaler()

    batch_size = 64
    dataset = ais_dataset(data_loaded())
    train_loader, val_loader = map(lambda x: cycle(x), split_dataset(dataset, batch_size=batch_size))

    train_iterations = 1000
    val_iterations = 100
    early_stop = 300

    if resume_training:
        start_epoch, train_loss, val_loss, val_losses = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch, train_loss, val_loss, val_losses = 0, 0, 0, []

    j = start_epoch

    while True:
        saved = False
        model.train()
        train_loss = 0

        train_loader_tqdm = tqdm(train_loader, total=train_iterations // batch_size, desc=f'Epoch {j+1} [Training]', leave=False)
        for i, batch in enumerate(train_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast('cuda'):
                dist_matrix = model.distance_matrix(ais1, ais2)  # Forward pass
                loss = loss_fn(dist_matrix)  
            scaler.scale(loss.mean()).backward()  

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.sum().item()  # Or loss.mean().item(),

            # Update the progress bar
            train_loader_tqdm.set_postfix({'train_loss': train_loss / ((i + 1) * batch_size)})

            if i * batch_size >= train_iterations:
                break

        model.eval()
        val_loss = 0

        # Use tqdm for the validation loop
        val_loader_tqdm = tqdm(val_loader, total=val_iterations // batch_size, desc=f'Epoch {j+1} [Validation]', leave=False)
        for i, batch in enumerate(val_loader_tqdm):
            ais1, ais2 = batch[0].to(device), batch[1].to(device)  # Move data to device

            with torch.no_grad():
                # Use autocast for validation too
                with torch.amp.autocast('cuda'):
                    dist_matrix = model.distance_matrix(ais1, ais2)  # Forward pass
                    loss = loss_fn(dist_matrix)  # Calculate validation loss (returns tensor)

            val_loss += loss.mean().item()  # Or loss.sum().item()

            val_loader_tqdm.set_postfix({'val_loss': val_loss / ((i + 1) * batch_size)})

            if i * batch_size >= val_iterations:
                break

        val_losses.append(val_loss)

        # Save the model if validation loss improved
        if val_losses[j] <= min(val_losses):
            saved = True
            torch.save(model.state_dict(), "models/model_s.pth")

        save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)

        # Progress feedback after each epoch
        log_training(epoch=j+1, train_loss=train_loss, train_iterations=train_iterations, 
                     val_loss=val_loss, val_iterations=val_iterations, saved=saved)

        tqdm.write(f'Epoch {j+1}: train_loss={train_loss/train_iterations:.4f}, val_loss={val_loss/val_iterations:.4f}, model_saved={saved}')

        # Early stopping logic
        if j > early_stop and all(
            [val_losses[j] >= vl for vl in val_losses[j-early_stop:j]]
            ):
            break
        elif math.isnan(val_losses[j]):
            break

        j += 1

#####################################################
def train_epoch_tqdm2(resume_training=False, checkpoint_path="models/checkpoint.pth"):
    EMBED_DIM = 2 # Number of features to study e.g. lat, lon, cog, sog
    model = MyModel(input_dim=EMBED_DIM, hidden_dim=1024, output_dim=EMBED_DIM, depth=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    batch_size = 64
    dataset = ais_dataset(data_loaded())
    train_loader, val_loader = map(lambda x: cycle(x), split_dataset(dataset, batch_size=batch_size))

    train_iterations = 1000
    val_iterations = 100
    early_stop = 1000

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
            torch.save(model.state_dict(), "models/model_epoch2.pth")

        save_checkpoint(model, optimizer, j, train_loss, val_loss, val_losses, checkpoint_path)

        # Progress feedback after each epoch
        log_training(epoch = j+1, train_loss = train_loss, train_iterations = train_iterations, 
                     val_loss=val_loss, val_iterations= val_iterations, saved=saved)
        tqdm.write(f'Epoch {j+1}: train_loss={train_loss/train_iterations:.4f}, val_loss={val_loss/val_iterations:.4f}, model_saved={saved}')

        if j > early_stop and all(
            [val_losses[j] >= vl for vl in val_losses[j-early_stop:j]]
        ):
            break
        elif math.isnan(val_losses[j]):
            break

        j += 1
