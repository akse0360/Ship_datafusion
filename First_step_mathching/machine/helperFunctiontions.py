import os
import torch
import csv

class Checkpoint:
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_losses, train_corrects, val_corrects, file_name="models/checkpoint.pth"):
        if not os.path.exists("models"):
            os.makedirs("models")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_losses': val_losses,
            'train_corrects': train_corrects,
            'val_corrects': val_corrects
        }
        try:
            torch.save(checkpoint, file_name)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    @staticmethod
    def load_checkpoint(model, optimizer, file_name="models/checkpoint.pth"):
        if os.path.isfile(file_name):
            try:
                checkpoint = torch.load(file_name, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                train_loss = checkpoint['train_loss']
                val_loss = checkpoint['val_loss']
                val_losses = checkpoint['val_losses']
                train_corrects = checkpoint.get('train_corrects', [])
                val_corrects = checkpoint.get('val_corrects', [])
                return epoch, train_loss, val_loss, val_losses, train_corrects, val_corrects
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return 0, 0, 0, [], [], []
        else:
            print(f"No checkpoint found at {file_name}")
            return 0, 0, 0, [], [], []
    
    @staticmethod
    def log_training(epoch, train_loss, val_loss, avg_train_correct, avg_val_correct, saved, file_name='training_log.csv'):
        # Check if file exists, if not, create it and write the header
        file_exists = os.path.isfile(file_name)
        
        # Calculate the losses
        train_loss_value = train_loss
        val_loss_value = val_loss
        
        # Write the data to CSV
        try:
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    # Write header if file does not exist
                    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Correct', 'Val Correct', 'Model Saved'])
                
                # Append the new data
                writer.writerow([f'Epoch {epoch + 1}', f'{train_loss_value:.4f}', f'{val_loss_value:.4f}', f'{avg_train_correct:.4f}', f'{avg_val_correct:.4f}', saved])
        except Exception as e:
            print(f"Error logging training data: {e}")
