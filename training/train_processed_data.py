import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import random
import numpy as np
from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork
import glob
import argparse
from collections import OrderedDict

class ProcessedTeethDataset(Dataset):
    """Simple dataset class that directly loads processed .pt files."""
    
    def __init__(self, processed_dir, is_train=True, train_ratio=0.8):
        self.processed_dir = processed_dir
        self.is_train = is_train
        
        # Get all processed files
        all_files = glob.glob(os.path.join(processed_dir, "*.pt"))
        all_files.sort()  # For reproducible splits
        
        # Split into train/test
        split_idx = int(len(all_files) * train_ratio)
        if is_train:
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
        
        print(f"{'Training' if is_train else 'Testing'} dataset: {len(self.files)} files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        return data

def train_full_dataset():
    """Train the model on the full processed dataset."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This model requires GPU for PointNet2 operations.")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load datasets
    processed_dir = "data/3dteethseg/processed"
    if not os.path.exists(processed_dir):
        print("Processed data directory not found.")
        return
    
    print("Loading processed datasets...")
    train_dataset = ProcessedTeethDataset(processed_dir, is_train=True, train_ratio=0.8)
    test_dataset = ProcessedTeethDataset(processed_dir, is_train=False, train_ratio=0.8)
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Testing: {len(test_dataset)} samples")
    
    # Create model
    num_classes = 17
    feature_dim = 24
    model = DilatedToothSegmentationNetwork(num_classes=num_classes, feature_dim=feature_dim).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"Model created with {num_classes} classes and {feature_dim} feature dimensions")
    
    # Training parameters
    epochs = 50
    batch_size = 1  # Use batch size 1 due to memory constraints
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")
    
    print(f"Starting training for {epochs} epochs...")
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training...")
        
        for batch_idx, (pos, x, labels) in enumerate(train_loader):
            # Move data to GPU
            pos = pos.to(device)  # Shape: (batch_size, num_points, 3)
            x = x.to(device)  # Shape: (batch_size, num_points, feature_dim)
            labels = labels.to(device)  # Shape: (batch_size, num_points)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x, pos)  # Shape: (batch_size, num_points, num_classes)
            
            # Reshape for loss calculation
            output_reshaped = output.view(-1, num_classes)  # Shape: (batch_size * num_points, num_classes)
            labels_reshaped = labels.view(-1)  # Shape: (batch_size * num_points)
            
            # Calculate loss
            loss = criterion(output_reshaped, labels_reshaped)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(output_reshaped, dim=1)
            correct = (predictions == labels_reshaped).sum().item()
            total = labels_reshaped.size(0)
            
            train_loss += loss.item()
            train_correct += correct
            train_total += total
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("Validating...")
        with torch.no_grad():
            for pos, x, labels in test_loader:
                pos = pos.to(device)
                x = x.to(device)
                labels = labels.to(device)
                
                output = model(x, pos)
                output_reshaped = output.view(-1, num_classes)
                labels_reshaped = labels.view(-1)
                
                loss = criterion(output_reshaped, labels_reshaped)
                predictions = torch.argmax(output_reshaped, dim=1)
                correct = (predictions == labels_reshaped).sum().item()
                total = labels_reshaped.size(0)
                
                val_loss += loss.item()
                val_correct += correct
                val_total += total
        
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_acc > best_accuracy:
            best_accuracy = avg_val_acc
            torch.save(model.state_dict(), "best_model_full_dataset.pth")
            print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': avg_train_acc,
                'val_acc': avg_val_acc
            }, f"checkpoint_epoch_{epoch + 1}.pth")
            print(f"  Checkpoint saved for epoch {epoch + 1}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "final_model_full_dataset.pth")
    print("Final model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume training on the full processed dataset.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to train')
    args = parser.parse_args()

    def train_full_dataset_resumable(checkpoint_path=None, total_epochs=100):
        """Train the model on the full processed dataset, resuming from a checkpoint if provided."""
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA is not available. This model requires GPU for PointNet2 operations.")
            return
        
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Load datasets
        processed_dir = "data/3dteethseg/processed"
        if not os.path.exists(processed_dir):
            print("Processed data directory not found.")
            return
        
        print("Loading processed datasets...")
        train_dataset = ProcessedTeethDataset(processed_dir, is_train=True, train_ratio=0.8)
        test_dataset = ProcessedTeethDataset(processed_dir, is_train=False, train_ratio=0.8)
        
        print(f"Dataset sizes:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Testing: {len(test_dataset)} samples")
        
        # Create data loaders
        batch_size = 1  # You can change this if needed
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        num_classes = 17
        feature_dim = 24
        model = DilatedToothSegmentationNetwork(num_classes=num_classes, feature_dim=feature_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        best_accuracy = 0.0
        start_epoch = 0

        # Resume from checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if present
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' if it exists
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed at epoch {start_epoch}, best accuracy so far: {best_accuracy}")
        else:
            print("No checkpoint provided or file does not exist. Starting from scratch.")

        # Use multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        else:
            print("Using single GPU")

        print(f"Starting training for {total_epochs} epochs (from epoch {start_epoch+1})...")

        for epoch in range(start_epoch, total_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            print("Training...")
            
            for batch_idx, (pos, x, labels) in enumerate(train_loader):
                # Move data to GPU
                pos = pos.to(device)  # Shape: (batch_size, num_points, 3)
                x = x.to(device)  # Shape: (batch_size, num_points, feature_dim)
                labels = labels.to(device)  # Shape: (batch_size, num_points)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(x, pos)  # Shape: (batch_size, num_points, num_classes)
                
                # Reshape for loss calculation
                output_reshaped = output.view(-1, num_classes)  # Shape: (batch_size * num_points, num_classes)
                labels_reshaped = labels.view(-1)  # Shape: (batch_size * num_points)
                
                # Calculate loss
                loss = criterion(output_reshaped, labels_reshaped)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(output_reshaped, dim=1)
                correct = (predictions == labels_reshaped).sum().item()
                total = labels_reshaped.size(0)
                
                train_loss += loss.item()
                train_correct += correct
                train_total += total
                
                # Print progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            print("Validating...")
            with torch.no_grad():
                for pos, x, labels in test_loader:
                    pos = pos.to(device)
                    x = x.to(device)
                    labels = labels.to(device)
                    
                    output = model(x, pos)
                    output_reshaped = output.view(-1, num_classes)
                    labels_reshaped = labels.view(-1)
                    
                    loss = criterion(output_reshaped, labels_reshaped)
                    predictions = torch.argmax(output_reshaped, dim=1)
                    correct = (predictions == labels_reshaped).sum().item()
                    total = labels_reshaped.size(0)
                    
                    val_loss += loss.item()
                    val_correct += correct
                    val_total += total
            
            avg_val_loss = val_loss / len(test_loader)
            avg_val_acc = val_correct / val_total
            
            # Update learning rate
            scheduler.step()
            
            # Print epoch results
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if avg_val_acc > best_accuracy:
                best_accuracy = avg_val_acc
                torch.save(model.state_dict(), "best_model_full_dataset.pth")
                print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': avg_train_acc,
                    'val_acc': avg_val_acc
                }, f"checkpoint_epoch_{epoch + 1}.pth")
                print(f"  Checkpoint saved for epoch {epoch + 1}")
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        # Save final model
        torch.save(model.state_dict(), "final_model_full_dataset.pth")
        print("Final model saved!")

    train_full_dataset_resumable(args.checkpoint, args.epochs) 