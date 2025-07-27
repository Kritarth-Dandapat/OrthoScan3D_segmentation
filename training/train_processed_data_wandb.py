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
import wandb
import argparse
from datetime import datetime

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

def train_full_dataset_wandb(args):
    """Train the model on the full processed dataset with Weights & Biases logging."""
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.experiment_name}_{args.experiment_version}",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_classes": 17,
                "feature_dim": 24,
                "devices": args.devices,
                "train_ratio": 0.8
            }
        )
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"Model created with {num_classes} classes and {feature_dim} feature dimensions")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU")
    
    print(f"Starting training for {args.epochs} epochs...")
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
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
            
            # Log batch metrics to wandb
            if args.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": correct / total,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "batch": batch_idx
                })
            
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
        
        # Log epoch metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Print epoch results
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_acc > best_accuracy:
            best_accuracy = avg_val_acc
            model_filename = f"best_model_full_dataset_{args.experiment_name}_{args.experiment_version}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")
            
            # Log best model to wandb
            if args.use_wandb:
                wandb.save(model_filename)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch + 1}_{args.experiment_name}_{args.experiment_version}.pth"
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
            }, checkpoint_filename)
            print(f"  Checkpoint saved for epoch {epoch + 1}")
            
            # Log checkpoint to wandb
            if args.use_wandb:
                wandb.save(checkpoint_filename)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Save final model
    final_model_filename = f"final_model_full_dataset_{args.experiment_name}_{args.experiment_version}.pth"
    torch.save(model.state_dict(), final_model_filename)
    print("Final model saved!")
    
    # Log final model to wandb
    if args.use_wandb:
        wandb.save(final_model_filename)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on processed data with Weights & Biases logging')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1], help='GPU devices to use')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='teeth-segmentation-3d', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/username')
    parser.add_argument('--experiment_name', type=str, default='full_dataset_training', help='Experiment name')
    parser.add_argument('--experiment_version', type=str, default='v2', help='Experiment version')
    
    args = parser.parse_args()
    
    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.devices))
    
    # Memory optimization settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train_full_dataset_wandb(args) 
