import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import random
import numpy as np
import json
import time
import glob
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from models.dilated_tooth_seg_network_tunable import TunableDilatedToothSegmentationNetwork
import wandb
from sklearn.model_selection import ParameterGrid
import logging

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

class HyperparameterTuner:
    def __init__(self, processed_dir="data/3dteethseg/processed", results_dir="hyperparameter_results", 
                 wandb_project="orthoscan3d-hyperparameter-tuning", wandb_entity=None):
        self.processed_dir = processed_dir
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Initialize wandb
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "total_experiments": 20,
                    "epochs_per_experiment": 100,
                    "device": str(self.device)
                }
            )
            self.wandb_available = True
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
            self.wandb_available = False
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        
        # Load datasets
        self.train_dataset = ProcessedTeethDataset(processed_dir, is_train=True, train_ratio=0.8)
        self.test_dataset = ProcessedTeethDataset(processed_dir, is_train=False, train_ratio=0.8)
        
        print(f"Dataset sizes: Train={len(self.train_dataset)}, Test={len(self.test_dataset)}")
        
        # Define hyperparameter search space
        self.hyperparameter_grid = {
            # Model architecture parameters
            'feature_dim': [16, 24, 32],
            'num_classes': [17],
            
            # Edge Graph Conv parameters
            'edge_conv_channels': [16, 24, 32],
            'edge_conv_k': [16, 32, 64],
            'edge_conv_hidden': [16, 24, 32],
            
            # Dilated Edge Graph Conv parameters
            'dilated_channels': [48, 60, 72],
            'dilated_k': [16, 32, 64],
            'dilated_hidden': [48, 60, 72],
            'dilation_k_values': [
                [100, 400, 800],
                [200, 900, 1800],
                [300, 1200, 2400]
            ],
            
            # Global hidden layer
            'global_hidden': [512, 1024, 2048],
            
            # Residual block parameters
            'res_channels': [256, 512],
            'res_hidden': [256, 512],
            
            # Training parameters
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'batch_size': [1],  # Keep batch size 1 due to memory constraints
            
            # Optimizer parameters
            'optimizer': ['adam', 'adamw', 'sgd'],
            'beta1': [0.9, 0.95],
            'beta2': [0.999, 0.99],
            
            # Learning rate scheduler
            'scheduler': ['step', 'cosine', 'plateau'],
            'scheduler_step_size': [30, 50, 70],
            'scheduler_gamma': [0.5, 0.7, 0.9],
            
            # Loss function
            'loss_function': ['cross_entropy', 'focal_loss'],
            'focal_alpha': [0.25, 0.5, 0.75],
            'focal_gamma': [1.0, 2.0, 3.0],
            
            # Regularization
            'dropout_rate': [0.0, 0.1, 0.2, 0.3],
            
            # Data augmentation
            'use_augmentation': [True, False],
            'rotation_range': [0, 5, 10],
            'translation_range': [0, 0.01, 0.02],
            'scale_range': [0.9, 1.0, 1.1],
        }
        
        # Generate 20 unique hyperparameter combinations
        self.experiments = self._generate_experiments(20)
        
    def _generate_experiments(self, num_experiments):
        """Generate diverse hyperparameter combinations for experiments."""
        experiments = []
        
        # Create a base configuration
        base_config = {
            'feature_dim': 24,
            'num_classes': 17,
            'edge_conv_channels': 24,
            'edge_conv_k': 32,
            'edge_conv_hidden': 24,
            'dilated_channels': 60,
            'dilated_k': 32,
            'dilated_hidden': 60,
            'dilation_k_values': [200, 900, 1800],
            'global_hidden': 1024,
            'res_channels': 512,
            'res_hidden': 512,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 1,
            'optimizer': 'adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'scheduler': 'step',
            'scheduler_step_size': 50,
            'scheduler_gamma': 0.5,
            'loss_function': 'cross_entropy',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'dropout_rate': 0.1,
            'use_augmentation': False,
            'rotation_range': 0,
            'translation_range': 0,
            'scale_range': 1.0,
        }
        
        # Add base configuration
        experiments.append(base_config.copy())
        
        # Generate variations
        for i in range(num_experiments - 1):
            config = base_config.copy()
            
            # Randomly modify parameters
            if i < 5:  # Focus on learning rate and optimizer
                config['learning_rate'] = random.choice(self.hyperparameter_grid['learning_rate'])
                config['optimizer'] = random.choice(self.hyperparameter_grid['optimizer'])
                config['weight_decay'] = random.choice(self.hyperparameter_grid['weight_decay'])
                
            elif i < 10:  # Focus on model architecture
                config['feature_dim'] = random.choice(self.hyperparameter_grid['feature_dim'])
                config['edge_conv_channels'] = random.choice(self.hyperparameter_grid['edge_conv_channels'])
                config['dilated_channels'] = random.choice(self.hyperparameter_grid['dilated_channels'])
                config['global_hidden'] = random.choice(self.hyperparameter_grid['global_hidden'])
                
            elif i < 15:  # Focus on regularization and loss
                config['dropout_rate'] = random.choice(self.hyperparameter_grid['dropout_rate'])
                config['loss_function'] = random.choice(self.hyperparameter_grid['loss_function'])
                config['scheduler'] = random.choice(self.hyperparameter_grid['scheduler'])
                
            else:  # Focus on data augmentation and edge parameters
                config['use_augmentation'] = random.choice(self.hyperparameter_grid['use_augmentation'])
                config['edge_conv_k'] = random.choice(self.hyperparameter_grid['edge_conv_k'])
                config['dilation_k_values'] = random.choice(self.hyperparameter_grid['dilation_k_values'])
            
            experiments.append(config)
        
        return experiments
    
    def create_model(self, config):
        """Create model with given configuration."""
        model = TunableDilatedToothSegmentationNetwork(
            num_classes=config['num_classes'],
            feature_dim=config['feature_dim'],
            edge_conv_channels=config['edge_conv_channels'],
            edge_conv_k=config['edge_conv_k'],
            edge_conv_hidden=config['edge_conv_hidden'],
            dilated_channels=config['dilated_channels'],
            dilated_k=config['dilated_k'],
            dilated_hidden=config['dilated_hidden'],
            dilation_k_values=config['dilation_k_values'],
            global_hidden=config['global_hidden'],
            res_channels=config['res_channels'],
            res_hidden=config['res_hidden'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        return model
    
    def create_optimizer(self, model, config):
        """Create optimizer with given configuration."""
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                betas=(config['beta1'], config['beta2']),
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                betas=(config['beta1'], config['beta2']),
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        return optimizer
    
    def create_scheduler(self, optimizer, config):
        """Create learning rate scheduler with given configuration."""
        if config['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['scheduler_step_size'],
                gamma=config['scheduler_gamma']
            )
        elif config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=1e-6
            )
        elif config['scheduler'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['scheduler_gamma'],
                patience=10,
                verbose=True
            )
        
        return scheduler
    
    def create_loss_function(self, config):
        """Create loss function with given configuration."""
        if config['loss_function'] == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif config['loss_function'] == 'focal_loss':
            # Implement focal loss
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def forward(self, inputs, targets):
                    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss.mean()
            
            return FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    
    def train_experiment(self, experiment_id, config, epochs=100):
        """Train a single experiment."""
        print(f"\n{'='*60}")
        print(f"Starting Experiment {experiment_id + 1}/20")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"{'='*60}")
        
        # Initialize wandb run for this experiment
        if self.wandb_available:
            try:
                run_name = f"experiment_{experiment_id:02d}"
                wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=run_name,
                    config=config,
                    reinit=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb run: {e}")
                self.wandb_available = False
        
        # Set random seeds for reproducibility
        torch.manual_seed(42 + experiment_id)
        random.seed(42 + experiment_id)
        np.random.seed(42 + experiment_id)
        
        # Create model, optimizer, scheduler, and loss function
        model = self.create_model(config)
        optimizer = self.create_optimizer(model, config)
        scheduler = self.create_scheduler(optimizer, config)
        criterion = self.create_loss_function(config)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Log initial configuration to wandb
        if self.wandb_available:
            try:
                wandb.log({
                    "experiment_id": experiment_id,
                    "config": config
                })
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")
                self.wandb_available = False
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 20
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (pos, x, y) in enumerate(train_loader):
                pos, x, y = pos.to(self.device), x.to(self.device), y.to(self.device)
                B, N, C = x.shape
                x = x.float()
                y = y.view(B, N).long()
                
                optimizer.zero_grad()
                pred = model(x, pos)
                pred = pred.transpose(2, 1)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(pred.data, 1)
                train_total += y.size(0) * y.size(1)
                train_correct += (predicted == y).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for pos, x, y in test_loader:
                    pos, x, y = pos.to(self.device), x.to(self.device), y.to(self.device)
                    B, N, C = x.shape
                    x = x.float()
                    y = y.view(B, N).long()
                    
                    pred = model(x, pos)
                    pred = pred.transpose(2, 1)
                    loss = criterion(pred, y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(pred.data, 1)
                    val_total += y.size(0) * y.size(1)
                    val_correct += (predicted == y).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss_avg = val_loss / len(test_loader)
            val_acc = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_avg)
            else:
                scheduler.step()
            
            # Store history
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(current_lr)
            
            # Log metrics to wandb
            if self.wandb_available:
                try:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss_avg,
                        "train_acc": train_acc,
                        "val_loss": val_loss_avg,
                        "val_acc": val_acc,
                        "learning_rate": current_lr,
                        "best_val_acc": best_val_acc
                    })
                except Exception as e:
                    print(f"Warning: Could not log metrics to wandb: {e}")
                    self.wandb_available = False
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint_path = os.path.join(self.results_dir, "checkpoints", f"experiment_{experiment_id}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, checkpoint_path)
                
                # Log best model to wandb
                if self.wandb_available:
                    try:
                        wandb.save(checkpoint_path)
                        wandb.run.summary["best_val_acc"] = best_val_acc
                        wandb.run.summary["best_epoch"] = epoch + 1
                    except Exception as e:
                        print(f"Warning: Could not save model to wandb: {e}")
                        self.wandb_available = False
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # Save final model
        final_checkpoint_path = os.path.join(self.results_dir, "checkpoints", f"experiment_{experiment_id}_final.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
            'history': history
        }, final_checkpoint_path)
        
        # Save experiment results
        experiment_results = {
            'experiment_id': experiment_id,
            'config': config,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'training_time': training_time,
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
        
        with open(os.path.join(self.results_dir, f"experiment_{experiment_id}_results.json"), 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        # Log final results to wandb
        if self.wandb_available:
            try:
                wandb.run.summary.update({
                    "final_val_acc": val_acc,
                    "training_time_minutes": training_time / 60,
                    "epochs_trained": len(history['train_loss']),
                    "early_stopped": patience_counter >= 20
                })
                
                # Save final model to wandb
                wandb.save(final_checkpoint_path)
                
                # Log training curves
                wandb.log({
                    "training_curves": wandb.plot.line_series(
                        xs=list(range(1, len(history['train_loss']) + 1)),
                        ys=[history['train_loss'], history['val_loss']],
                        keys=["train_loss", "val_loss"],
                        title="Training and Validation Loss",
                        xname="epoch"
                    )
                })
                
                wandb.log({
                    "accuracy_curves": wandb.plot.line_series(
                        xs=list(range(1, len(history['train_acc']) + 1)),
                        ys=[history['train_acc'], history['val_acc']],
                        keys=["train_acc", "val_acc"],
                        title="Training and Validation Accuracy",
                        xname="epoch"
                    )
                })
                
                # Finish wandb run
                wandb.finish()
            except Exception as e:
                print(f"Warning: Could not log final results to wandb: {e}")
                self.wandb_available = False
        
        print(f"Experiment {experiment_id + 1} completed. Best Val Acc: {best_val_acc:.2f}%")
        
        return experiment_results
    
    def run_all_experiments(self, epochs=100):
        """Run all experiments."""
        all_results = []
        
        # Log overall experiment start
        if self.wandb_available:
            try:
                wandb.log({"experiment_start": True})
            except Exception as e:
                print(f"Warning: Could not log experiment start to wandb: {e}")
                self.wandb_available = False
        
        for i, config in enumerate(self.experiments):
            try:
                result = self.train_experiment(i, config, epochs)
                all_results.append(result)
                
                # Log progress to main wandb run
                if self.wandb_available:
                    try:
                        wandb.log({
                            "experiments_completed": i + 1,
                            "total_experiments": 20,
                            "completion_percentage": (i + 1) / 20 * 100
                        })
                    except Exception as e:
                        print(f"Warning: Could not log progress to wandb: {e}")
                        self.wandb_available = False
                
            except Exception as e:
                print(f"Experiment {i + 1} failed: {e}")
                # Log failure to wandb
                if self.wandb_available:
                    try:
                        wandb.log({
                            "experiment_failed": i + 1,
                            "failure_reason": str(e)
                        })
                    except Exception as wandb_error:
                        print(f"Warning: Could not log failure to wandb: {wandb_error}")
                        self.wandb_available = False
                continue
        
        # Save all results
        with open(os.path.join(self.results_dir, "all_experiments_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        # Log final summary to wandb
        if self.wandb_available and all_results:
            try:
                best_result = max(all_results, key=lambda x: x['best_val_acc'])
                wandb.run.summary.update({
                    "best_experiment_id": best_result['experiment_id'],
                    "best_validation_accuracy": best_result['best_val_acc'],
                    "average_validation_accuracy": sum(r['best_val_acc'] for r in all_results) / len(all_results),
                    "total_training_time_hours": sum(r['training_time'] for r in all_results) / 3600
                })
                
                # Finish main wandb run
                wandb.finish()
            except Exception as e:
                print(f"Warning: Could not log final summary to wandb: {e}")
                self.wandb_available = False
        
        return all_results
    
    def generate_summary_report(self, results):
        """Generate a comprehensive summary report."""
        if not results:
            print("No results to analyze")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'experiment_id': result['experiment_id'],
                'best_val_acc': result['best_val_acc'],
                'final_val_acc': result['final_val_acc'],
                'training_time': result['training_time'],
                'epochs_trained': result['epochs_trained'],
                'learning_rate': result['config']['learning_rate'],
                'optimizer': result['config']['optimizer'],
                'weight_decay': result['config']['weight_decay'],
                'scheduler': result['config']['scheduler'],
                'loss_function': result['config']['loss_function'],
                'dropout_rate': result['config']['dropout_rate'],
                'feature_dim': result['config']['feature_dim'],
                'edge_conv_channels': result['config']['edge_conv_channels'],
                'dilated_channels': result['config']['dilated_channels'],
                'global_hidden': result['config']['global_hidden'],
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save summary
        df.to_csv(os.path.join(self.results_dir, "experiments_summary.csv"), index=False)
        
        # Generate plots
        self.generate_plots(df, results)
        
        # Log summary to wandb
        if self.wandb_available:
            try:
                wandb.log({
                    "experiments_summary": wandb.Table(dataframe=df),
                    "top_5_experiments": wandb.Table(
                        dataframe=df.nlargest(5, 'best_val_acc')[['experiment_id', 'best_val_acc', 'learning_rate', 'optimizer', 'scheduler']]
                    )
                })
            except Exception as e:
                print(f"Warning: Could not log summary to wandb: {e}")
                self.wandb_available = False
        
        # Print top 5 results
        print("\n" + "="*80)
        print("TOP 5 EXPERIMENTS BY VALIDATION ACCURACY")
        print("="*80)
        top_5 = df.nlargest(5, 'best_val_acc')
        for _, row in top_5.iterrows():
            print(f"Experiment {row['experiment_id']}: {row['best_val_acc']:.2f}% "
                  f"(LR: {row['learning_rate']}, Optimizer: {row['optimizer']}, "
                  f"Scheduler: {row['scheduler']}, Loss: {row['loss_function']})")
        
        # Print parameter importance analysis
        self.analyze_parameter_importance(df)
    
    def generate_plots(self, df, results):
        """Generate visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Validation accuracy distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.hist(df['best_val_acc'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Best Validation Accuracy')
        plt.xlabel('Validation Accuracy (%)')
        plt.ylabel('Frequency')
        
        # 2. Learning rate vs accuracy
        plt.subplot(2, 3, 2)
        plt.scatter(df['learning_rate'], df['best_val_acc'], alpha=0.6)
        plt.title('Learning Rate vs Best Validation Accuracy')
        plt.xlabel('Learning Rate')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.xscale('log')
        
        # 3. Optimizer comparison
        plt.subplot(2, 3, 3)
        optimizer_acc = df.groupby('optimizer')['best_val_acc'].mean()
        optimizer_acc.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Average Accuracy by Optimizer')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Best Validation Accuracy (%)')
        plt.xticks(rotation=45)
        
        # 4. Scheduler comparison
        plt.subplot(2, 3, 4)
        scheduler_acc = df.groupby('scheduler')['best_val_acc'].mean()
        scheduler_acc.plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('Average Accuracy by Scheduler')
        plt.xlabel('Scheduler')
        plt.ylabel('Average Best Validation Accuracy (%)')
        plt.xticks(rotation=45)
        
        # 5. Loss function comparison
        plt.subplot(2, 3, 5)
        loss_acc = df.groupby('loss_function')['best_val_acc'].mean()
        loss_acc.plot(kind='bar', color=['gold', 'lightcoral'])
        plt.title('Average Accuracy by Loss Function')
        plt.xlabel('Loss Function')
        plt.ylabel('Average Best Validation Accuracy (%)')
        plt.xticks(rotation=45)
        
        # 6. Training time vs accuracy
        plt.subplot(2, 3, 6)
        plt.scatter(df['training_time'], df['best_val_acc'], alpha=0.6)
        plt.title('Training Time vs Best Validation Accuracy')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Best Validation Accuracy (%)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, "plots", "experiments_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log plot to wandb
        if self.wandb_available:
            try:
                wandb.log({"experiments_analysis_plot": wandb.Image(plot_path)})
            except Exception as e:
                print(f"Warning: Could not log plot to wandb: {e}")
                self.wandb_available = False
        
        # Plot training curves for top 3 experiments
        plt.figure(figsize=(15, 5))
        top_3_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)[:3]
        
        for i, result in enumerate(top_3_results):
            plt.subplot(1, 3, i+1)
            history = result['history']
            plt.plot(history['train_acc'], label='Train Acc', alpha=0.7)
            plt.plot(history['val_acc'], label='Val Acc', alpha=0.7)
            plt.title(f'Experiment {result["experiment_id"]} (Best: {result["best_val_acc"]:.2f}%)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_plot_path = os.path.join(self.results_dir, "plots", "top_3_training_curves.png")
        plt.savefig(curves_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log training curves plot to wandb
        if self.wandb_available:
            try:
                wandb.log({"top_3_training_curves": wandb.Image(curves_plot_path)})
            except Exception as e:
                print(f"Warning: Could not log training curves to wandb: {e}")
                self.wandb_available = False
    
    def analyze_parameter_importance(self, df):
        """Analyze the importance of different parameters."""
        print("\n" + "="*80)
        print("PARAMETER IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Learning rate analysis
        lr_groups = df.groupby('learning_rate')['best_val_acc'].agg(['mean', 'std', 'count'])
        print(f"\nLearning Rate Analysis:")
        for lr, stats in lr_groups.iterrows():
            print(f"  LR {lr}: Mean={stats['mean']:.2f}±{stats['std']:.2f}% (n={stats['count']})")
        
        # Optimizer analysis
        opt_groups = df.groupby('optimizer')['best_val_acc'].agg(['mean', 'std', 'count'])
        print(f"\nOptimizer Analysis:")
        for opt, stats in opt_groups.iterrows():
            print(f"  {opt}: Mean={stats['mean']:.2f}±{stats['std']:.2f}% (n={stats['count']})")
        
        # Scheduler analysis
        sch_groups = df.groupby('scheduler')['best_val_acc'].agg(['mean', 'std', 'count'])
        print(f"\nScheduler Analysis:")
        for sch, stats in sch_groups.iterrows():
            print(f"  {sch}: Mean={stats['mean']:.2f}±{stats['std']:.2f}% (n={stats['count']})")
        
        # Loss function analysis
        loss_groups = df.groupby('loss_function')['best_val_acc'].agg(['mean', 'std', 'count'])
        print(f"\nLoss Function Analysis:")
        for loss, stats in loss_groups.iterrows():
            print(f"  {loss}: Mean={stats['mean']:.2f}±{stats['std']:.2f}% (n={stats['count']})")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for tooth segmentation model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per experiment")
    parser.add_argument("--processed_dir", type=str, default="data/3dteethseg/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--results_dir", type=str, default="hyperparameter_results",
                       help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="orthoscan3d-hyperparameter-tuning",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Weights & Biases entity/username")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This model requires GPU for PointNet2 operations.")
        return
    
    print("Starting hyperparameter tuning...")
    print(f"Will run 20 experiments with {args.epochs} epochs each")
    print(f"Results will be saved to: {args.results_dir}")
    print(f"WandB project: {args.wandb_project}")
    if args.wandb_entity:
        print(f"WandB entity: {args.wandb_entity}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    # Run all experiments
    results = tuner.run_all_experiments(epochs=args.epochs)
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Check the summary report and plots for detailed analysis.")
    if args.wandb_entity:
        print(f"View detailed results in Weights & Biases: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    else:
        print("Note: WandB logging was disabled. Results are saved locally only.")

if __name__ == "__main__":
    main() 