#!/usr/bin/env python3
"""
Monitor script for hyperparameter tuning experiments
Shows real-time progress and results
"""

import os
import json
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class ExperimentMonitor:
    def __init__(self, results_dir="hyperparameter_results"):
        self.results_dir = results_dir
        self.checkpoint_dir = os.path.join(results_dir, "checkpoints")
        
    def get_experiment_status(self):
        """Get status of all experiments"""
        status = {
            'total_experiments': 20,
            'completed': 0,
            'in_progress': 0,
            'failed': 0,
            'not_started': 0,
            'experiments': []
        }
        
        # Check for experiment result files
        result_files = glob.glob(os.path.join(self.results_dir, "experiment_*_results.json"))
        
        for i in range(20):
            exp_info = {
                'experiment_id': i,
                'status': 'not_started',
                'best_val_acc': None,
                'final_val_acc': None,
                'training_time': None,
                'epochs_trained': None,
                'config': None
            }
            
            # Check if experiment is completed
            result_file = os.path.join(self.results_dir, f"experiment_{i}_results.json")
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    exp_info.update({
                        'status': 'completed',
                        'best_val_acc': data.get('best_val_acc'),
                        'final_val_acc': data.get('final_val_acc'),
                        'training_time': data.get('training_time'),
                        'epochs_trained': data.get('epochs_trained'),
                        'config': data.get('config')
                    })
                    status['completed'] += 1
                except:
                    exp_info['status'] = 'failed'
                    status['failed'] += 1
            
            # Check if experiment is in progress (checkpoint exists but no results)
            elif os.path.exists(os.path.join(self.checkpoint_dir, f"experiment_{i}_best.pth")):
                exp_info['status'] = 'in_progress'
                status['in_progress'] += 1
            
            status['experiments'].append(exp_info)
        
        status['not_started'] = status['total_experiments'] - status['completed'] - status['in_progress'] - status['failed']
        return status
    
    def get_best_experiments(self, top_k=5):
        """Get top k experiments by validation accuracy"""
        status = self.get_experiment_status()
        completed_experiments = [exp for exp in status['experiments'] if exp['status'] == 'completed']
        
        # Sort by best validation accuracy
        completed_experiments.sort(key=lambda x: x['best_val_acc'] or 0, reverse=True)
        
        return completed_experiments[:top_k]
    
    def print_status(self):
        """Print current status"""
        status = self.get_experiment_status()
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING STATUS")
        print("="*80)
        print(f"Total Experiments: {status['total_experiments']}")
        print(f"Completed: {status['completed']}")
        print(f"In Progress: {status['in_progress']}")
        print(f"Failed: {status['failed']}")
        print(f"Not Started: {status['not_started']}")
        print(f"Progress: {status['completed']}/{status['total_experiments']} ({status['completed']/status['total_experiments']*100:.1f}%)")
        
        if status['completed'] > 0:
            print("\n" + "-"*80)
            print("TOP 5 EXPERIMENTS")
            print("-"*80)
            best_experiments = self.get_best_experiments(5)
            for i, exp in enumerate(best_experiments, 1):
                print(f"{i}. Experiment {exp['experiment_id']}: {exp['best_val_acc']:.2f}% "
                      f"(LR: {exp['config']['learning_rate']}, "
                      f"Optimizer: {exp['config']['optimizer']}, "
                      f"Time: {exp['training_time']/60:.1f}min)")
        
        print("\n" + "="*80)
    
    def generate_live_plot(self):
        """Generate a live plot of experiment results"""
        status = self.get_experiment_status()
        completed_experiments = [exp for exp in status['experiments'] if exp['status'] == 'completed']
        
        if len(completed_experiments) == 0:
            print("No completed experiments yet.")
            return
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Validation accuracy vs experiment ID
        plt.subplot(2, 3, 1)
        exp_ids = [exp['experiment_id'] for exp in completed_experiments]
        val_accs = [exp['best_val_acc'] for exp in completed_experiments]
        plt.scatter(exp_ids, val_accs, alpha=0.7, s=50)
        plt.xlabel('Experiment ID')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.title('Validation Accuracy by Experiment')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Learning rate vs accuracy
        plt.subplot(2, 3, 2)
        lrs = [exp['config']['learning_rate'] for exp in completed_experiments]
        plt.scatter(lrs, val_accs, alpha=0.7, s=50)
        plt.xlabel('Learning Rate')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.title('Learning Rate vs Accuracy')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Optimizer comparison
        plt.subplot(2, 3, 3)
        optimizers = [exp['config']['optimizer'] for exp in completed_experiments]
        optimizer_acc = {}
        for opt, acc in zip(optimizers, val_accs):
            if opt not in optimizer_acc:
                optimizer_acc[opt] = []
            optimizer_acc[opt].append(acc)
        
        opt_names = list(optimizer_acc.keys())
        opt_means = [np.mean(optimizer_acc[opt]) for opt in opt_names]
        plt.bar(opt_names, opt_means, alpha=0.7)
        plt.xlabel('Optimizer')
        plt.ylabel('Average Validation Accuracy (%)')
        plt.title('Average Accuracy by Optimizer')
        plt.xticks(rotation=45)
        
        # Plot 4: Training time vs accuracy
        plt.subplot(2, 3, 4)
        training_times = [exp['training_time']/60 for exp in completed_experiments]  # Convert to minutes
        plt.scatter(training_times, val_accs, alpha=0.7, s=50)
        plt.xlabel('Training Time (minutes)')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.title('Training Time vs Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Scheduler comparison
        plt.subplot(2, 3, 5)
        schedulers = [exp['config']['scheduler'] for exp in completed_experiments]
        scheduler_acc = {}
        for sch, acc in zip(schedulers, val_accs):
            if sch not in scheduler_acc:
                scheduler_acc[sch] = []
            scheduler_acc[sch].append(acc)
        
        sch_names = list(scheduler_acc.keys())
        sch_means = [np.mean(scheduler_acc[sch]) for sch in sch_names]
        plt.bar(sch_names, sch_means, alpha=0.7)
        plt.xlabel('Scheduler')
        plt.ylabel('Average Validation Accuracy (%)')
        plt.title('Average Accuracy by Scheduler')
        plt.xticks(rotation=45)
        
        # Plot 6: Progress over time
        plt.subplot(2, 3, 6)
        plt.bar(['Completed', 'In Progress', 'Failed', 'Not Started'], 
                [status['completed'], status['in_progress'], status['failed'], status['not_started']],
                alpha=0.7)
        plt.ylabel('Number of Experiments')
        plt.title('Experiment Status')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", "live_progress.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Live plot saved to: {os.path.join(self.results_dir, 'plots', 'live_progress.png')}")
    
    def monitor_continuously(self, interval=60):
        """Monitor experiments continuously"""
        print(f"Starting continuous monitoring (updates every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_status()
                self.generate_live_plot()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="Monitor hyperparameter tuning experiments")
    parser.add_argument("--results_dir", type=str, default="hyperparameter_results",
                       help="Directory containing experiment results")
    parser.add_argument("--continuous", action="store_true",
                       help="Monitor continuously with updates")
    parser.add_argument("--interval", type=int, default=60,
                       help="Update interval in seconds for continuous monitoring")
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor(args.results_dir)
    
    if args.continuous:
        monitor.monitor_continuously(args.interval)
    else:
        monitor.print_status()
        monitor.generate_live_plot()

if __name__ == "__main__":
    import numpy as np
    main() 