#!/bin/bash

# Activate the virtual environment
source braces_env/bin/activate

# Start training in the background, log output
nohup python3 train_network_wandb.py --epochs 100 --train_batch_size 2 --devices 0 > train_output.log 2>&1 &

echo "Training started in background. Check train_output.log for progress." 