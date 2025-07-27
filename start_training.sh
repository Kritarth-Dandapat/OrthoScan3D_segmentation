#!/bin/bash

# Activate virtual environment
source braces_env/bin/activate

# Start training with Weights & Biases logging on 3 GPUs
python3 train_processed_data_wandb.py \
    --epochs 100 \
    --batch_size 1 \
    --devices 0 1 2 \
    --use_wandb \
    --wandb_project "teeth-segmentation-3d" \
    --experiment_name "full_dataset_training" \
    --experiment_version "v1" \
    --learning_rate 0.001 \
    --weight_decay 1e-4 