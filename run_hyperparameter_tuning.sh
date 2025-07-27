#!/bin/bash

# Hyperparameter Tuning Script for OrthoScan3D Segmentation
# This script runs 20 experiments with 100 epochs each

echo "=========================================="
echo "OrthoScan3D Hyperparameter Tuning"
echo "=========================================="
echo "Will run 20 experiments with 100 epochs each"
echo "Results will be saved to: hyperparameter_results/"
echo "=========================================="

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "Error: PyTorch not available or CUDA not detected"
    exit 1
fi

# Check if processed data exists
if [ ! -d "data/3dteethseg/processed" ]; then
    echo "Error: Processed data directory not found at data/3dteethseg/processed"
    echo "Please ensure the data is processed before running hyperparameter tuning"
    exit 1
fi

# Create results directory
mkdir -p hyperparameter_results
mkdir -p hyperparameter_results/checkpoints
mkdir -p hyperparameter_results/plots

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Warning: Weights & Biases (wandb) not installed. Installing..."
    pip install wandb
fi

# Check if user is logged into wandb
if ! wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "Warning: Not logged into Weights & Biases"
    echo "Please run 'wandb login' to enable experiment tracking"
    echo "You can still run the experiments without wandb, but tracking will be disabled"
    if [[ -t 0 ]]; then
        read -p "Continue without wandb? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Running in non-interactive mode, continuing without wandb..."
    fi
fi

# Run hyperparameter tuning
echo "Starting hyperparameter tuning..."
echo "This will take several hours depending on your hardware."
echo "You can monitor progress in the hyperparameter_results/ directory"
echo "Results will also be logged to Weights & Biases for detailed tracking"
echo ""

python hyperparameter_tuning.py \
    --epochs 100 \
    --processed_dir "data/3dteethseg/processed" \
    --results_dir "hyperparameter_results" \
    --wandb_project "orthoscan3d-hyperparameter-tuning"

echo ""
echo "=========================================="
echo "Hyperparameter tuning completed!"
echo "Check results in: hyperparameter_results/"
echo "==========================================" 