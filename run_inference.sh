#!/bin/bash

# Script to run tooth segmentation inference using experiment6 model

# Activate virtual environment if it exists
if [ -d "orthoscan_env" ]; then
    echo "Activating virtual environment..."
    source orthoscan_env/bin/activate
fi

echo "=========================================="
echo "OrthoScan3D Tooth Segmentation Inference"
echo "=========================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA is available"
    USE_CPU=""
else
    echo "⚠ CUDA not available, will use CPU"
    USE_CPU="--cpu"
fi

# Default checkpoint path
CHECKPOINT="hyperparameter_results/checkpoints/experiment_6_best.pth"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la hyperparameter_results/checkpoints/
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Function to run inference on a single file
run_single_inference() {
    local input_file=$1
    local output_dir="segmentation_results/$(basename "$input_file" .ply)"
    
    echo ""
    echo "Processing: $input_file"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    python run_experiment6_inference.py "$input_file" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$output_dir" \
        --max_points 10000 \
        $USE_CPU
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $input_file"
    else
        echo "✗ Failed to process $input_file"
    fi
}

# Function to run batch inference
run_batch_inference() {
    local input_dir=$1
    local output_dir="batch_segmentation_results"
    
    echo ""
    echo "Running batch inference on directory: $input_dir"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    python batch_inference.py "$input_dir" \
        --output_dir "$output_dir" \
        --checkpoint "$CHECKPOINT" \
        --max_points 10000 \
        $USE_CPU
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed batch inference"
    else
        echo "✗ Batch inference failed"
    fi
}

# Main execution
if [ $# -eq 0 ]; then
    # No arguments provided, run on the default file
    if [ -f "00OMSZGW_lower.ply" ]; then
        echo "Running inference on default file: 00OMSZGW_lower.ply"
        run_single_inference "00OMSZGW_lower.ply"
    else
        echo "Error: No input file specified and 00OMSZGW_lower.ply not found"
        echo ""
        echo "Usage:"
        echo "  $0 <input_file.ply>                    # Run on single file"
        echo "  $0 --batch <input_directory>           # Run on all PLY files in directory"
        echo "  $0 --checkpoint <path> <input_file.ply> # Use specific checkpoint"
        echo ""
        echo "Examples:"
        echo "  $0 00OMSZGW_lower.ply"
        echo "  $0 --batch ./data"
        echo "  $0 --checkpoint hyperparameter_results/checkpoints/experiment_6_final.pth 00OMSZGW_lower.ply"
        exit 1
    fi
elif [ "$1" = "--batch" ]; then
    # Batch mode
    if [ -z "$2" ]; then
        echo "Error: Please specify input directory for batch mode"
        exit 1
    fi
    run_batch_inference "$2"
elif [ "$1" = "--checkpoint" ]; then
    # Custom checkpoint
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Error: Please specify checkpoint path and input file"
        exit 1
    fi
    CHECKPOINT="$2"
    run_single_inference "$3"
else
    # Single file mode
    run_single_inference "$1"
fi

echo ""
echo "=========================================="
echo "Inference completed!"
echo "Check the output directories for results."
echo "==========================================" 