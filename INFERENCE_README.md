# Tooth Segmentation Inference with Experiment6 Model

This directory contains scripts to run tooth segmentation inference using the trained experiment6 model on PLY files.

## Files Overview

- `run_experiment6_inference.py` - Main inference script for single PLY files
- `batch_inference.py` - Batch processing script for multiple PLY files
- `run_inference.sh` - Convenient shell script wrapper
- `INFERENCE_README.md` - This documentation file

## Quick Start

### Option 1: Using the Shell Script (Recommended)

The easiest way to run inference is using the provided shell script:

```bash
# Run on the default file (00OMSZGW_lower.ply)
./run_inference.sh

# Run on a specific file
./run_inference.sh your_file.ply

# Run on all PLY files in a directory
./run_inference.sh --batch ./data_directory

# Use a different checkpoint
./run_inference.sh --checkpoint hyperparameter_results/checkpoints/experiment_6_final.pth your_file.ply
```

### Option 2: Using Python Scripts Directly

#### Single File Inference

```bash
python run_experiment6_inference.py 00OMSZGW_lower.ply \
    --checkpoint hyperparameter_results/checkpoints/experiment_6_best.pth \
    --output_dir segmentation_results \
    --max_points 10000
```

#### Batch Processing

```bash
python batch_inference.py ./data_directory \
    --output_dir batch_segmentation_results \
    --checkpoint hyperparameter_results/checkpoints/experiment_6_best.pth \
    --max_points 10000
```

## Available Checkpoints

The following experiment6 checkpoints are available:

- `hyperparameter_results/checkpoints/experiment_6_best.pth` - Best model during training
- `hyperparameter_results/checkpoints/experiment_6_final.pth` - Final model after training

## Model Configuration

The experiment6 model uses the following configuration:
- **Number of classes**: 17 (different tooth types)
- **Feature dimension**: 24
- **Maximum points**: 10,000 (configurable)
- **Architecture**: Dilated Edge Graph Convolutional Network

## Output Files

For each processed PLY file, the following outputs are generated:

1. **`{filename}_predictions.npy`** - Raw segmentation predictions as numpy array
2. **`{filename}_segmented.ply`** - Colored mesh with segmentation results
3. **`{filename}_pointcloud.ply`** - Colored point cloud with segmentation results

## Segmentation Classes

The model predicts 17 different classes representing different tooth types:
- Class 0-15: Different tooth types (incisors, canines, premolars, molars)
- Class 16: Background/non-tooth regions

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install trimesh numpy matplotlib
pip install lightning torchmetrics
```

## GPU vs CPU

The scripts automatically detect if CUDA is available and use GPU if possible. To force CPU usage:

```bash
python run_experiment6_inference.py your_file.ply --cpu
```

## Performance Notes

- **GPU**: Recommended for faster processing
- **CPU**: Slower but works on any machine
- **Memory**: Large PLY files (>10,000 points) will be downsampled automatically
- **Processing time**: Depends on file size and hardware (typically 30 seconds to 5 minutes per file)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--max_points` parameter
2. **File not found**: Check file paths and ensure PLY files exist
3. **Import errors**: Make sure all dependencies are installed
4. **Model loading errors**: Verify checkpoint file exists and is not corrupted

### Error Messages

- `"CUDA is not available"`: The script will automatically fall back to CPU
- `"Model has X points, exceeding max limit"`: The file will be automatically downsampled
- `"No valid vertex normals found"`: The script will use dummy features (this is normal)

## Example Usage

Here's a complete example workflow:

```bash
# 1. Check available files
ls *.ply

# 2. Run inference on a single file
./run_inference.sh 00OMSZGW_lower.ply

# 3. Check results
ls segmentation_results/00OMSZGW_lower/

# 4. Run batch processing on multiple files
mkdir test_files
cp *.ply test_files/
./run_inference.sh --batch test_files

# 5. View batch results
ls batch_segmentation_results/
```

## Visualization

The output PLY files can be opened in various 3D visualization software:
- **MeshLab**: Free, open-source 3D mesh processing software
- **CloudCompare**: Point cloud and mesh processing
- **Blender**: 3D modeling and visualization
- **ParaView**: Scientific visualization

Each tooth segment will be colored differently to distinguish between different tooth types.

## Advanced Usage

### Custom Model Configuration

You can modify the model parameters by editing the `load_experiment_config()` function in `run_experiment6_inference.py`.

### Integration with Other Tools

The prediction arrays (`.npy` files) can be easily loaded in Python for further analysis:

```python
import numpy as np
predictions = np.load('segmentation_results/00OMSZGW_lower/00OMSZGW_lower_predictions.npy')
print(f"Predicted {len(np.unique(predictions))} different tooth types")
```

### Batch Processing with Custom Parameters

```bash
python batch_inference.py ./data \
    --output_dir custom_results \
    --checkpoint hyperparameter_results/checkpoints/experiment_6_final.pth \
    --max_points 5000 \
    --cpu
```

## Support

If you encounter any issues, please check:
1. All dependencies are installed correctly
2. File paths are correct
3. PLY files are valid and readable
4. Sufficient disk space for output files 