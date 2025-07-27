# OrthoScan3D Hyperparameter Tuning System

This system provides comprehensive hyperparameter tuning for the OrthoScan3D tooth segmentation model with Weights & Biases (wandb) integration for experiment tracking and visualization.

## Features

- **20 Experiments**: Runs 20 different hyperparameter configurations
- **100 Epochs Each**: Each experiment runs for 100 epochs with early stopping
- **Comprehensive Tracking**: All experiments are logged to Weights & Biases
- **Real-time Monitoring**: Live progress tracking and visualization
- **Automatic Analysis**: Generates comprehensive reports and plots
- **Model Checkpointing**: Saves best models from each experiment

## Hyperparameters Being Tuned

### Model Architecture
- `feature_dim`: [16, 24, 32] - Input feature dimensionality
- `edge_conv_channels`: [16, 24, 32] - Edge convolution channels
- `edge_conv_k`: [16, 32, 64] - K-nearest neighbors for edge conv
- `dilated_channels`: [48, 60, 72] - Dilated convolution channels
- `global_hidden`: [512, 1024, 2048] - Global hidden layer size
- `dropout_rate`: [0.0, 0.1, 0.2, 0.3] - Dropout for regularization

### Training Parameters
- `learning_rate`: [0.0001, 0.0005, 0.001, 0.005] - Learning rate
- `optimizer`: [adam, adamw, sgd] - Optimizer choice
- `weight_decay`: [1e-5, 1e-4, 1e-3] - Weight decay for regularization
- `scheduler`: [step, cosine, plateau] - Learning rate scheduler
- `loss_function`: [cross_entropy, focal_loss] - Loss function choice

### Advanced Parameters
- `dilation_k_values`: Different dilation configurations
- `beta1`, `beta2`: Adam optimizer parameters
- `scheduler_step_size`, `scheduler_gamma`: Scheduler parameters
- `focal_alpha`, `focal_gamma`: Focal loss parameters

## Setup

### 1. Install Dependencies

```bash
# Install wandb if not already installed
pip install wandb

# Install other required packages
pip install -r requirements.txt
```

### 2. Setup Weights & Biases

```bash
# Login to wandb (you'll need to create an account at wandb.ai)
wandb login

# Verify login
wandb status
```

### 3. Prepare Data

Ensure your processed data is available at:
```
data/3dteethseg/processed/
```

## Usage

### Quick Start

Run the complete hyperparameter tuning:

```bash
# Make the script executable
chmod +x run_hyperparameter_tuning.sh

# Run the tuning
./run_hyperparameter_tuning.sh
```

### Manual Execution

```bash
python hyperparameter_tuning.py \
    --epochs 100 \
    --processed_dir "data/3dteethseg/processed" \
    --results_dir "hyperparameter_results" \
    --wandb_project "orthoscan3d-hyperparameter-tuning" \
    --wandb_entity "your-username"
```

### Monitor Progress

```bash
# Check current status
python monitor_experiments.py

# Monitor continuously (updates every 60 seconds)
python monitor_experiments.py --continuous --interval 60
```

## Output Structure

```
hyperparameter_results/
├── checkpoints/
│   ├── experiment_0_best.pth
│   ├── experiment_0_final.pth
│   └── ...
├── plots/
│   ├── experiments_analysis.png
│   ├── top_3_training_curves.png
│   └── live_progress.png
├── experiment_0_results.json
├── experiment_1_results.json
├── ...
├── experiments_summary.csv
└── all_experiments_results.json
```

## Weights & Biases Integration

### What Gets Logged

1. **Experiment Configuration**: All hyperparameters for each experiment
2. **Training Metrics**: Loss, accuracy, learning rate for each epoch
3. **Model Checkpoints**: Best models from each experiment
4. **Training Curves**: Loss and accuracy plots
5. **Analysis Plots**: Parameter importance and comparison charts
6. **Summary Tables**: Top performing experiments and configurations

### Dashboard Features

- **Real-time Tracking**: Live updates as experiments complete
- **Interactive Charts**: Click to explore different parameter relationships
- **Model Comparison**: Side-by-side comparison of different configurations
- **Resource Monitoring**: GPU usage and training time analysis
- **Artifact Management**: Easy access to saved models and results

### Accessing Results

1. **Web Dashboard**: Visit your wandb project page
2. **API Access**: Use wandb API to programmatically access results
3. **Local Files**: All results are also saved locally

## Monitoring and Analysis

### Real-time Monitoring

The monitoring script provides:
- Current experiment status
- Top 5 performing experiments
- Progress percentage
- Live plots and visualizations

### Analysis Features

1. **Parameter Importance**: Which hyperparameters matter most
2. **Best Configurations**: Top performing parameter combinations
3. **Training Efficiency**: Time vs accuracy trade-offs
4. **Failure Analysis**: Understanding failed experiments

## Example Results

After running the experiments, you'll get:

```
TOP 5 EXPERIMENTS BY VALIDATION ACCURACY
================================================================================
1. Experiment 12: 94.23% (LR: 0.001, Optimizer: adamw, Scheduler: cosine, Loss: cross_entropy)
2. Experiment 8: 93.87% (LR: 0.0005, Optimizer: adam, Scheduler: step, Loss: cross_entropy)
3. Experiment 15: 93.45% (LR: 0.001, Optimizer: adam, Scheduler: plateau, Loss: focal_loss)
4. Experiment 3: 92.98% (LR: 0.0005, Optimizer: adamw, Scheduler: cosine, Loss: cross_entropy)
5. Experiment 19: 92.76% (LR: 0.001, Optimizer: sgd, Scheduler: step, Loss: cross_entropy)
```

## Tips for Best Results

1. **Start with Default**: The system includes a baseline configuration
2. **Monitor Early**: Use the monitoring script to track progress
3. **Check WandB**: The web dashboard provides detailed insights
4. **Resource Management**: Ensure sufficient GPU memory
5. **Patience**: Full tuning takes several hours

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model complexity
2. **WandB Login Issues**: Run `wandb login` and check credentials
3. **Data Not Found**: Ensure processed data directory exists
4. **Experiment Failures**: Check logs for specific error messages

### Performance Optimization

1. **GPU Memory**: Monitor GPU usage and adjust model size
2. **Data Loading**: Use appropriate number of workers
3. **Checkpointing**: Regular saves prevent data loss
4. **Early Stopping**: Prevents overfitting and saves time

## Advanced Usage

### Custom Hyperparameter Ranges

Edit the `hyperparameter_grid` in `hyperparameter_tuning.py` to modify search spaces.

### Adding New Parameters

1. Add parameter to the grid
2. Update the model creation function
3. Modify the tunable model class
4. Update logging and analysis

### Parallel Execution

For faster execution, you can run multiple experiments in parallel:
- Use multiple GPUs
- Run on different machines
- Use distributed training

## Results Interpretation

### Key Metrics to Watch

1. **Validation Accuracy**: Primary performance metric
2. **Training Time**: Efficiency consideration
3. **Convergence**: How quickly models reach best performance
4. **Stability**: Consistency across different runs

### Parameter Insights

- **Learning Rate**: Most critical parameter
- **Optimizer**: AdamW often performs best
- **Architecture**: Larger models don't always help
- **Regularization**: Dropout can improve generalization

## Next Steps

After finding the best hyperparameters:

1. **Train Final Model**: Use best configuration for full training
2. **Ensemble Models**: Combine top performing models
3. **Further Tuning**: Fine-tune around best parameters
4. **Production Deployment**: Use best model for inference

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review wandb documentation
3. Examine experiment logs
4. Contact the development team

---

**Note**: This system is designed for research and development. For production use, ensure proper validation and testing procedures are followed. 