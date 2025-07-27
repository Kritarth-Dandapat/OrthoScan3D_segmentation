#!/usr/bin/env python3
"""
Script to run inference using experiment6.pth model for tooth segmentation on PLY files.
This script loads the trained model and performs segmentation on input 3D models.
"""

import torch
import trimesh
import numpy as np
import argparse
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the model
from models.dilated_tooth_seg_network_tunable import TunableDilatedToothSegmentationNetwork

# Tooth numbering utilities
from utils.teeth_numbering import color_mesh

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    # Initialize centroids with the first point
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    # Pick the first point randomly for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        # Calculate distances from the new centroid to all points
        dist = torch.sum((xyz - centroid)**2, -1)
        
        # Update distances: keep the minimum distance to any already-selected centroid
        distance = torch.min(distance, dist)
        
        # Find the point farthest from any already-selected centroid
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def load_experiment_config(experiment_id=6):
    """Load the configuration for the specified experiment."""
    config_path = f"hyperparameter_results/experiment_{experiment_id}_results.json"
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return {
            "feature_dim": 24,
            "num_classes": 17,
            "edge_conv_channels": 16,
            "edge_conv_k": 32,
            "edge_conv_hidden": 24,
            "dilated_channels": 72,
            "dilated_k": 32,
            "dilated_hidden": 60,
            "dilation_k_values": [200, 900, 1800],
            "global_hidden": 2048,
            "res_channels": 512,
            "res_hidden": 512
        }
    
    with open(config_path, 'r') as f:
        data = json.load(f)
        return data['config']

def process_ply_file(model_path, checkpoint_path, output_dir=None, max_points=10000, use_gpu=True, save_outputs=True):
    """
    Process a PLY file using the experiment6 model for tooth segmentation.
    
    Args:
        model_path: Path to the input PLY file
        checkpoint_path: Path to the model checkpoint (.pth file)
        output_dir: Directory to save results (optional if save_outputs=False)
        max_points: Maximum number of points to process
        use_gpu: Whether to use GPU for inference
        save_outputs: Whether to save output files (default: True)
    
    Returns:
        dict: Dictionary containing segmentation results:
            - 'predictions': numpy array of segmentation labels
            - 'vertices': numpy array of point coordinates
            - 'colors': numpy array of RGB colors for visualization
            - 'statistics': dict with segmentation statistics
            - 'output_files': list of saved file paths (if save_outputs=True)
    """
    
    # Load experiment configuration
    config = load_experiment_config(6)
    
    # Set device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the 3D model
    try:
        mesh = trimesh.load(model_path)
        print(f"Loaded mesh from '{model_path}' with {len(mesh.vertices)} vertices.")
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            print(f"Number of faces: {len(mesh.faces)}")
        else:
            print("The loaded model appears to be a point cloud (no faces).")
    except Exception as e:
        print(f"Error loading 3D model '{model_path}': {e}")
        return None
    
    # Extract vertices
    pos_np = mesh.vertices.astype(np.float32)
    current_num_points = pos_np.shape[0]
    
    # Downsample if necessary
    if current_num_points > max_points:
        print(f"Model has {current_num_points} points, exceeding max limit of {max_points}.")
        print(f"Downsampling to {max_points} points using Farthest Point Sampling...")
        
        pos_tensor_for_fps = torch.from_numpy(pos_np).unsqueeze(0).to(device)
        sampled_indices = farthest_point_sample(pos_tensor_for_fps, max_points)
        pos_np = pos_np[sampled_indices.cpu().numpy()[0]]
        print(f"Successfully downsampled to {pos_np.shape[0]} points.")
    else:
        print(f"Model has {current_num_points} points, which is within the max limit of {max_points}.")
    
    # Prepare input tensors
    num_points = pos_np.shape[0]
    pos = torch.from_numpy(pos_np).unsqueeze(0).to(device)  # Shape: (1, num_points, 3)
    
    # Generate features (using normals if available, otherwise dummy features)
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and mesh.vertex_normals.shape[0] == current_num_points:
        normals_np = mesh.vertex_normals.astype(np.float32)
        if current_num_points > max_points:
            normals_np = normals_np[sampled_indices.cpu().numpy()[0]]
        
        if normals_np.shape[1] < config['feature_dim']:
            padding_needed = config['feature_dim'] - normals_np.shape[1]
            x_np = np.pad(normals_np, ((0, 0), (0, padding_needed)), 'constant')
            print(f"Using vertex normals and padding to {config['feature_dim']} features.")
        else:
            x_np = normals_np[:, :config['feature_dim']]
            print(f"Using vertex normals, truncated to {config['feature_dim']} features.")
    else:
        print(f"No valid vertex normals found. Creating dummy features of shape ({num_points}, {config['feature_dim']}).")
        x_np = np.ones((num_points, config['feature_dim']), dtype=np.float32)
    
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # Shape: (1, num_points, feature_dim)
    
    print(f"Final shape of pos: {pos.shape}")
    print(f"Final shape of x: {x.shape}")
    
    # Create and load the model
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
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            # Lightning checkpoint with model_state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint with state_dict
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct model weights
            model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(x, pos)
    
    print(f"Output shape: {output.shape}")
    
    # Get predictions
    predictions = torch.argmax(output, dim=2).cpu().numpy()[0]  # Shape: (num_points,)
    
    # Generate colors for visualization
    colors = generate_colors_from_predictions(predictions)
    
    # Calculate statistics
    unique_labels, counts = np.unique(predictions, return_counts=True)
    statistics = {
        'total_points': len(predictions),
        'unique_classes': len(unique_labels),
        'class_distribution': {int(label): int(count) for label, count in zip(unique_labels, counts)},
        'class_percentages': {int(label): float((count / len(predictions)) * 100) for label, count in zip(unique_labels, counts)}
    }
    
    # Print segmentation statistics
    print(f"\nSegmentation Results:")
    print(f"Total points: {statistics['total_points']}")
    for label, count in zip(unique_labels, counts):
        percentage = statistics['class_percentages'][int(label)]
        print(f"Class {label}: {count} points ({percentage:.2f}%)")
    
    # Save results if requested
    output_files = []
    if save_outputs and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(model_path).stem
        
        # Save predictions as numpy array
        predictions_file = os.path.join(output_dir, f"{base_name}_predictions.npy")
        np.save(predictions_file, predictions)
        output_files.append(predictions_file)
        
        # Save colored point cloud
        pointcloud_file = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
        save_point_cloud_with_colors(pos_np, predictions, pointcloud_file)
        output_files.append(pointcloud_file)
        
        # Save colored mesh (only if we have faces and didn't downsample)
        if hasattr(mesh, 'faces') and mesh.faces is not None and current_num_points <= max_points:
            colored_mesh = color_mesh(mesh, predictions)
            mesh_file = os.path.join(output_dir, f"{base_name}_segmented.ply")
            colored_mesh.export(mesh_file)
            output_files.append(mesh_file)
            print(f"Saved colored mesh to {mesh_file}")
        else:
            print("Skipping mesh export due to downsampling or no faces (point cloud)")
    
    # Return results
    results = {
        'predictions': predictions,
        'vertices': pos_np,
        'colors': colors,
        'statistics': statistics,
        'output_files': output_files if save_outputs else []
    }
    
    if save_outputs:
        print(f"\nSegmentation completed successfully!")
        print(f"Results saved to: {output_dir}")
    
    return results

def save_point_cloud_with_colors(vertices, labels, output_path):
    """Save point cloud with colors based on segmentation labels."""
    # Create a simple color mapping for 17 classes
    colors = plt.cm.tab20(np.linspace(0, 1, 17))  # 17 distinct colors
    
    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        for vertex, label in zip(vertices, labels):
            color = colors[label % 17]  # Use modulo to handle any label values
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {r} {g} {b}\n")
    
    print(f"Saved colored point cloud to {output_path}")

def generate_colors_from_predictions(predictions):
    """Generate RGB colors for each prediction label."""
    # Create a simple color mapping for 17 classes
    colors = plt.cm.tab20(np.linspace(0, 1, 17))  # 17 distinct colors
    
    # Map predictions to colors
    color_array = np.zeros((len(predictions), 3))
    for i, pred in enumerate(predictions):
        color = colors[pred % 17]  # Use modulo to handle any label values
        color_array[i] = color[:3]  # RGB only
    
    return color_array

def segment_ply_file(ply_file_path, checkpoint_path="hyperparameter_results/checkpoints/experiment_6_best.pth", max_points=10000, use_gpu=True):
    """
    Simple function to segment a PLY file and return results without saving files.
    
    Args:
        ply_file_path: Path to the input PLY file
        checkpoint_path: Path to the model checkpoint (default: experiment_6_best.pth)
        max_points: Maximum number of points to process (default: 10000)
        use_gpu: Whether to use GPU (default: True)
    
    Returns:
        dict: Dictionary containing:
            - 'predictions': numpy array of segmentation labels (0-16)
            - 'vertices': numpy array of point coordinates
            - 'colors': numpy array of RGB colors for visualization
            - 'statistics': dict with segmentation statistics
            - 'class_names': dict mapping class IDs to tooth names
    
    Example:
        results = segment_ply_file("my_teeth.ply")
        print(f"Found {results['statistics']['unique_classes']} tooth types")
        print(f"Predictions shape: {results['predictions'].shape}")
    """
    
    # Tooth class names mapping
    class_names = {
        0: 'gum',
        1: 'l_central_incisor',
        2: 'l_lateral_incisor', 
        3: 'l_canine',
        4: 'l_1_st_premolar',
        5: 'l_2_nd_premolar',
        6: 'l_1_st_molar',
        7: 'l_2_nd_molar',
        8: 'l_3_rd_molar',
        9: 'r_central_incisor',
        10: 'r_lateral_incisor',
        11: 'r_canine',
        12: 'r_1_st_premolar',
        13: 'r_2_nd_premolar',
        14: 'r_1_st_molar',
        15: 'r_2_nd_molar',
        16: 'r_3_rd_molar'
    }
    
    # Perform segmentation without saving files
    results = process_ply_file(
        model_path=ply_file_path,
        checkpoint_path=checkpoint_path,
        output_dir=None,
        max_points=max_points,
        use_gpu=use_gpu,
        save_outputs=False
    )
    
    if results is not None:
        results['class_names'] = class_names
        return results
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Run tooth segmentation inference using experiment6 model")
    parser.add_argument("input_file", type=str, help="Path to input PLY file")
    parser.add_argument("--checkpoint", type=str, 
                       default="hyperparameter_results/checkpoints/experiment_6_best.pth",
                       help="Path to model checkpoint (default: experiment_6_best.pth)")
    parser.add_argument("--output_dir", type=str, default="segmentation_results",
                       help="Output directory for results (default: segmentation_results)")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum number of points to process (default: 10000)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' does not exist.")
        return
    
    # Run inference
    use_gpu = not args.cpu and torch.cuda.is_available()
    if not use_gpu and not args.cpu:
        print("Warning: CUDA not available, using CPU.")
    
    results = process_ply_file(
        args.input_file, 
        args.checkpoint, 
        args.output_dir, 
        args.max_points, 
        use_gpu
    )
    
    if results is not None:
        print(f"\nSegmentation completed successfully!")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 