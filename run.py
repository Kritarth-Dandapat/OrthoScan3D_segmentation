import torch
import trimesh
import numpy as np
import argparse
import os

# The correct import statement for your model
from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork

# --- Farthest Point Sampling (FPS) Function ---
# This is a common implementation for GPU-accelerated FPS.
# You might find this in various PointNet/PointNet++ implementations.
# Make sure you have PyTorch installed with CUDA support.

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
    
    return centroids # Returns indices of the sampled points

# --- End of FPS Function ---

def process_model(model_path, num_classes, feature_dim, max_points_limit):
    """
    Loads a 3D model, processes it, and passes it through the segmentation network.
    """
    # --- 1. Load your 3D model ---
    try:
        mesh = trimesh.load(model_path)
        print(f"Loaded mesh from '{model_path}' with {len(mesh.vertices)} vertices.")
        if hasattr(mesh, 'faces') and mesh.faces is not None:
             print(f"Number of faces: {len(mesh.faces)}")
        else:
            print("The loaded model appears to be a point cloud (no faces).")

    except FileNotFoundError:
        print(f"Error: 3D model file not found at '{model_path}'.")
        return
    except Exception as e:
        print(f"Error loading 3D model '{model_path}': {e}")
        return

    pos_np = mesh.vertices.astype(np.float32)
    current_num_points = pos_np.shape[0]

    # --- Downsample if points exceed the limit ---
    if current_num_points > max_points_limit:
        print(f"Model has {current_num_points} points, exceeding max limit of {max_points_limit}.")
        print(f"Downsampling to {max_points_limit} points using Farthest Point Sampling...")

        # Convert to PyTorch tensor, add batch dimension, and move to CUDA for FPS
        pos_tensor_for_fps = torch.from_numpy(pos_np).unsqueeze(0).cuda() # Shape: (1, current_num_points, 3)
        
        # Get indices of sampled points
        sampled_indices = farthest_point_sample(pos_tensor_for_fps, max_points_limit)
        
        # Use sampled indices to select points from original numpy array
        # Note: sampled_indices is (1, npoint), so we take [0] to get the actual indices
        pos_np = pos_np[sampled_indices.cpu().numpy()[0]]
        
        print(f"Successfully downsampled to {pos_np.shape[0]} points.")
    else:
        print(f"Model has {current_num_points} points, which is within the max limit of {max_points_limit}.")


    # --- 2. Extract pos (xyz coordinates) after potential downsampling ---
    num_points = pos_np.shape[0]

    # Convert to a PyTorch tensor, add batch dimension, and move to CUDA
    pos = torch.from_numpy(pos_np).unsqueeze(0).cuda() # Shape: (1, num_points, 3)

    # --- 3. Generate or extract x (features) ---
    x_np = None

    # If the mesh had vertex normals, we need to sample them too based on the same indices.
    # If normals are not available or are generated, this part remains similar.
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and mesh.vertex_normals.shape[0] == current_num_points:
        normals_np = mesh.vertex_normals.astype(np.float32)
        if current_num_points > max_points_limit: # If downsampling happened
            normals_np = normals_np[sampled_indices.cpu().numpy()[0]] # Apply same indices
        
        if normals_np.shape[1] < feature_dim:
            padding_needed = feature_dim - normals_np.shape[1]
            x_np = np.pad(normals_np, ((0, 0), (0, padding_needed)), 'constant')
            print(f"Using vertex normals and padding to {feature_dim} features.")
        else:
            x_np = normals_np[:, :feature_dim] # Truncate if normals somehow exceed feature_dim
            print(f"Using vertex normals, truncated to {feature_dim} features.")
    else:
        print(f"No valid vertex normals found or shape mismatch. Creating dummy features of shape ({num_points}, {feature_dim}).")
        x_np = np.ones((num_points, feature_dim), dtype=np.float32) # Or np.random.rand(...)

    x = torch.from_numpy(x_np).unsqueeze(0).cuda() # Shape: (1, num_points, feature_dim)

    # --- Verify shapes before passing to model ---
    print(f"Final shape of pos: {pos.shape}")
    print(f"Final shape of x: {x.shape}")

    # Create the model using the actual imported class
    model = DilatedToothSegmentationNetwork(num_classes=num_classes, feature_dim=feature_dim).cuda()

    # --- Load pre-trained weights (if available) ---
    # Uncomment and provide the path to your .pth file if you have trained weights
    # try:
    #     # Example path: os.path.join(os.path.dirname(__file__), 'model_weights', 'dilated_tooth_seg.pth')
    #     model_weights_path = "path/to/your/model_weights.pth"
    #     model.load_state_dict(torch.load(model_weights_path))
    #     print(f"Loaded model weights from {model_weights_path}")
    # except FileNotFoundError:
    #     print("Warning: Model weights not found. Using randomly initialized weights.")
    # except Exception as e:
    #     print(f"Error loading model weights: {e}")

    # Pass the actual 3D model data to the network
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        out = model(x, pos)
    print(f"Output shape: {out.shape}") # Shape: (batch_size, num_points, num_classes)
    print(f"First 5 output values for the first point: {out[0, 0, :5]}") # Example: print first 5 output values for first point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a 3D model for tooth segmentation with point limit.")
    parser.add_argument("model_path", type=str,
                        help="Path to the 3D model file (e.g., .obj, .ply, .stl).")
    parser.add_argument("--num_classes", type=int, default=17,
                        help="Number of output classes for segmentation (default: 17).")
    parser.add_argument("--feature_dim", type=int, default=24,
                        help="Dimensionality of point features (default: 24).")
    parser.add_argument("--max_points", type=int, default=10000,
                        help="Maximum number of points to process. Models with more points will be downsampled (default: 10000).")

    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a CUDA-enabled GPU for FPS and model inference.")
        print("Please ensure your GPU drivers are installed and PyTorch is configured for CUDA.")
        exit()

    # Verify model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: The specified model path '{args.model_path}' does not exist.")
        exit()

    process_model(args.model_path, args.num_classes, args.feature_dim, args.max_points)
