#!/usr/bin/env python3
"""
Batch processing script to run tooth segmentation inference on multiple PLY files.
"""

import os
import glob
import subprocess
import argparse
from pathlib import Path

def run_batch_inference(input_dir, output_dir, checkpoint_path, max_points=10000, use_cpu=False):
    """
    Run inference on all PLY files in the input directory.
    
    Args:
        input_dir: Directory containing PLY files
        output_dir: Directory to save results
        checkpoint_path: Path to the model checkpoint
        max_points: Maximum number of points to process
        use_cpu: Whether to force CPU usage
    """
    
    # Find all PLY files
    ply_files = glob.glob(os.path.join(input_dir, "*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in {input_dir}")
        return
    
    print(f"Found {len(ply_files)} PLY files to process:")
    for file in ply_files:
        print(f"  - {os.path.basename(file)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for i, ply_file in enumerate(ply_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(ply_files)}: {os.path.basename(ply_file)}")
        print(f"{'='*60}")
        
        # Create subdirectory for this file's results
        file_name = Path(ply_file).stem
        file_output_dir = os.path.join(output_dir, file_name)
        
        # Build command
        cmd = [
            "python", "run_experiment6_inference.py",
            ply_file,
            "--checkpoint", checkpoint_path,
            "--output_dir", file_output_dir,
            "--max_points", str(max_points)
        ]
        
        if use_cpu:
            cmd.append("--cpu")
        
        try:
            # Run the inference
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Successfully processed")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("✗ Error processing file:")
            print(e.stdout)
            print(e.stderr)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Run batch tooth segmentation inference on multiple PLY files")
    parser.add_argument("input_dir", type=str, help="Directory containing PLY files")
    parser.add_argument("--output_dir", type=str, default="batch_segmentation_results",
                       help="Output directory for results (default: batch_segmentation_results)")
    parser.add_argument("--checkpoint", type=str, 
                       default="hyperparameter_results/checkpoints/experiment_6_best.pth",
                       help="Path to model checkpoint (default: experiment_6_best.pth)")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum number of points to process (default: 10000)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' does not exist.")
        return
    
    # Run batch inference
    run_batch_inference(
        args.input_dir,
        args.output_dir,
        args.checkpoint,
        args.max_points,
        args.cpu
    )

if __name__ == "__main__":
    main() 