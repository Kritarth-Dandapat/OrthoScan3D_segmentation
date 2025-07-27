#!/usr/bin/env python3
"""
Simple test script to demonstrate the segmentation function.
"""

from run_experiment6_inference import segment_ply_file

def test_segmentation():
    """Test the segmentation function on the sample PLY file."""
    
    # Perform segmentation
    print("Starting tooth segmentation...")
    results = segment_ply_file("00OMSZGW_lower.ply")
    
    if results is None:
        print("Segmentation failed!")
        return
    
    # Display results
    print("\n" + "="*50)
    print("SEGMENTATION RESULTS")
    print("="*50)
    
    print(f"Total points processed: {results['statistics']['total_points']}")
    print(f"Number of unique tooth classes: {results['statistics']['unique_classes']}")
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Vertices shape: {results['vertices'].shape}")
    print(f"Colors shape: {results['colors'].shape}")
    
    print("\nClass Distribution:")
    for class_id, count in results['statistics']['class_distribution'].items():
        percentage = results['statistics']['class_percentages'][class_id]
        class_name = results['class_names'].get(class_id, f"Unknown_{class_id}")
        print(f"  {class_name} (Class {class_id}): {count} points ({percentage:.1f}%)")
    
    # Example: Find the most common tooth type
    most_common_class = max(results['statistics']['class_distribution'].items(), key=lambda x: x[1])
    most_common_name = results['class_names'].get(most_common_class[0], f"Unknown_{most_common_class[0]}")
    print(f"\nMost common tooth type: {most_common_name} ({most_common_class[1]} points)")
    
    # Example: Count non-gum points
    non_gum_points = sum(count for class_id, count in results['statistics']['class_distribution'].items() if class_id != 0)
    print(f"Total tooth points (excluding gum): {non_gum_points}")
    
    return results

if __name__ == "__main__":
    results = test_segmentation() 