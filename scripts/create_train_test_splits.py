#!/usr/bin/env python3
"""
Script to create train/test split files for the 3DTeethSeg dataset.
"""

import os
import random
from pathlib import Path

def create_train_test_splits():
    """Create train/test split files for upper and lower jaws."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get all patient IDs from upper directory (should be same as lower)
    upper_dir = Path("data/3dteethseg/raw/upper")
    patient_ids = [d.name for d in upper_dir.iterdir() if d.is_dir()]
    
    print(f"Total patients found: {len(patient_ids)}")
    
    # Shuffle the patient IDs
    random.shuffle(patient_ids)
    
    # Split into train/test (80% train, 20% test)
    split_idx = int(0.8 * len(patient_ids))
    train_patients = patient_ids[:split_idx]
    test_patients = patient_ids[split_idx:]
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Create the split files
    splits_dir = Path("data/3dteethseg/raw")
    
    # Training splits
    with open(splits_dir / "training_upper.txt", "w") as f:
        for patient in train_patients:
            f.write(f"{patient}\n")
    
    with open(splits_dir / "training_lower.txt", "w") as f:
        for patient in train_patients:
            f.write(f"{patient}\n")
    
    # Test splits
    with open(splits_dir / "test_upper.txt", "w") as f:
        for patient in test_patients:
            f.write(f"{patient}\n")
    
    with open(splits_dir / "test_lower.txt", "w") as f:
        for patient in test_patients:
            f.write(f"{patient}\n")
    
    print("Train/test split files created successfully!")
    print(f"Files created:")
    print(f"  - {splits_dir / 'training_upper.txt'}")
    print(f"  - {splits_dir / 'training_lower.txt'}")
    print(f"  - {splits_dir / 'test_upper.txt'}")
    print(f"  - {splits_dir / 'test_lower.txt'}")

if __name__ == "__main__":
    create_train_test_splits() 