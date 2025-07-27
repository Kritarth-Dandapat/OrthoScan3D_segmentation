#!/usr/bin/env python3
"""
Script to create train/test split files for patients with complete data.
"""

import os
import random
from pathlib import Path

def create_complete_train_test_splits():
    """Create train/test split files only for patients with complete data."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Find patients with complete data (both .obj and .json files)
    upper_dir = Path("data/3dteethseg/raw/upper")
    lower_dir = Path("data/3dteethseg/raw/lower")
    
    complete_patients = []
    
    for patient_dir in upper_dir.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            
            # Check if both upper and lower have complete data
            upper_obj = patient_dir / f"{patient_id}_upper.obj"
            upper_json = patient_dir / f"{patient_id}_upper.json"
            lower_obj = lower_dir / patient_id / f"{patient_id}_lower.obj"
            lower_json = lower_dir / patient_id / f"{patient_id}_lower.json"
            
            if (upper_obj.exists() and upper_json.exists() and 
                lower_obj.exists() and lower_json.exists()):
                complete_patients.append(patient_id)
    
    print(f"Patients with complete data: {len(complete_patients)}")
    
    # Shuffle the patient IDs
    random.shuffle(complete_patients)
    
    # Split into train/test (80% train, 20% test)
    split_idx = int(0.8 * len(complete_patients))
    train_patients = complete_patients[:split_idx]
    test_patients = complete_patients[split_idx:]
    
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
    
    print("Complete train/test split files created successfully!")
    print(f"Files created:")
    print(f"  - {splits_dir / 'training_upper.txt'}")
    print(f"  - {splits_dir / 'training_lower.txt'}")
    print(f"  - {splits_dir / 'test_upper.txt'}")
    print(f"  - {splits_dir / 'test_lower.txt'}")

if __name__ == "__main__":
    create_complete_train_test_splits() 