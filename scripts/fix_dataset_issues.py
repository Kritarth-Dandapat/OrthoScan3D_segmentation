#!/usr/bin/env python3
"""
Script to fix dataset issues and create proper train/test splits.
"""

import os
import random
from pathlib import Path

def find_complete_patients():
    """Find patients with complete data (both .obj and .json files for upper and lower)."""
    
    upper_dir = Path("data/3dteethseg/raw/upper")
    lower_dir = Path("data/3dteethseg/raw/lower")
    
    complete_patients = []
    missing_data = []
    
    for patient_dir in upper_dir.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            
            # Check files
            upper_obj = patient_dir / f"{patient_id}_upper.obj"
            upper_json = patient_dir / f"{patient_id}_upper.json"
            lower_obj = lower_dir / patient_id / f"{patient_id}_lower.obj"
            lower_json = lower_dir / patient_id / f"{patient_id}_lower.json"
            
            # Check if all files exist
            files_exist = all([
                upper_obj.exists(),
                upper_json.exists(),
                lower_obj.exists(),
                lower_json.exists()
            ])
            
            if files_exist:
                complete_patients.append(patient_id)
            else:
                missing = []
                if not upper_obj.exists():
                    missing.append("upper_obj")
                if not upper_json.exists():
                    missing.append("upper_json")
                if not lower_obj.exists():
                    missing.append("lower_obj")
                if not lower_json.exists():
                    missing.append("lower_json")
                missing_data.append((patient_id, missing))
    
    print(f"Patients with complete data: {len(complete_patients)}")
    print(f"Patients with missing data: {len(missing_data)}")
    
    if missing_data:
        print("\nSample of missing data:")
        for patient_id, missing in missing_data[:5]:
            print(f"  {patient_id}: missing {missing}")
    
    return complete_patients

def create_proper_splits(complete_patients):
    """Create train/test splits for patients with complete data."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle the patient IDs
    random.shuffle(complete_patients)
    
    # Split into train/test (80% train, 20% test)
    split_idx = int(0.8 * len(complete_patients))
    train_patients = complete_patients[:split_idx]
    test_patients = complete_patients[split_idx:]
    
    print(f"\nSplit results:")
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Test patients: {len(test_patients)}")
    
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
    
    print("\nSplit files created successfully!")
    print(f"Files created:")
    print(f"  - {splits_dir / 'training_upper.txt'}")
    print(f"  - {splits_dir / 'training_lower.txt'}")
    print(f"  - {splits_dir / 'test_upper.txt'}")
    print(f"  - {splits_dir / 'test_lower.txt'}")

def main():
    """Main function to fix dataset issues."""
    print("Analyzing dataset...")
    complete_patients = find_complete_patients()
    
    if complete_patients:
        print(f"\nCreating splits for {len(complete_patients)} patients with complete data...")
        create_proper_splits(complete_patients)
    else:
        print("No patients with complete data found!")

if __name__ == "__main__":
    main() 