import os

processed_dir = "data/3dteethseg/processed"
split_files = [
    "data/3dteethseg/raw/training_lower.txt",
    "data/3dteethseg/raw/training_upper.txt",
    "data/3dteethseg/raw/testing_lower.txt",
    "data/3dteethseg/raw/testing_upper.txt",
    "data/3dteethseg/raw/public-training-set-1.txt",
    "data/3dteethseg/raw/public-training-set-2.txt",
    "data/3dteethseg/raw/private-testing-set.txt",
]

# Get all processed .pt files
processed_files = set(os.listdir(processed_dir))

for split_path in split_files:
    if not os.path.exists(split_path):
        print(f"Split file {split_path} does not exist, skipping.")
        continue

    with open(split_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Only keep lines where the corresponding processed file exists
    kept = []
    for line in lines:
        pt_file = f"data_{line}.pt"
        if pt_file in processed_files:
            kept.append(line)
        else:
            print(f"Missing: {pt_file} (removing from {split_path})")

    # Overwrite the split file with only valid entries
    with open(split_path, "w") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"Cleaned {split_path}: {len(kept)} entries remain.")

print("All split files cleaned.") 