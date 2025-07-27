import subprocess
import time
import random

PYTHON = "./braces_env/bin/python"

# Generate 5 experiment configs with epochs from 50 to 100 and batch_size fixed at 5
experiment_configs = []
for i in range(5):
    config = {
        "epochs": random.randint(50, 100),
        "batch_size": 5,  # Fixed batch size
        "experiment_version": f"auto_{i+1}",
        "learning_rate": random.choice([0.001, 0.0005, 0.0001]),
        "weight_decay": random.choice([0.0001, 0.0005]),
        "devices": "0"
    }
    experiment_configs.append(config)

for i, config in enumerate(experiment_configs):
    args = [
        PYTHON, "train_processed_data_wandb.py",
        "--use_wandb",
        "--wandb_project", "teeth-segmentation-3d",
        "--experiment_name", "overnight_sweep",
        "--experiment_version", config["experiment_version"],
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--weight_decay", str(config["weight_decay"]),
        "--devices", config["devices"]
    ]
    log_file = f"exp_{i+1}_{config['experiment_version']}.log"
    print(f"Launching experiment {i+1}/5: {log_file} with config: {config}")
    subprocess.Popen(args, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    time.sleep(2)  # Stagger launches

print("All experiments launched in background. Check .log files and wandb for progress.") 