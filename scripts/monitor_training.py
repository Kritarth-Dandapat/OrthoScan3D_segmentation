import os
import time
import subprocess
import glob

def monitor_training():
    """Monitor the training progress."""
    
    print("=== Training Monitor ===")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for running training processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        training_processes = []
        
        for line in lines:
            if 'train_full_dataset' in line and 'python' in line:
                training_processes.append(line)
        
        print(f"\nFound {len(training_processes)} training processes:")
        for i, proc in enumerate(training_processes):
            print(f"  {i+1}. {proc.strip()}")
    
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    # Check for log files
    print("\n=== Log Files ===")
    log_dirs = ['logs', 'tensorboard_logs', 'lightning_logs']
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"✓ {log_dir} directory exists")
            files = os.listdir(log_dir)
            print(f"  Files: {len(files)}")
            for file in files[:5]:  # Show first 5 files
                print(f"    - {file}")
        else:
            print(f"✗ {log_dir} directory not found")
    
    # Check for model files
    print("\n=== Model Files ===")
    model_files = glob.glob("*.pth") + glob.glob("checkpoint_*.pth")
    if model_files:
        for model_file in model_files:
            size = os.path.getsize(model_file) / (1024*1024)  # MB
            mtime = time.ctime(os.path.getmtime(model_file))
            print(f"  {model_file} ({size:.1f} MB, modified: {mtime})")
    else:
        print("  No model files found yet")
    
    # Check for any output files
    print("\n=== Recent Files ===")
    try:
        result = subprocess.run(['find', '.', '-name', '*.pth', '-o', '-name', '*.ckpt', '-o', '-name', '*.log', '-mtime', '-1'], 
                              capture_output=True, text=True)
        recent_files = result.stdout.strip().split('\n')
        if recent_files and recent_files[0]:
            for file in recent_files[:10]:  # Show first 10 files
                if file:
                    print(f"  {file}")
        else:
            print("  No recent files found")
    except Exception as e:
        print(f"Error checking recent files: {e}")

if __name__ == "__main__":
    monitor_training() 