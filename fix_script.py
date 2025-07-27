#!/usr/bin/env python3
import sys

# Read the original file
with open('run_hyperparameter_tuning.sh', 'r') as f:
    lines = f.readlines()

# Find the problematic section and replace it
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'read -p "Continue without wandb?' in line:
        # Replace the interactive section
        new_lines.append('    if [[ -t 0 ]]; then\n')
        new_lines.append('        read -p "Continue without wandb? (y/n): " -n 1 -r\n')
        new_lines.append('        echo\n')
        new_lines.append('        if [[ ! $REPLY =~ ^[Yy]$ ]]; then\n')
        new_lines.append('            exit 1\n')
        new_lines.append('        fi\n')
        new_lines.append('    else\n')
        new_lines.append('        echo "Running in non-interactive mode, continuing without wandb..."\n')
        new_lines.append('    fi\n')
        # Skip the original lines
        i += 1
        while i < len(lines) and ('echo' in lines[i] and lines[i].strip() == 'echo'):
            i += 1
        while i < len(lines) and 'if [[ ! $REPLY' in lines[i]:
            i += 1
            while i < len(lines) and ('exit 1' in lines[i] or 'fi' in lines[i]):
                i += 1
                break
    else:
        new_lines.append(line)
        i += 1

# Write the fixed file
with open('run_hyperparameter_tuning.sh', 'w') as f:
    f.writelines(new_lines)

print("Fixed the bad file descriptor issue in run_hyperparameter_tuning.sh")
