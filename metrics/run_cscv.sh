#!/bin/bash

base_path="your_data_path"
# Define array of paths to process
paths=(
    "ditctrl"
)

# Define array of seeds to process
seeds=(42)

# Define array of Python scripts to run
scripts=(
    "cscv_metric.py"
)

# Iterate through all combinations
for script in "${scripts[@]}"; do
    for path in "${paths[@]}"; do
        # Concatenate full path
        full_path="${base_path}/${path}"
        for seed in "${seeds[@]}"; do
            echo "Running script: $script, Processing path: $full_path, seed: $seed"
            python "$script" \
                --video_path "$full_path" \
                --target_seed $seed
        done
    done
done