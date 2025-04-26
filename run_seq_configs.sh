#!/bin/bash

# List of config paths
configs=(
    "./configs/demo/other_hallway.json"
    "./configs/demo/box_stacking_dep.json"
    "./configs/demo/unordered_box_reorientation.json"
)

# Base command
base_cmd="python3 examples/run_experiment.py"

# Run each experiment
for config in "${configs[@]}"; do
    cmd="$base_cmd $config --parallel_execution --num_processes=6"
    echo "Running: $cmd"
    eval "$cmd"
    echo "Finished: $cmd"
done

echo "All experiments completed."
