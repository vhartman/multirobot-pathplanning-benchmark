#!/bin/bash

# List of config paths
configs=(
    "./configs/report/other_hallway.json"
    "./configs/report/box_stacking_dep.json"
    "./configs/report/unordered_box_reorientation.json"
    "./configs/report/unassigned_cleanup.json"
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
