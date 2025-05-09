#!/bin/bash

# List of config paths
configs=(
    "./configs/report/unassigned_cleanup_mode_validation.json"
    "./configs/report/box_stacking_dep_mode_validation.json"
    "./configs/report/unassigned_cleanup_mode_rewiring.json"
    "./configs/report/box_stacking_dep_mode_rewiring.json"
    "./configs/report/box_stacking_dep_mode_sampling.json"
    "./configs/report/unassigned_cleanup_mode_sampling.json"
    "./configs/report/unassigned_cleanup_batch_size.json"
    "./configs/report/box_stacking_dep_batch_size.json"
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
