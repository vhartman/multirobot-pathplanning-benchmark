#!/bin/bash

# List of config paths
configs=(
    "./configs/experiments/demo/box_stacking_dep.json"
    "./configs/experiments/demo/hallway.json"
)

# Base command
base_cmd="python3 ./examples/run_experiment.py"

# Run each experiment
for config in "${configs[@]}"; do
    cmd="$base_cmd $config --parallel_execution --num_processes=6"
    echo "Running: $cmd"
    $cmd
    echo "Finished: $cmd"
done

echo "All experiments completed."
