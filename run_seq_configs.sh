#!/bin/bash

# List of config paths
configs=(
    "./configs/experiments-ral/hallway_benchmark.json"
)

# Base command
base_cmd="python3 ./examples/run_experiment.py"

# Run each experiment
for config in "${configs[@]}"; do
    cmd="$base_cmd $config --parallel_execution --num_processes=2"
    echo "Running: $cmd"
    $cmd
    echo "Finished: $cmd"
done

echo "All experiments completed."
