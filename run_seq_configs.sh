#!/bin/bash

# List of config paths
configs=(
    "./configs/experiments/2d_handover_benchmark.json"
    "./configs/experiments/box_stacking_benchmark.json"
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
