#!/bin/bash

# List of config paths
configs=(

)

# Base command
base_cmd="python3 ./examples/make_plots.py"

# Run each experiment
for config in "${configs[@]}"; do
    cmd="$base_cmd $config --save --no_display --logscale --legend --png"
    echo "Running: $cmd"
    $cmd
    echo "Finished: $cmd"
done

echo "All experiments completed."
