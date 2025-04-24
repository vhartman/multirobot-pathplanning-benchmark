import json
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
import argparse

# Hardcoded parametrize mapping from test definition
DIMS_MAP = {
    "dims0": (2, 2),
    "dims1": (7, 7),
    "dims2": (3, 3, 3),
    "dims3": (14,)
}

def parse_benchmark_name(name):
    # Example name: test_batch_config_cost_benchmark[100-dims0-max]
    match = re.search(r"\[(\d+)-(dims\d+)-(\w+)\]", name)
    if not match:
        return None
    num_points_str, dims_key, reduction = match.groups()
    dims = DIMS_MAP.get(dims_key)
    if dims is None:
        return None
    return reduction, dims, int(num_points_str)

def load_results_from_folder(folder_path):
    results = defaultdict(lambda: defaultdict(list))  # results[(reduction, dims)][file_name] = [(num_points, mean_time, stddev)]
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder '{folder_path}' does not exist or is not a directory.")

    files = list(folder.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in '{folder_path}'.")

    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {file.name}: JSON load error: {e}")
            continue

        for b in data.get("benchmarks", []):
            parsed = parse_benchmark_name(b.get("name", ""))
            if parsed is None:
                continue
            reduction, dims, num_points = parsed
            stats = b.get("stats", {})
            mean = stats.get("mean")
            stddev = stats.get("stddev")
            if mean is None or stddev is None:
                continue
            results[(reduction, dims)][file.name].append((num_points, mean, stddev))
    
    return results

def plot_results(results, output_dir="plots"):
    Path(output_dir).mkdir(exist_ok=True)

    # Plot for each (reduction, dims)
    for (reduction, dims), file_results in results.items():
        plt.figure()

        # Plot each file's results as a separate line
        for file_name, entries in file_results.items():
            entries.sort()  # Sort entries by num_points
            x = [e[0] for e in entries]
            y = [e[1] for e in entries]
            yerr = [e[2] for e in entries]

            plt.errorbar(x, y, yerr=yerr, fmt="o-", label=f"{file_name}", capsize=3)

        plt.title(f"Reduction: {reduction}, Dims: {dims}")
        plt.xlabel("Number of Points")
        plt.ylabel("Mean Time (s)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(title="Benchmark File")
        plt.grid(True)
        plt.tight_layout()

        filename = f"{reduction}_{'_'.join(map(str, dims))}.png"
        plt.savefig(Path(output_dir) / filename)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing benchmark JSON files")
    parser.add_argument("--output", default="plots", help="Output folder for plots")
    args = parser.parse_args()

    try:
        results = load_results_from_folder(args.folder)
        if not results:
            print("No valid benchmark entries found.")
        else:
            plot_results(results, output_dir=args.output)
            print(f"Saved plots to '{args.output}'")
    except Exception as e:
        print(f"Error: {e}")
