"""
Plot time spent in specific functions over time, from a py-spy speedscope profile.

Usage:
    python scripts/analysis/plot_aggregated_pyspy.py [profile.out]

The profile should be generated with:
    py-spy record --format speedscope -o profile.out -- python your_script.py

Configuration is done by editing the FUNCTIONS_TO_TRACK and OPTIONS dicts below.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import colorsys

import matplotlib.pyplot as plt
import numpy as np

path = "profile.out"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each entry is a display label mapped to a filter dict.
# Filter keys (all optional, combined with AND):
#   "name":  substring or regex matched against the function name
#   "file":  substring or regex matched against the file path
#
# Attribution is hierarchical: for each sample, all matching labels are
# collected outermost→innermost, forming a path tuple used as the legend key.
# E.g. a stack with is_edge_collision_free above is_collision_free gets the
# path ("is_edge_collision_free", "is_collision_free"), while is_collision_free
# called from sample_configuration gets ("sample_configuration", "is_collision_free").
# Unique paths are discovered at parse time and each gets its own bar segment.
FUNCTIONS_TO_TRACK: dict[str, dict] = {
    "is_collision_free": {"name": "is_collision_free"},
    "is_edge_collision_free": {"name": "is_edge_collision_free"},
    "sample_configuration": {"name": "sample_configuration"},
    "sample_informed": {"name": "sample_informed"},
    "_sample_goal": {"name": "sample_goal"},
    "shortcutting": {"name": "robot_mode_shortcut"}

}

OPTIONS = {
    # Which thread profile to use (index or substring of thread name).
    # None → use the longest profile.
    "profile": None,
    # Bucket size in seconds.
    "bucket_size": 0.5,
    # If True, normalise each bucket so that bars sum to 1.
    # If False, show absolute fraction of total profile time.
    "normalise_per_bucket": False,
    # Whether to stack the bars or draw them side-by-side.
    "stacked": True,
}

# Only consider samples whose call stack contains a frame matching this filter.
# Uses the same format as entries in FUNCTIONS_TO_TRACK.
# Set to None to consider all samples.
CALL_STACK_FILTER: dict | None = {"name": "plan"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches(frame: dict, filt: dict) -> bool:
    for key in ("name", "file"):
        if key not in filt:
            continue
        pattern = filt[key]
        value = frame.get(key) or ""
        if isinstance(pattern, str):
            if pattern not in value:
                return False
        else:  # compiled regex
            if not pattern.search(value):
                return False
    return True


def select_profile(profiles: list, selector) -> dict:
    if selector is None:
        return max(profiles, key=lambda p: p["endValue"] - p["startValue"])
    if isinstance(selector, int):
        return profiles[selector]
    # substring match on name
    matches = [p for p in profiles if selector in p["name"]]
    if not matches:
        raise ValueError(f"No profile matching {selector!r}")
    return matches[0]


def build_frame_index(frames: list, filters: dict[str, dict]) -> dict[int, list[str]]:
    """Map frame index → list of labels it belongs to."""
    index: dict[int, list[str]] = defaultdict(list)
    for i, frame in enumerate(frames):
        for label, filt in filters.items():
            if _matches(frame, filt):
                index[i].append(label)
    return index


def compute_buckets(
    profile: dict,
    frame_index: dict[int, list[str]],
    bucket_size: float,
    call_stack_filter_set: set[int] | None,
) -> tuple[np.ndarray, list[tuple[str, ...]], np.ndarray, np.ndarray]:
    """
    Returns:
        times:     centre of each bucket (seconds)
        paths:     list of label-path tuples, grouped by root (heaviest group first),
                   within each group sorted by total weight descending
        fractions: (n_buckets, n_paths) — each value is fraction of bucket duration
        other:     (n_buckets,) — remaining fraction of bucket not attributed to any path
        total_span: duration in seconds from first to last plan() sample
    """
    samples = profile["samples"]
    weights = profile["weights"]

    # Find the wall-clock span during which call_stack_filter_set is active,
    # and trim start/end to the first/last matching sample.
    if call_stack_filter_set is not None:
        t = profile["startValue"]
        start = end = None
        for stack, w in zip(samples, weights):
            if call_stack_filter_set.intersection(stack):
                if start is None:
                    start = t
                end = t + w
            t += w
        if start is None:
            raise ValueError("CALL_STACK_FILTER matched no samples — nothing to plot.")
        print(f"Trimmed to plan() span: {start:.2f}s – {end:.2f}s")
    else:
        start = profile["startValue"]
        end = profile["endValue"]

    n_buckets = max(1, int(np.ceil((end - start) / bucket_size)))
    all_bucket_weight = np.zeros(n_buckets)   # all samples, used as denominator
    bucket_weight = np.zeros(n_buckets)       # filtered samples only
    path_weight: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_buckets))

    t = profile["startValue"]
    for stack, w in zip(samples, weights):
        if t + w <= start:
            t += w
            continue
        if t >= end:
            break
        bucket = min(int((t - start) / bucket_size), n_buckets - 1)
        all_bucket_weight[bucket] += w

        if call_stack_filter_set is None or call_stack_filter_set.intersection(stack):
            bucket_weight[bucket] += w

            # Collect matched labels outermost→innermost, skip consecutive duplicates.
            path: list[str] = []
            for fi in stack:
                lbls = frame_index.get(fi)
                if lbls and (not path or path[-1] != lbls[0]):
                    path.append(lbls[0])
            if path:
                path_weight[tuple(path)][bucket] += w

        t += w

    # Sort paths: group by root, groups ordered by group total weight descending,
    # within each group ordered by path total weight descending.
    root_total = defaultdict(float)
    for p, w in path_weight.items():
        root_total[p[0]] += w.sum()
    paths = sorted(
        path_weight.keys(),
        key=lambda p: (-root_total[p[0]], -path_weight[p].sum()),
    )

    # Divide by actual bucket duration (sum of all sample weights in bucket).
    denom = all_bucket_weight.copy()
    denom[denom == 0] = 1.0

    fractions = np.zeros((n_buckets, len(paths)))
    path_seconds = np.zeros(len(paths))
    path_total = np.zeros(n_buckets)
    for j, p in enumerate(paths):
        fractions[:, j] = path_weight[p] / denom
        path_seconds[j] = path_weight[p].sum()
        path_total += path_weight[p]

    other = (all_bucket_weight - path_total) / denom

    times = (np.arange(n_buckets) + 0.5) * bucket_size  # t=0 at first plan() call
    return times, paths, fractions, other, path_seconds, end - start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    profile_path = sys.argv[1] if len(sys.argv) > 1 else path
    with open(profile_path) as f:
        data = json.load(f)

    frames = data["shared"]["frames"]
    profiles = data["profiles"]

    profile = select_profile(profiles, OPTIONS["profile"])
    print(f"Using profile: {profile['name']!r}  "
          f"[{profile['startValue']:.1f}s – {profile['endValue']:.1f}s]  "
          f"samples={len(profile['samples'])}")

    frame_index = build_frame_index(frames, FUNCTIONS_TO_TRACK)

    # Build the set of frame indices that satisfy CALL_STACK_FILTER.
    if CALL_STACK_FILTER is not None:
        csf_set = {i for i, fr in enumerate(frames) if _matches(fr, CALL_STACK_FILTER)}
        print(f"Call-stack filter {CALL_STACK_FILTER!r} matches {len(csf_set)} frames")
    else:
        csf_set = None

    times, paths, fractions, other, path_seconds, total_span = compute_buckets(
        profile, frame_index, OPTIONS["bucket_size"], csf_set
    )

    # Format path tuples as "a → b → c" for legend labels.
    path_labels = [" → ".join(p) for p in paths]
    print(f"Discovered {len(paths)} unique call paths:")
    for i, lbl in enumerate(path_labels):
        print(f"  {lbl!r}: avg {fractions[:, i].mean():.3f}")

    # --- Colors: one base hue per root label, lighten by depth ---
    roots = list(dict.fromkeys(p[0] for p in paths))  # unique roots, insertion-ordered
    base_hues = np.linspace(0, 1, len(roots) + 1)[:-1]  # evenly spaced hues
    root_hue = {r: h for r, h in zip(roots, base_hues)}
    # Count how many siblings share the same root, to spread lightness values.
    root_counts: dict[str, int] = defaultdict(int)
    root_seen: dict[str, int] = defaultdict(int)
    for p in paths:
        root_counts[p[0]] += 1

    def path_color(p: tuple[str, ...]) -> tuple:
        h = root_hue[p[0]]
        idx = root_seen[p[0]]
        root_seen[p[0]] += 1
        n = root_counts[p[0]]
        # Lightness: 0.35 (darkest, first sibling) → 0.70 (lightest, last sibling)
        lightness = 0.35 + 0.35 * (idx / max(n - 1, 1))
        r, g, b = colorsys.hls_to_rgb(h, lightness, 0.65)
        return (r, g, b)

    colors = [path_color(p) for p in paths]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    width = OPTIONS["bucket_size"] * 0.9

    if OPTIONS["stacked"]:
        bottom = np.zeros(len(times))
        for i, (lbl, color) in enumerate(zip(path_labels, colors)):
            ax.bar(times, fractions[:, i], width=width, bottom=bottom,
                   label=lbl, color=color, alpha=0.85)
            bottom += fractions[:, i]
        ax.bar(times, other, width=width, bottom=bottom,
               label="other", color="lightgray", alpha=0.85)
    else:
        offsets = np.linspace(-width / 2, width / 2, len(path_labels) + 1)[:-1]
        bar_w = width / max(len(path_labels), 1)
        for i, (lbl, color) in enumerate(zip(path_labels, colors)):
            ax.bar(times + offsets[i], fractions[:, i], width=bar_w,
                   label=lbl, color=color, alpha=0.85)

    y_label = "Fraction of bucket duration"
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Time breakdown — {Path(profile_path).name}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, total_span)
    ax.set_ylim(0, None)

    plt.tight_layout()

    # --- Aggregated summary ---
    print(f"\nAggregated over {total_span:.2f}s total plan() time:")
    col = max(len(lbl) for lbl in path_labels)
    for lbl, secs in zip(path_labels, path_seconds):
        print(f"  {lbl:<{col}}  {secs:7.2f}s  ({100 * secs / total_span:5.1f}%)")
    other_secs = total_span - path_seconds.sum()
    print(f"  {'other':<{col}}  {other_secs:7.2f}s  ({100 * other_secs / total_span:5.1f}%)")

    plt.show()


if __name__ == "__main__":
    main()
