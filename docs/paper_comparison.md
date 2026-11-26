# Paper comparison

Run from the repository root with the required planner/backends installed.
`benchmarks/paper/inputs/` holds frozen geometry, goals, starts and assignments;
`benchmarks/paper/protocols/` fixes settings, seeds and budgets. Inputs are checked
by hash. Packing reuses the tracked RoboBallet models. No IK or separate data
archive is needed.

```bash
# Default: table suite, estimate only; does not plan.
python scripts/paper/run.py

# Load/check only, without planning or warmup.
python scripts/paper/run.py --check --output out/paper-check

# Run later; serial by default. Repeat to resume completed cells.
python scripts/paper/run.py --run --resume --max-hours 8
```

The table suite is 190 cells / 5.75 planning hours. To include welding/packing,
add `--protocol benchmarks/paper/protocols/paper_fixed_sequence.json
benchmarks/paper/protocols/paper_additional.json`: 240 cells / 7.08 hours total,
plus setup and validation. Both use seeds 2–6. Select subsets with `--scenes
rai.box_stacking --seeds 2 3 --backends comotion --seconds 60`.

Results default to ignored `out/paper-comparison/`. Source, dependencies, input
hashes and settings are recorded. Changed settings require a new `--output`.
Review failed/interrupted cells before resuming with `--retry-failed`; previous
attempts are archived. Use `--allow-battery` only for exploratory runs.

`convergence/` holds per-cell histories and first/final paths. Incremental
`.progress.jsonl` files preserve interrupted histories, which are not scored.
Validation uses the planner's collision resolution. Timings exclude loading,
warmup and validation. `summary.csv` reports medians and published-cost targets;
timeouts count as infinity, incomplete groups have no summary median.

Complete scenes are exported to `analysis/` in the existing plotting format:

```bash
python scripts/paper/run.py --output out/paper-comparison --summarize
python scripts/analysis/make_plots.py out/paper-comparison/analysis \
  --recursive --cost_only --logscale --legend --save --png --no_display
```

The plot export includes all seeds and keeps timeouts as infinite cost; it waits
for a scene to finish rather than plotting a successful subset. The existing plotting
tool reads its `config.json`, `timestamps.txt` and `costs.txt`.

Commit the runner/loader, protocols, inputs and this document. Keep `out/` out of
Git. `scripts/paper/frozen_inputs.py` restores the frozen problems and selects the
backend; it does not alter registered environment builders. Published references
retain their original hardware/runtime context. The four-arm spiral CoMotion
case requires the joint-limit roundoff guard noted in its protocol.
