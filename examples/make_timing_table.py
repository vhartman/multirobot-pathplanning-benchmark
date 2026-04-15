"""
Print a LaTeX table of median timing breakdown per environment.

Rows: environments (experiment folders)
Columns: sampling | edge (free) | edge (blocked) | coll. check | other

"other" = first_solution_time - (sampling + edge_free + edge_blocked + coll_check)

If multiple planners are present in an experiment, only birrtstar is used.
"""
import os
import sys
import numpy as np
from string import Template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_plots import load_data_from_folder


def load_timing_for_folder(experiment_folder: str) -> tuple[str | None, list[tuple]]:
    """
    Returns (planner_name, [(first_sol_time, sampling, edge_free, edge_blocked, coll_check), ...])
    for the chosen planner, or (None, []) if no timing.csv found.

    Pairs timing.csv rows with timestamps by run_id.
    If multiple planners have timing.csv, prefer birrtstar.
    """
    candidates = {}
    for entry in os.scandir(experiment_folder):
        if not entry.is_dir():
            continue
        timing_file = os.path.join(entry.path, "timing.csv")
        if not os.path.isfile(timing_file):
            continue

        # load timestamps for this planner (first solution time per run)
        planner_data = load_data_from_folder(experiment_folder + "/", load_paths=0)
        first_sol_times = {}
        for run_idx, run in enumerate(planner_data.get(entry.name, [])):
            times = run.get("times", [])
            if times:
                first_sol_times[run_idx] = times[0]

        rows = []
        with open(timing_file) as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 5:
                    run_id, s, es, ef, cc = parts
                elif len(parts) == 4:
                    run_id, s, es, ef = parts
                    cc = "0"
                else:
                    continue
                run_id = int(run_id)
                t_first = first_sol_times.get(run_id)
                if t_first is None:
                    continue
                rows.append((t_first, float(s), float(es), float(ef), float(cc)))

        if rows:
            candidates[entry.name] = rows

    if not candidates:
        return None, []

    for name in candidates:
        if "birrt" in name.lower():
            return name, candidates[name]
    name, rows = next(iter(candidates.items()))
    return name, rows


def aggregate_timing(folders: list[str]) -> dict:
    """
    Returns:
        results[env_name] = {
            "planner": str,
            "sampling": float,       # median fraction
            "edge_free": float,
            "edge_blocked": float,
            "coll_check": float,
            "other": float,          # median(first_sol_time) - above four
        }
    """
    results = {}
    for folder in folders:
        env_name = folder.split(".")[-1][:-1]
        planner_name, rows = load_timing_for_folder(folder)
        if not rows:
            continue

        t_firsts    = [r[0] for r in rows]
        samplings   = [r[1] for r in rows]
        edge_frees  = [r[2] for r in rows]
        edge_blocks = [r[3] for r in rows]
        coll_checks = [r[4] for r in rows]

        med_t   = np.median(t_firsts)
        med_s   = np.median(samplings)
        med_ef  = np.median(edge_frees)
        med_eb  = np.median(edge_blocks)
        med_cc  = np.median(coll_checks)

        other = med_t - med_s - med_ef - med_eb - med_cc
        results[env_name] = {
            "planner": planner_name,
            "first_sol_time": med_t,
            "sampling": med_s / med_t if med_t > 0 else 0.0,
            "edge_free": med_ef / med_t if med_t > 0 else 0.0,
            "edge_blocked": med_eb / med_t if med_t > 0 else 0.0,
            "coll_check": med_cc / med_t if med_t > 0 else 0.0,
            "other": other / med_t if med_t > 0 else 0.0,
        }

    return results


def print_latex_table(aggregated: dict) -> None:
    colspec = "l | r | r@{\\hspace{8pt}} r@{\\hspace{8pt}} r@{\\hspace{8pt}} r@{\\hspace{8pt}} r"
    subheader  = r"& & \multicolumn{5}{c}{Fraction of total time [\%]} \\"
    cmidrule1  = r"\cmidrule(lr){3-7}"
    subheader2 = r"& & & \multicolumn{3}{c}{Collision checking} & \\"
    cmidrule2  = r"\cmidrule(lr){4-6}"
    header = (
        r"Environment & $t_\text{init}$ [s] "
        r"& Sampling & Single config & Edge (free) & Edge (blocked) & Other"
    )

    rows = []
    for env_name, d in aggregated.items():
        escaped = env_name.replace("_", r"\_")
        fracs = {
            "sampling":     d["sampling"],
            "edge_free":    d["edge_free"],
            "edge_blocked": d["edge_blocked"],
            "coll_check":   d["coll_check"],
            "other":        d["other"],
        }
        best = max(fracs, key=fracs.get)

        def fmt_frac(key):
            val = f"{fracs[key]*100:.1f}"
            if key == best:
                return f"$\\mathbf{{{val}}}$"
            return f"${val}$"

        t = f"${d['first_sol_time']:.2f}$"
        rows.append(
            f"{escaped} & {t} & {fmt_frac('sampling')} & {fmt_frac('coll_check')} & {fmt_frac('edge_free')}"
            f" & {fmt_frac('edge_blocked')} & {fmt_frac('other')} \\\\"
        )

    body = "\n".join(rows)

    TEX_TEMPLATE = Template(r"""
\begin{tabular}{$colspec}
\toprule
$subheader
$cmidrule1
$subheader2
$cmidrule2
$header \\
\midrule
$body
\bottomrule
\end{tabular}
""")
    print(TEX_TEMPLATE.substitute(
        colspec=colspec, subheader=subheader, cmidrule1=cmidrule1,
        subheader2=subheader2, cmidrule2=cmidrule2, header=header, body=body,
    ))


def get_subfolders(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    subdirs = [
        os.path.join(folder, e.name) + "/"
        for e in os.scandir(folder)
        if e.is_dir()
    ]
    subdirs.sort()
    return subdirs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Top-level experiment folder containing subfolders per environment")
    args = parser.parse_args()

    folders = get_subfolders(args.folder)
    if not folders:
        folders = [args.folder]

    aggregated = aggregate_timing(folders)
    if not aggregated:
        print("No timing.csv files found.")
        return

    print_latex_table(aggregated)


if __name__ == "__main__":
    main()