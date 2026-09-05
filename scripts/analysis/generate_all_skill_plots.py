import argparse
import json
import pathlib
import re
import subprocess
import sys

ANALYSIS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from make_plots_stochastic_scalability import robot_count

TWO_D_ENVS = ("square_island", "rectangle_island", "shared_point")


def env_of(folder: pathlib.Path) -> str:
    config = folder / "config.json"
    if config.exists():
        with open(config) as f:
            return json.load(f).get("environment", folder.name)
    return folder.name


def has_executions(folder: pathlib.Path) -> bool:
    return any(folder.glob("*/*/executions.json"))


def family_key(env_name: str):
    if robot_count(env_name) is not None:
        return re.sub(r"_\d+r_\d+b", "_Nr_Nb", env_name)
    return None


def run(cmd, what):
    print(f"\n--- {what}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"    [FAILED] {what} (exit {result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("folders", nargs="+", help="experiment folders (globs are fine)")
    parser.add_argument("--out", default="out/plots",
                        help="where the cross-environment figures go; per-folder figures stay "
                             "in their own folder")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--exec_run", type=int, default=None,
                        help="restrict the distribution figures to ONE planning run")
    args = parser.parse_args()

    folders = [pathlib.Path(f) for f in args.folders]
    folders = [f for f in folders if f.is_dir() and (f / "config.json").exists()]
    if not folders:
        print("No experiment folders found (need a config.json in each).")
        return

    shared = (["--pdf"] if args.pdf else []) + (["--no_legend"] if args.no_legend else [])
    shared += ["--paper"] if args.paper else []
    out_root = pathlib.Path(args.out)

    families = {}
    for folder in sorted(folders):
        env_name = env_of(folder)
        print(f"\n=== {folder.name}  ({env_name})")

        if has_executions(folder):
            cmd = [str(ANALYSIS / "make_plots_stochastic_skill.py"), str(folder),
                   "--out", str(folder / "plots_stochastic")] + shared
            if args.exec_run is not None:
                cmd += ["--exec_run", str(args.exec_run)]
            run(cmd, "cost figures")

            if any(tag in env_name for tag in TWO_D_ENVS):
                run([str(ANALYSIS / "make_plots_stochastic_topdown.py"), str(folder),
                     "--out", str(folder / "plots_stochastic")] + shared, "top-down view")
        else:
            det_flags = ["--save", "--no_display"] + (["--png"] if not args.pdf else [])
            if not args.no_legend:
                det_flags.append("--legend")
            run([str(ANALYSIS / "make_plots_deterministic_skill.py"), str(folder)] + det_flags,
                "deterministic anytime figures")

        task = family_key(env_name)
        if task is not None:
            families.setdefault(task, []).append(folder)

    for task, members in sorted(families.items()):
        if len(members) < 2:
            print(f"\n=== skipping scalability figure for {task}: only {len(members)} environment")
            continue
        out_dir = out_root / task.replace("rai.", "")
        run([str(ANALYSIS / "make_plots_stochastic_scalability.py"),
             "--out", str(out_dir)] + [str(m) for m in members] + shared,
            f"scalability figures over {len(members)} environments -> {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
