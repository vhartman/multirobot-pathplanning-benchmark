#!/usr/bin/env python3
"""Compare the final paper tag with committed HEAD, serially, using RAI/FCL.

Dry run: python scripts/run_revision_comparison.py
Run:     python scripts/run_revision_comparison.py --run --output out/tag-vs-head
Resume:  repeat the command with --resume

Requires Linux/macOS, git, and one Python environment with the repository's
RAI dependencies. Both revisions use that same installed dependency stack.
No editable install, checkout change, native adapter, or repository commit is
performed. Uncommitted source changes are deliberately NOT benchmarked.
"""
import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback

PACKAGE = 'multi_robot_multi_goal_planning'
# Original suite's scene seeds; simple/single_agent_mover also appear in Table 2.
SCENES = dict((f'rai.{name}', seed) for name, seed in [
    ('simple', 2), ('single_agent_mover', 2), ('dep_piano', 2),
    ('2d_handover', 2), ('random_2d', 2), ('other_hallway', 2),
    ('three_agents', 2), ('triple_waypoints', 2), ('welding', 2),
    ('handover', 2), ('eggcartons', 2), ('bottles', 2),
    ('box_rearrangement', 2), ('box_reorientation', 2),
    ('dep_box_reorientation', 2), ('pyramid', 2), ('box_stacking', 2),
    ('dep_box_stacking', 2), ('mobile_wall_four', 2),
    ('dep_mobile_wall_four', 2), ('mobile_strut', 0),
    ('three_robot_truss', 7), ('spiral_tower', 0), ('spiral_tower_two', 0),
    ('cube_four', 2), ('unordered_box_reorientation', 0),
    ('unassigned_tsp', 0), ('unassigned_cleanup', 0),
    ('unassigned_stacking', 0), ('unordered_bottles', 0)])


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def dependencies():
    return sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())


def on_mains():
    supplies = list(Path('/sys/class/power_supply').glob('*/online'))
    return any(p.read_text().strip() == '1' for p in supplies) if supplies else None


def sleep_offset():
    if hasattr(time, 'CLOCK_BOOTTIME'):
        return time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()
    return time.time() - time.monotonic()


def power_guard(allow_battery, before=None):
    if not allow_battery and on_mains() is False:
        raise RuntimeError('Mains disconnected; use --allow-battery only for non-benchmark checks')
    if before is not None and sleep_offset() - before > 1:
        raise RuntimeError('System sleep detected; interrupted cell is not counted')


def clean_env(output, variant):
    env = os.environ.copy()
    for key in list(env):
        if key in ('PYTHONPATH', 'LD_PRELOAD') or key.startswith(('MR_COAL', 'MR_FCL', 'MR_MIDPHASE')):
            env.pop(key)
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               NUMBA_CACHE_DIR=str(output / 'cache' / variant),
               MPLCONFIGDIR=str(output / 'cache' / 'matplotlib'), PYTHONHASHSEED='0')
    return env


def snapshot(repo, commit, dest):
    dest.mkdir(parents=True)
    with subprocess.Popen(['git', '-C', str(repo), 'archive', commit], stdout=subprocess.PIPE) as archive:
        subprocess.run(['tar', '-x', '-f', '-', '-C', str(dest)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError(f'Could not export {commit}')
    # Record all tracked contents, including assets, to detect modified snapshots.
    return {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}


def same(a, b, where='scene'):
    """Compare numeric scene data tolerantly; structure and names exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f'{where}: different keys')
        for k in a:
            same(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(f'{where}: different lengths')
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{where}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-7):
            raise ValueError(f'{where}: {a} != {b}')
    elif a != b:
        raise ValueError(f'{where}: values differ')


class BudgetExpired(BaseException):
    pass


class FirstFound(BaseException):
    pass


def alarm(*_):
    raise BudgetExpired()


def paper_scene_factory(output, variant):
    """Use tagged task/geometry constructors with the selected revision's runtime.

    Only scene construction is shared. Planner, collision checking, mode logic,
    configuration types and goals still come from the selected revision.
    """
    import importlib
    import types
    if variant == 'paper':
        return importlib.import_module(PACKAGE + '.problems').get_env_by_name
    namespace = '_revision_comparison_scenes'
    package = types.ModuleType(namespace)
    package.__path__ = [str(output / 'sources/paper/src' / PACKAGE / 'problems')]
    sys.modules[namespace] = package
    for short, actual in [('planning_env', 'planning_env'), ('rai_base_env', 'rai_base_env'),
                          ('goals', 'core.goals'), ('configuration', 'core.configuration'),
                          ('dependency_graph', 'core.dependency_graph')]:
        sys.modules[namespace + '.' + short] = importlib.import_module(PACKAGE + '.problems.' + actual)
    config = importlib.import_module(namespace + '.rai_config')
    # Tagged scene constructors use this old absolute import; current runtime
    # imports problems.rai.rai_config instead and is not replaced.
    sys.modules[PACKAGE + '.problems.rai_config'] = config
    for module in ('rai_envs', 'rai_single_goal_envs', 'rai_unordered_envs', 'rai_free_envs'):
        importlib.import_module(namespace + '.' + module)
    return importlib.import_module(namespace + '.registry').get_env_by_name


def worker(a):
    import copy
    import random
    import numpy as np
    root = a.output / 'sources' / a.variant
    sys.path.insert(0, str(root / 'src'))
    from multi_robot_multi_goal_planning.problems import get_env_by_name
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
    from multi_robot_multi_goal_planning.planners.termination_conditions import RuntimeTerminationCondition
    import multi_robot_multi_goal_planning.planners.rrtstar_base as module
    import robotic as ry
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Wrong planner package imported')
    p = read(a.output / 'protocol.json')['settings']
    random.seed(SCENES[a.scene]); np.random.seed(SCENES[a.scene])
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, p['setup_timeout'])
    env = paper_scene_factory(a.output, a.variant)(a.scene)
    signal.setitimer(signal.ITIMER_REAL, 0)
    env.cost_metric, env.cost_reduction = 'euclidean', 'max'
    if env.is_collision_free.__module__ != PACKAGE + '.problems.rai_base_env':
        raise RuntimeError('Expected standard RAI collision checking')
    cfg = BaseRRTConfig()
    def plain(value):
        return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, 'tolist') else sorted(x)))
    tasks, goals = [], []
    for t in env.tasks:
        if t.constraints:
            raise ValueError('This comparison supports the unconstrained paper suite only')
        tasks.append(plain({k: getattr(t, k) for k in ('name', 'robots', 'type', 'frames', 'side_effect', 'side_effect_data')}))
        if type(t.goal).__name__ == 'ConditionalGoal':
            value = dict(conditions=plain(t.goal.conditions), goals=plain(t.goal.goals))
        else:
            value = plain(t.goal.serialize())
        goals.append(dict(type=type(t.goal).__name__, data=value))
    # C_orig retains visual and collision geometry; mesh vertices catch asset changes.
    C = env.C_orig
    geometry = C.write()
    for variant in ('paper', 'current'):
        geometry = geometry.replace(str(a.output / 'sources' / variant), '<REPO>')
    numeric = re.compile(r'([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)')
    tokens = numeric.split(geometry)
    geometry = [float(t) if i % 2 else ' '.join(t.split()) for i, t in enumerate(tokens)]
    meshes = {}
    for name in C.getFrameNames():
        f = C.getFrame(name)
        if str(f.getShapeType()).lower().endswith('mesh'):
            meshes[name] = [np.asarray(f.getMeshPoints()).round(9).tolist(), np.asarray(f.getMeshTriangles()).tolist()]
    scene = plain(dict(start=env.start_pos.state(), limits=env.limits,
        robots=env.robots, robot_dims=env.robot_dims, tasks=tasks,
        sequence=getattr(env, 'sequence', None),
        dependencies=getattr(getattr(env, 'graph', None), 'dependencies', None),
        resolution=env.collision_resolution, tolerance=env.collision_tolerance,
        geometry=geometry, meshes=meshes))
    reference = a.output / 'scenes' / f'{a.scene}.json'
    if a.action == 'prepare' and a.variant == 'paper' and not reference.exists():
        write(reference, dict(scene=scene, goals=goals))
    frozen = read(reference)
    same(frozen['scene'], scene)
    for t, g in zip(env.tasks, frozen['goals']):
        if type(t.goal).__name__ != g['type']:
            raise ValueError('Goal class changed')
        if g['type'] == 'ConditionalGoal':
            t.goal = type(t.goal)(np.array(g['data']['conditions']), np.array(g['data']['goals']))
        else:
            t.goal = type(t.goal).from_data(g['data'])
    metadata = dict(config=dataclasses.asdict(cfg), robotic_version=importlib.metadata.version('robotic'),
        robotic_module=ry.__file__, scene_match=True, goals_restored=True,
        generated_goals_match=goals == frozen['goals'])
    if a.action == 'prepare':
        write(a.output / 'preflight' / f'{a.scene}-{a.variant}.json', metadata)
        return
    # Warm compilation/import paths outside the measured interval on a disposable copy.
    if p['warmup']:
        warm = BidirectionalRRTstar(copy.deepcopy(env), copy.deepcopy(cfg))
        signal.setitimer(signal.ITIMER_REAL, p['warmup'])
        try:
            warm.plan(RuntimeTerminationCondition(p['warmup']), optimize=True)
        except BudgetExpired:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        del warm
    import gc
    gc.collect()
    random.seed(a.seed); np.random.seed(a.seed)
    history, paths = [], {}
    class Recorder(BidirectionalRRTstar):
        def update_results_tracking(self, cost, path):
            super().update_results_tracking(cost, path)
            elapsed = time.perf_counter() - start
            if path and self.env.is_terminal_mode(path[-1].mode) and elapsed <= p['seconds']:
                if not history or cost < history[-1]['cost']:
                    history.append(dict(seconds=elapsed, cost=float(cost)))
                    saved = [dict(q=s.q.state().tolist(), mode=list(s.mode.task_ids)) for s in path]
                    paths.setdefault('first', saved); paths['final'] = saved
                if p['mode'] == 'first':
                    raise FirstFound()
    result = dict(scene=a.scene, variant=a.variant, seed=a.seed, status='timeout',
                  history=history, budget_seconds=p['seconds'], metadata=metadata,
                  reference_sha256=sha(reference))
    before = sleep_offset()
    power_guard(p['allow_battery'])
    start = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, p['seconds'])
    try:
        planner = Recorder(env, cfg)
        planner.plan(RuntimeTerminationCondition(p['seconds']), optimize=p['mode'] == 'convergence')
    except (FirstFound, BudgetExpired):
        pass
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result['wall_seconds'] = time.perf_counter() - start
    power_guard(p['allow_battery'], before)
    if history and result['status'] != 'error':
        result['status'] = 'solved'
    dest = a.output / 'cells' / f'{a.scene}-{a.variant}-{a.seed}'
    write(dest / 'paths.json', paths)
    write(dest / 'result.json', result)


def launch(a, action, scene, variant, seed=0):
    cmd = [sys.executable, '-I', str(Path(__file__).resolve()), '--worker',
           '--output', str(a.output), '--action', action, '--scene', scene,
           '--variant', variant, '--seed', str(seed)]
    logfile = a.output / 'logs' / f'{action}-{scene}-{variant}-{seed}.log'
    logfile.parent.mkdir(exist_ok=True)
    cap = a.setup_timeout + a.seconds + a.warmup + 120
    before, started = sleep_offset(), time.monotonic()
    with logfile.open('a') as log:
        proc = subprocess.Popen(cmd, cwd=a.output / 'sources' / variant,
            env=clean_env(a.output, variant), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                time.sleep(.5)
                power_guard(a.allow_battery, before)
                if time.monotonic() - started > cap:
                    raise RuntimeError(f'Worker watchdog expired; see {logfile}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    if proc.returncode:
        raise RuntimeError(f'{action} failed: {scene}/{variant}; see {logfile}')


def summarize(output):
    rows = [read(p) for p in sorted((output / 'cells').glob('*/result.json'))]
    summary = []
    for scene in dict.fromkeys(r['scene'] for r in rows):
        for variant in ('paper', 'current'):
            group = [r for r in rows if r['scene'] == scene and r['variant'] == variant]
            solved = [r for r in group if r['status'] == 'solved']
            # Errors are not timeouts, and successful-only medians are biased.
            times = [r['history'][0]['seconds'] if r['status'] == 'solved' else math.inf for r in group]
            median = statistics.median(times) if times else math.inf
            errors = sum(r['status'] == 'error' for r in group)
            summary.append(dict(scene=scene, variant=variant, attempted=len(group), solved=len(solved), errors=errors,
                median_first_s=median if math.isfinite(median) and not errors else None,
                median_censored=bool(group) and not math.isfinite(median) and not errors,
                median_final_cost_solved_only=statistics.median(r['history'][-1]['cost'] for r in solved) if solved else None))
    write(output / 'summary.json', summary)
    if summary:
        with (output / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline-commit', '--baseline', dest='baseline',
                        default='wafr_final_exp', metavar='COMMIT',
                        help='Baseline commit hash or Git ref (default: wafr_final_exp)')
    parser.add_argument('--current-commit', '--current', dest='current',
                        default='HEAD', metavar='COMMIT',
                        help='Comparison commit hash or Git ref (default: HEAD)')
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--setup-timeout', type=float, default=600)
    parser.add_argument('--warmup', type=float, default=1)
    parser.add_argument('--mode', choices=['first', 'convergence'], default='first')
    parser.add_argument('--scenes', nargs='+', choices=list(SCENES), default=list(SCENES))
    parser.add_argument('--output', type=Path, default=Path('out/revision-comparison'))
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Export revisions and check scenes; no planning')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-battery', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    for key in ('action', 'scene', 'variant'):
        parser.add_argument('--' + key, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    a = parser.parse_args()
    a.output = a.output.resolve(); a.repo = a.repo.resolve()
    if a.worker:
        worker(a); return
    if a.runs < 1 or min(a.seconds, a.setup_timeout) <= 0 or a.warmup < 0:
        parser.error('Runs/timeouts must be positive, warmup nonnegative')
    if a.run and a.prepare_only:
        parser.error('Choose --run or --prepare-only')
    revisions = {k: git(a.repo, 'rev-parse', '--verify', ref + '^{commit}')
                 for k, ref in [('paper', a.baseline), ('current', a.current)]}
    print(json.dumps(revisions, indent=2))
    total = 2 * len(a.scenes) * a.runs
    print(f'{total} serial cells; {a.mode}; cap {a.seconds:g}s; worst-case planning {total*a.seconds/3600:.1f}h + setup/warmup')
    print('Committed revisions only; working-tree changes are excluded.')
    if not (a.run or a.prepare_only):
        print('Dry run. Add --prepare-only to validate, or --run to execute.'); return
    power_guard(a.allow_battery)
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = dict(revisions=revisions, scenes=a.scenes, runs=a.runs, seconds=a.seconds,
            setup_timeout=a.setup_timeout, warmup=a.warmup, mode=a.mode,
            allow_battery=a.allow_battery, runner_sha256=sha(Path(__file__)),
            dependencies=dependencies(), python=sys.version, executable=sys.executable)
        protocol_file = a.output / 'protocol.json'
        # JSON normalizes tuples from package metadata.
        settings = json.loads(json.dumps(settings))
        if protocol_file.exists():
            protocol = read(protocol_file)
            if not a.resume or protocol['settings'] != settings:
                raise RuntimeError('Use --resume with unchanged revisions/settings/dependencies, or a new output')
            for variant, manifest in protocol['snapshots'].items():
                for name, digest in manifest.items():
                    if sha(a.output / 'sources' / variant / name) != digest:
                        raise RuntimeError(f'Snapshot modified: {variant}/{name}')
            for name, digest in protocol.get('scene_hashes', {}).items():
                if sha(a.output / 'scenes' / name) != digest:
                    raise RuntimeError(f'Frozen scene modified: {name}')
        else:
            manifests = {v: snapshot(a.repo, rev, a.output / 'sources' / v) for v, rev in revisions.items()}
            protocol = dict(settings=settings, snapshots=manifests,
                created=datetime.datetime.now().astimezone().isoformat(), platform=platform.platform(),
                machine=platform.uname()._asdict(), cpu_count=os.cpu_count(),
                timing='perf_counter, planner construction included; scene setup/import/warmup excluded; first path before shortcutting',
                backend='Same installed robotic/RAI dependency stack, no Coal/preload/adapters',
                scene_source='Tagged scene builders/assets for both; selected revision runtime and goal/configuration types',
                comparison='End-to-end implementation defaults, not isolated algorithm or historical hardware reproduction')
            write(protocol_file, protocol)
        for scene in a.scenes:
            for variant in ('paper', 'current'):
                print(f'Preflight {scene}/{variant}', flush=True)
                launch(a, 'prepare', scene, variant)
        protocol['scene_hashes'] = {p.name: sha(p) for p in (a.output / 'scenes').glob('*.json')}
        write(protocol_file, protocol)
        if a.prepare_only:
            print('Preflight complete. Repeat with --run --resume to execute.'); return
        try:
            for rep in range(a.runs):
                for index, scene in enumerate(a.scenes):
                    order = ('paper', 'current') if (rep + index) % 2 == 0 else ('current', 'paper')
                    for variant in order:
                        seed = SCENES[scene] + rep
                        dest = a.output / 'cells' / f'{scene}-{variant}-{seed}' / 'result.json'
                        if dest.exists():
                            continue
                        print(f'Run {scene}/{variant}/seed={seed}', flush=True)
                        launch(a, 'run', scene, variant, seed)
                        rows = summarize(a.output)
                        print(f'{len(rows)}/{total} finished', flush=True)
        finally:
            rows = summarize(a.output)
        if any(r['status'] == 'error' for r in rows):
            raise SystemExit('Completed with planner errors; retained in cells and summary, not scored as solutions')
        print(f'Complete: {a.output}')


if __name__ == '__main__':
    main()
