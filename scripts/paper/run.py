#!/usr/bin/env python3
"""Run BiRRT* on frozen paper inputs; dry-run by default.

The installed dependency stack and --source checkout supply the implementation.
The protocol pins goals, geometry, settings and reported reference values. No IK
or scene generation occurs. Inputs and protocols ship in benchmarks/paper/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARK = ROOT / 'benchmarks/paper'
PACKAGE = 'multi_robot_multi_goal_planning'
DONE = {'solved', 'timeout'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(source):
    files = sorted((source / PACKAGE).rglob('*.py'))
    if not files:
        raise ValueError('--source must contain the multi_robot_multi_goal_planning package')
    return {str(p.relative_to(source)): sha(p) for p in files}


def check_inputs(inputs, scenes, source):
    for scene in scenes:
        folder = inputs / scene['scene']
        if sha(folder / 'manifest.json') != scene['manifest_sha256']:
            raise ValueError('Wrong frozen scene: ' + scene['scene'])
        for name, expected in read(folder / 'manifest.json')['files'].items():
            if Path(name).is_absolute() or not (folder/name).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe filename in input manifest')
            if sha(folder / name) != expected:
                raise ValueError(f'Frozen input changed: {folder / name}')
        assets = source / PACKAGE / 'assets'
        for name, expected in read(folder/'manifest.json').get('package_assets', {}).items():
            if Path(name).is_absolute() or not (assets/name).resolve().is_relative_to(assets.resolve()):
                raise ValueError('Unsafe package asset filename')
            if sha(assets/name) != expected:
                raise ValueError(f'Package asset differs from frozen benchmark: {name}')


def mains():
    supplies = list(Path('/sys/class/power_supply').glob('*/online'))
    return any(p.read_text().strip() == '1' for p in supplies) if supplies else None


def suspend_offset():
    if hasattr(time, 'CLOCK_BOOTTIME'):
        return time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()
    return 0.


def power_guard(allow_battery, offset=None):
    if not allow_battery and mains() is False:
        raise RuntimeError('Mains disconnected; use --allow-battery for exploratory runs')
    if offset is not None and suspend_offset() - offset > 1:
        raise RuntimeError('System slept during the cell')


def result_path(output, scene, backend, seed):
    return output / 'convergence' / scene / f'{backend}-s{seed}.json'


def summarize(output, settings):
    rows = []
    for scene in settings['plan']['scenes']:
        for backend in scene['backends']:
            results = [read(p) for seed in settings['plan']['seeds']
                       if (p := result_path(output, scene['scene'], backend, seed)).exists()]
            ready = len(results) == len(settings['plan']['seeds']) and all(r['status'] in DONE for r in results)
            def median(values):
                value = statistics.median(values) if ready else math.inf
                return value if math.isfinite(value) else None
            first = median([r['history'][0]['seconds'] if r.get('history') else math.inf for r in results])
            final = median([r['history'][-1]['cost'] if r.get('history') else math.inf for r in results])
            published = scene.get('published') or {}
            row = dict(scene=scene['scene'], backend=backend, complete=ready,
                       runs=len(results), expected=len(settings['plan']['seeds']),
                       solved=sum(r['status'] == 'solved' for r in results),
                       errors=sum(r['status'] not in DONE for r in results),
                       budget_s=scene['seconds'], median_first_s=first, median_final_cost=final,
                       published_first_s=published.get('first_s'), published_final_cost=published.get('final_cost'),
                       published_budget_s=published.get('budget'),
                       published_to_current_first_ratio=published.get('first_s') / first if first and published.get('first_s') is not None else None)
            for factor in [1., 1.01, 1.05]:
                hits = [next((h['seconds'] for h in r.get('history', [])
                              if published.get('final_cost') is not None and h['cost'] <= factor * published['final_cost']), math.inf) for r in results]
                row[f'median_time_to_{factor:g}x_published_cost_s'] = median(hits)
            rows.append(row)
    write(output / 'summary.json', rows)
    if rows:
        with (output / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    export_plot_data(output, settings['plan'])
    return rows


def export_plot_data(output, plan):
    """Export complete scenes in run_experiment's format for analysis/make_plots.py."""
    for scene in plan['scenes']:
        runs = {}
        for backend in scene['backends']:
            for seed in sorted(plan['seeds']):
                file = result_path(output, scene['scene'], backend, seed)
                if file.exists():
                    runs[backend, seed] = read(file)
        target = output/'analysis'/scene['scene']
        if (len(runs) != len(scene['backends'])*len(plan['seeds'])
                or any(r['status'] not in DONE for r in runs.values())):
            continue
        write(target/'config.json', dict(environment=scene['scene'],
              num_runs=len(plan['seeds']), max_planning_time=scene['seconds']))
        for backend in scene['backends']:
            times, costs = [], []
            for seed in sorted(plan['seeds']):
                history = [h for h in runs[backend, seed].get('history', [])
                           if h['seconds'] <= scene['seconds']]
                t = [h['seconds'] for h in history]
                c = [h['cost'] for h in history]
                if not t or t[-1] < scene['seconds']:
                    t.append(scene['seconds']); c.append(c[-1] if c else math.inf)
                times.append(''.join(f'{v},' for v in t) + '\n')
                costs.append(''.join(f'{v},' for v in c) + '\n')
                (target/backend/str(seed)).mkdir(parents=True, exist_ok=True)
            (target/backend/'timestamps.txt').write_text(''.join(times))
            (target/backend/'costs.txt').write_text(''.join(costs))


class Deadline(BaseException):
    pass


def alarm(*_):
    raise Deadline()


def expired(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, Deadline):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def validate(env, path):
    """Use the planner's resolution and compact transition-path convention."""
    import numpy as np
    errors = dict(states=[], edges=[], bounds=[], transitions=[], terminal=False)
    for i, state in enumerate(path):
        if not env.is_collision_free(state.q, state.mode): errors['states'].append(i)
        q = state.q.state()
        if np.any(q < env.limits[0] - 1e-8) or np.any(q > env.limits[1] + 1e-8): errors['bounds'].append(i)
    for i, (left, right) in enumerate(zip(path, path[1:])):
        if left.mode != right.mode:
            if not env.is_collision_free(left.q, right.mode): errors['states'].append(i)
            if not env.is_transition(left.q, left.mode): errors['transitions'].append(i)
        if not env.is_edge_collision_free(left.q, right.q, right.mode): errors['edges'].append(i)
    errors['terminal'] = bool(env.done(path[-1].q, path[-1].mode))
    return errors


def worker(a):
    # Imports stay out of the coordinator; each cell has its own process and RNG.
    import copy
    import gc
    import random
    import numpy as np
    import robotic as ry
    import multi_robot_multi_goal_planning as package
    if Path(package.__file__).resolve().parent != (a.source / PACKAGE).resolve():
        raise RuntimeError('Imported planner is not the selected source checkout')
    from frozen_inputs import build_env, convert_env
    from multi_robot_multi_goal_planning.planners import BidirectionalRRTstar, BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.termination_conditions import RuntimeTerminationCondition
    settings = read(a.output / 'protocol.json')
    if fingerprint(a.source) != settings['source_files']:
        raise RuntimeError('Source changed after campaign preparation')
    if any(sha(HERE/name) != expected for name, expected in settings['scripts'].items()):
        raise RuntimeError('Runner changed after campaign preparation')
    scene = next(s for s in settings['plan']['scenes'] if s['scene'] == a.scene)
    check_inputs(a.inputs, [scene], a.source)
    ry.params_add({'disableGui': True})
    signal.signal(signal.SIGALRM, alarm)
    result = dict(scene=a.scene, backend=a.backend, seed=a.seed, budget_s=scene['seconds'],
                  status='error', history=[], input_sha256=scene['manifest_sha256'],
                  started_epoch=time.time())
    out = result_path(a.output, a.scene, a.backend, a.seed)
    if a.check: out = a.output / 'checks' / a.scene / f'{a.backend}.json'
    try:
        signal.setitimer(signal.ITIMER_REAL, settings['setup_timeout'])
        random.seed(scene['source_seed']); np.random.seed(scene['source_seed'])
        started = time.perf_counter()
        base, manifest = build_env(a.inputs, a.scene)
        def convert():
            return convert_env(base, a.backend, manifest)
        env = convert()
        cfg = BaseRRTConfig(**(settings['plan']['planner_config'] | scene.get('planner_overrides', {})))
        result['config'] = dataclasses.asdict(cfg)
        if not env.is_collision_free(env.start_pos, env.start_mode):
            raise ValueError('Invalid start configuration')
        for task in base.tasks:
            indices = np.concatenate([base.robot_idx[r] for r in task.robots])
            goals = getattr(task.goal, 'goals', [getattr(task.goal, 'goal', None)])
            for goal in goals:
                if goal is None or np.any(goal < base.limits[0, indices]-1e-8) or np.any(goal > base.limits[1, indices]+1e-8):
                    raise ValueError('Invalid frozen goal')
        result['setup_s'] = time.perf_counter() - started
        if a.check:
            result['status'] = 'ready'
            return
        # Compile kernels on a disposable environment before starting the clock.
        if settings['warmup_s']:
            warm = convert()
            random.seed(991); np.random.seed(991)
            warm_planner = BidirectionalRRTstar(warm, copy.deepcopy(cfg))
            signal.setitimer(signal.ITIMER_REAL, settings['warmup_s'] + 1.)
            try: warm_planner.plan(RuntimeTerminationCondition(settings['warmup_s']), optimize=True)
            except BaseException as exc:
                if not expired(exc): raise
            finally: signal.setitimer(signal.ITIMER_REAL, 0)
            del warm_planner, warm
            gc.collect()
        random.seed(a.seed); np.random.seed(a.seed)
        if hasattr(env, '_make_uniform_sampler'): env._uniform_sampler = env._make_uniform_sampler()
        paths = {}
        history = result['history']
        progress = out.with_suffix('.progress.jsonl').open('w', buffering=1)
        class Record(BidirectionalRRTstar):
            def save_tree_data(self): pass
            def update_results_tracking(self, cost, path):
                super().update_results_tracking(cost, path)
                elapsed = time.perf_counter() - t0
                if path and self.env.is_terminal_mode(path[-1].mode) and elapsed <= scene['seconds'] and np.isfinite(cost):
                    if not history or cost < history[-1]['cost']:
                        history.append(dict(seconds=elapsed, cost=float(cost)))
                        progress.write(json.dumps(history[-1]) + '\n')
                        paths.setdefault('first', list(path)); paths['final'] = list(path)
        power_guard(settings['allow_battery'])
        offset = suspend_offset()
        t0 = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, scene['seconds'])
        planner = None
        try:
            planner = Record(env, cfg)
            planner.plan(RuntimeTerminationCondition(scene['seconds']), optimize=True)
        except BaseException as exc:
            if not expired(exc): raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            progress.close()
        result['shortcut_allocation_stats'] = getattr(getattr(planner, 'shortcut_allocator', None), 'stats', None)
        result['wall_s'] = time.perf_counter() - t0
        power_guard(settings['allow_battery'], offset)
        result['status'] = 'solved' if history else 'timeout'
        result['validation'] = {}
        signal.setitimer(signal.ITIMER_REAL, settings['setup_timeout'])
        for name, path in paths.items():
            audit = validate(env, path)
            result['validation'][name] = audit
            write(out.with_name(out.stem + f'-{name}-path.json'),
                  [dict(q=s.q.state().tolist(), task_ids=[int(x) for x in s.mode.task_ids]) for s in path])
            if any(audit[k] for k in ['states','edges','bounds','transitions']) or not audit['terminal']:
                result['status'] = 'invalid_path'
    except BaseException:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['finished_epoch'] = time.time()
        write(out, result)
    if result['status'] not in DONE | {'ready'}:
        raise SystemExit('Cell failed; inspect ' + str(out))


def merge_protocols(paths):
    plans = [read(path) for path in paths]
    plan = plans[0]
    if any(p.get('format') != 'paper-comparison-v1' for p in plans):
        raise ValueError('Unsupported protocol format')
    for extra in plans[1:]:
        if any(extra[k] != plan[k] for k in ('seeds', 'planner_config')):
            raise ValueError('Combined protocols must have the same seeds and planner settings')
        plan['scenes'].extend(extra['scenes'])
    names = [s['scene'] for s in plan['scenes']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate scene in selected protocols')
    if any(Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('Invalid scene identifier')
    return plan


def backend_fingerprints():
    import importlib.util
    result = {}
    for name in ('robotic', 'comotion', 'mujoco'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            continue
        base = Path(spec.origin).parent
        files = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py', '.so'))
        result[name] = dict(origin=str(base), files={str(p.relative_to(base)): sha(p) for p in files})
    return result


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', nargs='+', type=Path, default=[BENCHMARK/'protocols/paper_fixed_sequence.json'])
    p.add_argument('--inputs', type=Path, default=BENCHMARK/'inputs')
    p.add_argument('--output', type=Path, default=ROOT/'out/paper-comparison')
    p.add_argument('--source', type=Path, default=ROOT/'src', help='Directory containing the planner package')
    p.add_argument('--python', default=sys.executable)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--check', action='store_true', help='Load/convert/start-check only; no planning or warmup')
    actions.add_argument('--summarize', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='Archive and retry failed/interrupted cells; requires --resume')
    p.add_argument('--max-hours', type=float, help='Wall-clock limit for this invocation, including setup and validation')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--scenes', nargs='+')
    p.add_argument('--backends', nargs='+', choices=['rai','comotion','mujoco'])
    p.add_argument('--seconds', type=float, help='Override all scene budgets')
    p.add_argument('--workers', type=int, default=1, help='Independent processes; serial by default')
    p.add_argument('--warmup-s', type=float, default=1.)
    p.add_argument('--setup-timeout', type=float, default=240.)
    p.add_argument('--allow-battery', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--backend-fingerprints', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--scene', help=argparse.SUPPRESS)
    p.add_argument('--backend', help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.backend_fingerprints:
        print(json.dumps(backend_fingerprints())); return
    a.output = a.output.resolve(); a.source = a.source.resolve()
    if a.inputs: a.inputs = a.inputs.resolve()
    if a.worker: worker(a); return
    if a.summarize:
        summarize(a.output, read(a.output/'protocol.json')); return
    plan = merge_protocols(a.protocol)
    if a.retry_failed and not a.resume: p.error('--retry-failed requires --resume')
    if a.max_hours is not None and (not math.isfinite(a.max_hours) or a.max_hours <= 0): p.error('--max-hours must be positive')
    deadline = time.monotonic() + a.max_hours * 3600 if a.max_hours is not None else math.inf
    if any((parent/'_config.yml').exists() and (parent/'_posts').is_dir() for parent in [a.output, *a.output.parents]):
        p.error('Choose an output directory outside a Jekyll website')
    if a.scenes:
        if set(a.scenes) - {s['scene'] for s in plan['scenes']}: p.error('Unknown scene')
        plan['scenes'] = [s for s in plan['scenes'] if s['scene'] in a.scenes]
    if a.seeds is not None: plan['seeds'] = a.seeds
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']): p.error('Seeds must be nonempty and unique')
    if a.workers < 1 or a.setup_timeout <= 0 or a.warmup_s < 0: p.error('Invalid worker count/timeout')
    for s in plan['scenes']:
        if a.seconds is not None: s['seconds'] = a.seconds
        if not math.isfinite(s['seconds']) or s['seconds'] <= 0: p.error('Budgets must be finite and positive')
        if a.backends:
            # Intersection retains the protocol's explicitly unsupported backend exclusions.
            s['backends'] = [b for b in s['backends'] if b in a.backends]
    plan['scenes'] = [s for s in plan['scenes'] if s['backends']]
    if not plan['scenes']: p.error('No supported cells selected')
    if not a.inputs: p.error('--inputs is required')
    check_inputs(a.inputs, plan['scenes'], a.source)
    count = sum(len(s['backends'])*len(plan['seeds']) for s in plan['scenes'])
    seconds = sum(s['seconds']*len(s['backends'])*len(plan['seeds']) for s in plan['scenes'])
    print(f'{count} cells; {seconds/3600:.2f} planning hours; {a.workers} worker(s).')
    print('Same archived inputs; published time ratios still include hardware/budget differences.')
    if not (a.run or a.check):
        print('DRY RUN. Use --check for setup validation or --run for experiments.'); return
    if a.run: power_guard(a.allow_battery)
    # Refuse concurrent coordinators and source/settings changes on resume.
    import fcntl
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy()
        for key in list(env):
            if key in ('PYTHONPATH','LD_PRELOAD') or key.startswith(('MR_COAL','MR_FCL','MR_MIDPHASE')): env.pop(key)
        env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        env.update(PYTHONPATH=str(a.source), PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR=str(a.output/'mpl'), NUMBA_CACHE_DIR=str(a.output/'numba'))
        for name in ['mpl','numba','logs']: (a.output/name).mkdir(exist_ok=True)
        # Query the selected interpreter, not the coordinator's dependency stack.
        probe = ('import importlib.metadata as m,json,pathlib,hashlib,sys; '
                 'd=list(m.distributions()); '
                 'native={str(f):hashlib.sha256(pathlib.Path(x.locate_file(f)).read_bytes()).hexdigest() '
                 'for x in d if any(t in x.metadata["Name"].lower() for t in ["comotion","robotic","mujoco"]) '
                 'for f in (x.files or []) if str(f).endswith(".so")}; '
                 'print(json.dumps({"packages":sorted((x.metadata["Name"],x.version) for x in d),"native":native,"python":sys.version}))')
        dependencies = json.loads(subprocess.check_output([a.python,'-c',probe], env=env, text=True))
        dependencies['imported_backends'] = json.loads(subprocess.check_output([a.python, str(Path(__file__).resolve()), '--backend-fingerprints'], env=env, text=True))
        settings = dict(plan=plan, workers=a.workers, warmup_s=a.warmup_s, setup_timeout=a.setup_timeout,
                        allow_battery=a.allow_battery, source_files=fingerprint(a.source), dependencies=dependencies,
                        scripts={f.name:sha(f) for f in [Path(__file__),HERE/'frozen_inputs.py']},
                        machine=dict(platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
                                     cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None),
                        timing='Planner construction included; imports/setup/warmup/validation excluded; shortcutting and informed sampling enabled')
        # CPU MHz changes continuously: preserve hardware identity, not instantaneous frequency.
        if settings['machine']['cpuinfo']:
            settings['machine']['cpuinfo']='\n'.join(x for x in settings['machine']['cpuinfo'].splitlines()
                                                    if not x.startswith(('cpu MHz','bogomips')))
        try:
            settings['git_commit'] = subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            settings['git_commit'] = None
        file = a.output/'protocol.json'
        if file.exists():
            if not a.resume or read(file) != settings: raise ValueError('Use --resume with unchanged settings/source/hardware, or a new output')
        else: write(file, settings)
        cells=[]
        for index,s in enumerate(plan['scenes']):
            for seed in ([plan['seeds'][0]] if a.check else plan['seeds']):
                order=s['backends']; offset=(index+seed)%len(order)
                cells.extend((s,b,seed) for b in order[offset:]+order[:offset])
        stopping = threading.Event()
        processes = set()
        process_lock = threading.Lock()
        def launch(cell):
            scene, backend, seed = cell
            dest = result_path(a.output, scene['scene'], backend, seed)
            label = f"{scene['scene']}/{backend}/{seed}"
            if not a.check and dest.exists():
                status = read(dest)['status']
                if status in DONE:
                    return 'REUSED ' + label
                if not a.retry_failed:
                    raise RuntimeError(f'Existing {status} cell needs review: {dest}; use --resume --retry-failed')
                archive = a.output/'attempts'/scene['scene']/f'{backend}-s{seed}'/str(time.time_ns())
                archive.mkdir(parents=True)
                for previous in dest.parent.glob(dest.stem + '*'):
                    shutil.move(str(previous), archive/previous.name)
            remaining = deadline - time.monotonic()
            if stopping.is_set() or remaining < (0 if a.check else scene['seconds'] + a.warmup_s + 5):
                stopping.set()
                return 'DEFERRED ' + label
            if not a.check:
                try:
                    power_guard(a.allow_battery)
                except RuntimeError:
                    stopping.set()
                    raise
            cmd = [a.python, str(Path(__file__).resolve()), '--worker', '--inputs', str(a.inputs),
                   '--source', str(a.source), '--output', str(a.output), '--scene', scene['scene'],
                   '--backend', backend, '--seed', str(seed)]
            if a.check:
                cmd.append('--check')
            logfile = a.output/'logs'/f"{scene['scene']}-{backend}-{seed}{'-check' if a.check else ''}.log"
            if not a.check:
                write(dest, dict(scene=scene['scene'], backend=backend, seed=seed, budget_s=scene['seconds'],
                                 status='running', history=[], started_epoch=time.time()))
            timeout = min(remaining, 2*a.setup_timeout + scene['seconds'] + a.warmup_s + 30)
            with logfile.open('w') as log:
                with process_lock:
                    if stopping.is_set():
                        return 'DEFERRED ' + label
                    process = subprocess.Popen(cmd, env=env, cwd=a.output, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    processes.add(process)
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    stop_process(process)
                    returncode = process.returncode
                    if time.monotonic() >= deadline:
                        stopping.set()
                finally:
                    with process_lock:
                        processes.discard(process)
            if returncode:
                if not a.check:
                    record = read(dest)
                    if record['status'] == 'running':
                        record.update(status='interrupted', returncode=returncode, finished_epoch=time.time())
                        progress = dest.with_suffix('.progress.jsonl')
                        if progress.exists():
                            for line in progress.read_text().splitlines():
                                try: record['history'].append(json.loads(line))
                                except json.JSONDecodeError: break
                        write(dest, record)
                raise RuntimeError(f'Worker exited {returncode}: {label}; see {logfile}')
            return 'DONE ' + label
        errors = []
        pool = ThreadPoolExecutor(max_workers=a.workers)
        try:
            futures = {pool.submit(launch, cell): cell for cell in cells}
            for future in as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    errors.append(str(exc))
                    print('ERROR', str(exc), flush=True)
                if not a.check:
                    summarize(a.output, settings)
        except BaseException:
            stopping.set()
            with process_lock:
                for process in list(processes):
                    stop_process(process)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if stopping.is_set():
            print('Stopped; completed cells are preserved. Resume with the same protocol.')
        if errors:
            raise SystemExit(f'{len(errors)} failed cells; inspect logs. No failed cells are scored.')
        if not stopping.is_set():
            print('CHECKED (no planning)' if a.check else 'COMPLETE; see summary.csv and convergence/')


if __name__ == '__main__':
    main()
