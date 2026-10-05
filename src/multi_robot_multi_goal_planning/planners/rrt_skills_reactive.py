import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import itertools
import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import BaseProblem, Mode, Task

from .rrt_skills import RRTSkills, RRTSkillsConfig
from .termination_conditions import IterationTerminationCondition, RuntimeTerminationCondition

DONE = -1
_MASK_BIN_SAMPLES_CAP = 8

@dataclass
class ReactiveRoadmapConfig:
    """Configuration parameters for the roadmap and nature set construction"""
    rrg_runtime: float = 60
    rrg_iters: Optional[int] = None
    inactive_n_samples: int = 1500
    inactive_max_vertices: Optional[int] = None
    inactive_memory_fraction: float = 0.25
    inactive_sample_budget: Optional[int] = None
    inactive_n_samples_min: int = 800
    inactive_goal_bias: float = 0.1
    inactive_max_vel: float = 2.0
    rrg_kinodynamic_steps: int = 5
    rrg_inactive_transition_source: str = "uniform_random"
    inactive_k_max: int = 15
    inactive_dedup_frac: float = 1e-3
    build_shortcut: bool = True
    n_mc: int = 100
    decision_frequency: float = 10.0
    nature_state: str = "position_time"
    bin_tol: Optional[float] = None
    bin_tol_floor: float = 0.05
    joint_entry_states_per_component: int = 32
    inactive_roadmap_structure: str = "auto"
    inactive_lattice_max_dim: int = 2
    inactive_lattice_max_vertices: int = 40000
    inactive_speed: Optional[float] = None
    inactive_reach_epochs: int = 1
    inactive_connect_epochs: int = 1
    seed: int = 0

def active_skill_tasks(env: BaseProblem, mode: Mode) -> List[Task]:
    """
    Returns all tasks currently running an active skill in the given mode. If the mode is terminal
    or none of the active tasks have a skill attached, returns an empty list
    """
    if env.is_terminal_mode(mode):
        return []
    return [env.tasks[t] for t in dict.fromkeys(mode.task_ids)
            if getattr(env.tasks[t], "skill", None) is not None]

def inactive_robots(env: BaseProblem, active_tasks: Sequence[Task]) -> List[str]:
    """
    Returns the names of all robots that are free to move (not executing a skill) These are the
    decision variables that will navigate the inactive roadmap
    """
    active = {r for task in active_tasks for r in task.robots}
    return [r for r in env.robots if r not in active]

def subspace_indices(env: BaseProblem, robots: Sequence[str]) -> np.ndarray:
    """
    Returns the slice of the full configuration vector q that belongs to the given robots Used to
    factorize full states q -> (active_q, inactive_q)
    """
    indices, offset = [], 0
    for robot in env.robots:
        dim = env.robot_dims[robot]
        if robot in robots:
            indices.extend(range(offset, offset + dim))
        offset += dim
    return np.asarray(indices, dtype=int)

def decision_epoch_indices(n_steps: int, dt: float, decision_frequency: float) -> List[int]:
    """Micro-step index observed at each decision epoch t_k = k / decision_frequency"""
    period = 1.0 / decision_frequency
    horizon = (n_steps - 1) * dt
    times = np.arange(0.0, horizon + 1e-12, period)
    indices = np.unique(np.clip(np.rint(times / dt).astype(int), 0, n_steps - 1)).tolist()
    if indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)
    return indices

def reachable_from(adjacency, sources: Sequence[int]) -> List[int]:
    """Vertices reachable from `sources` in an undirected adjacency list, as sorted indices"""
    seen = set(int(v) for v in sources)
    queue = deque(seen)
    while queue:
        for neighbour, _ in adjacency[queue.popleft()]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return sorted(seen)

def component_labels(adjacency) -> Tuple[List[int], int]:
    """Connected component membership for an undirected adjacency list"""
    labels, count = [-1] * len(adjacency), 0
    for start in range(len(adjacency)):
        if labels[start] >= 0:
            continue
        for vertex in reachable_from(adjacency, [start]):
            labels[vertex] = count
        count += 1
    return labels, count

class ExitBranch(NamedTuple):
    """
    One mutually exclusive way a multi-skill mode can end, seen from one joint nature state. A joint
    transition advances EVERY component by one decision epoch, so a branch is named by which
    components hit DONE on that step; the ones that did not are still running and carry their new
    states in `remaining`
    """
    prob: float
    finished: Tuple[str, ...]
    active_config: np.ndarray
    remaining: Tuple[int, ...]

@dataclass
class NatureSet:
    """"Discrete" stochastic model (Markov Chain) of the active robot's skill"""
    keys: List[Tuple[int, ...]]
    configs: np.ndarray
    kernel: List[List[Tuple[int, float]]]
    initial: int
    exits: Dict[int, np.ndarray]
    visits: np.ndarray
    bin_tol: float
    rollout_lengths: np.ndarray
    epochs: np.ndarray
    disp: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    index: Optional[Dict[Tuple[int, ...], int]] = None
    exit_branches: Optional[Dict[int, List["ExitBranch"]]] = None
    components: Optional[List[str]] = None
    samples: Optional[List[np.ndarray]] = None

    @property
    def n_states(self) -> int:
        return len(self.keys)

    @property
    def is_joint(self) -> bool:
        return self.exit_branches is not None

    @property
    def time_indexed(self) -> bool:
        return bool(np.any(self.epochs >= 0))

def compute_grid_resolution(observations, config: ReactiveRoadmapConfig) -> float:
    """Computes spatial quantization threshold (bin_tol)"""
    steps = np.concatenate(
        [np.linalg.norm(np.diff(np.stack(o), axis=0), axis=1) for o in observations if len(o) > 1]
    )
    positive = steps[steps > 1e-9]
    derived = max(1e-9, 0.5 * float(np.median(positive))) if len(positive) else 0.05
    return config.bin_tol if config.bin_tol is not None else max(derived, config.bin_tol_floor)

def quantize_to_nature_key(q: np.ndarray, bin_tol: float) -> Tuple[int, ...]:
    """Maps continuous active-robot spatial coordinates into discrete integer grid indices"""
    return tuple(np.round(q / bin_tol).astype(int).tolist())

def nature_key(q: np.ndarray, epoch: int, bin_tol: float, time_indexed: bool) -> Tuple[int, ...]:
    """Builds the full discrete nature-state key"""
    key = quantize_to_nature_key(q, bin_tol)
    return key + (int(epoch),) if time_indexed else key

def build_nature_set(env: BaseProblem, task: Task, config: ReactiveRoadmapConfig) -> NatureSet:
    """Constructs a discrete Markov transition model for a stochastic skill"""
    q_init = np.asarray(task.initiation_goal.sample(None), dtype=np.float64)
    trajectories = [
        np.asarray(task.skill.rollout(q_init, task, env.get_joint_names(), env, t0=0.0).trajectory,
                   dtype=np.float64)
        for _ in range(config.n_mc)
    ]
    observations = [[t[i] for i in decision_epoch_indices(len(t), task.skill.dt, config.decision_frequency)]
                    for t in trajectories]
    bin_tol = compute_grid_resolution(observations, config)
    time_indexed = config.nature_state == "position_time"
    key_of = lambda q, epoch: nature_key(q, epoch, bin_tol, time_indexed)
    keys, index, epochs, visits = [], {}, [], []
    counts = defaultdict(lambda: defaultdict(int))
    exit_sums, exit_counts = {}, defaultdict(int)
    bin_index, bin_sums, bin_visits, bin_of = {}, [], [], []
    bin_obs = defaultdict(list)

    def state_of(q: np.ndarray, epoch: int) -> int:
        """
        Maps a continuous configuration q observed at a decision epoch to a unique state index and
        accumulates stats
        """
        spatial = quantize_to_nature_key(q, bin_tol)
        if spatial not in bin_index:
            bin_index[spatial] = len(bin_sums)
            bin_sums.append(np.zeros_like(q))
            bin_visits.append(0)
        b = bin_index[spatial]
        bin_sums[b] += q
        bin_visits[b] += 1
        bin_obs[b].append(q)

        key = key_of(q, epoch)
        if key not in index:
            index[key] = len(keys)
            keys.append(key)
            epochs.append(epoch if time_indexed else -1)
            visits.append(0)
            bin_of.append(b)
        i = index[key]
        visits[i] += 1
        return i

    active_slices = inactive_robot_slices(env, task.robots)
    disp_sums = defaultdict(lambda: np.zeros(len(active_slices)))
    disp_counts = defaultdict(int)

    for rollout in observations:
        i_prev, q_prev = None, None
        for epoch, q in enumerate(rollout):
            i = state_of(q, epoch)
            if i_prev is not None:
                counts[i_prev][i] += 1
                step = np.asarray([float(np.linalg.norm((q - q_prev)[a:b]))
                                   for a, b in active_slices])
                disp_sums[(i_prev, i)] += step
                disp_counts[(i_prev, i)] += 1
            i_prev, q_prev = i, q
        q_exit = rollout[-1]
        counts[i_prev][DONE] += 1
        exit_sums[i_prev] = exit_sums.get(i_prev, np.zeros_like(q_exit)) + q_exit
        exit_counts[i_prev] += 1

    kernel = [
        [(j, c / sum(counts[i].values())) for j, c in sorted(counts[i].items())]
        for i in range(len(keys))
    ]
    bin_means = np.stack([bin_sums[b] / bin_visits[b] for b in range(len(bin_sums))])
    bin_repr = np.stack([
        min(bin_obs[b], key=lambda q, m=bin_means[b]: float(np.dot(q - m, q - m)))
        for b in range(len(bin_sums))
    ])
    bin_samples = []
    for b in range(len(bin_sums)):
        obs = np.asarray(bin_obs[b], dtype=np.float64)
        chosen = [int(np.argmin(np.linalg.norm(obs - bin_repr[b], axis=1)))]
        d = np.linalg.norm(obs - obs[chosen[0]], axis=1)
        while len(chosen) < min(_MASK_BIN_SAMPLES_CAP, len(obs)):
            nxt = int(np.argmax(d))
            if d[nxt] <= 0.0:
                break
            chosen.append(nxt)
            d = np.minimum(d, np.linalg.norm(obs - obs[nxt], axis=1))
        bin_samples.append(obs[chosen])
    return NatureSet(
        keys=keys,
        configs=bin_repr[np.asarray(bin_of, dtype=int)],
        samples=[bin_samples[b] for b in np.asarray(bin_of, dtype=int)],
        kernel=kernel,
        initial=index[key_of(observations[0][0], 0)],
        exits={i: exit_sums[i] / exit_counts[i] for i in exit_sums},
        visits=np.asarray(visits, dtype=int),
        bin_tol=bin_tol,
        rollout_lengths=np.asarray([len(t) for t in trajectories], dtype=int),
        epochs=np.asarray(epochs, dtype=int),
        disp={k: disp_sums[k] / disp_counts[k] for k in disp_sums},
    )

def product_nature_set(natures: Sequence[NatureSet], task_names: Sequence[str],
                       entries: Sequence[Tuple[int, ...]]) -> NatureSet:
    """
    Builds the joint stochastic model of a mode that runs SEVERAL skills at the same time. The joint
    state is the tuple of component states
    """
    K = len(natures)
    if K == 1:
        return natures[0]
    widths = [n.configs.shape[1] for n in natures]
    if not all(n.time_indexed for n in natures):

        raise ValueError("product_nature_set needs nature_state='position_time' on every "
                         "component; the joint model advances all skills in lockstep")

    index: Dict[Tuple[int, ...], int] = {}
    states: List[Tuple[int, ...]] = []

    def state_id(tup: Tuple[int, ...]) -> int:
        if tup not in index:
            index[tup] = len(states)
            states.append(tup)
        return index[tup]

    queue = deque()
    for entry in dict.fromkeys(tuple(int(c) for c in e) for e in entries):
        if entry not in index:
            state_id(entry)
            queue.append(entry)
    if not states:
        raise ValueError("product_nature_set was given no entry states")

    kernel: List[List[Tuple[int, float]]] = []
    branches: Dict[int, List[ExitBranch]] = {}
    exits: Dict[int, np.ndarray] = {}

    while queue:
        tup = queue.popleft()
        j = index[tup]
        while len(kernel) <= j:
            kernel.append([])

        onward: Dict[Tuple[int, ...], float] = {}
        raw_branches: List[Tuple[float, Tuple[str, ...], np.ndarray, Tuple[int, ...]]] = []
        combos: List[Tuple[Tuple[int, ...], float]] = [((), 1.0)]
        for i in range(K):
            nxt = []
            for prefix, p in combos:
                for x, q in natures[i].kernel[tup[i]]:
                    nxt.append((prefix + (int(x),), p * q))
            combos = nxt

        for succ, p in combos:
            if p <= 0.0:
                continue
            finished = tuple(i for i in range(K) if succ[i] == DONE)
            if not finished:
                onward[succ] = onward.get(succ, 0.0) + p
                continue

            active = np.concatenate([
                natures[i].exits[tup[i]] if i in finished else natures[i].configs[succ[i]]
                for i in range(K)])
            raw_branches.append((p, tuple(task_names[i] for i in finished), active, succ))

        total_done = sum(p for p, _, _, _ in raw_branches)
        for succ, p in onward.items():
            if succ not in index:
                state_id(succ)
                queue.append(succ)
            kernel[j].append((index[succ], p))
        if total_done > 0.0:
            kernel[j].append((DONE, total_done))
            branches[j] = [ExitBranch(p / total_done, names, active, rem)
                           for p, names, active, rem in raw_branches]
            exits[j] = sum((p / total_done) * active for p, _, active, _ in raw_branches)
        kernel[j].sort()

    while len(kernel) < len(states):
        kernel.append([])

    configs = np.empty((len(states), sum(widths)), dtype=np.float64)
    keys: List[Tuple[int, ...]] = []
    epochs = np.empty(len(states), dtype=int)
    visits = np.empty(len(states), dtype=float)
    for j, tup in enumerate(states):
        configs[j] = np.concatenate([natures[i].configs[tup[i]] for i in range(K)])
        keys.append(tuple(k for i in range(K) for k in natures[i].keys[tup[i]]))

        epochs[j] = max(int(natures[i].epochs[tup[i]]) for i in range(K))
        visits[j] = float(np.prod([natures[i].visits[tup[i]] for i in range(K)]))

    fresh = tuple(n.initial for n in natures)
    initial = index.get(fresh)
    if initial is None:
        entry_ids = [index[tuple(int(c) for c in e)] for e in entries if tuple(int(c) for c in e) in index]
        initial = max(entry_ids, key=lambda j: visits[j]) if entry_ids else 0

    return NatureSet(
        keys=keys,
        configs=configs,
        kernel=kernel,
        initial=int(initial),
        exits=exits,
        visits=visits.astype(int),
        bin_tol=float(min(n.bin_tol for n in natures)),
        rollout_lengths=np.concatenate([n.rollout_lengths for n in natures]),
        epochs=epochs,
        index=index,
        exit_branches=branches,
        components=list(task_names),
    )

def sample_kernel_trajectory(nature: NatureSet, rng, max_epochs: int = 10000) -> List[int]:
    """Walks the counted kernel from `initial` until DONE, returning the nature states visited"""
    state, visited = nature.initial, [nature.initial]
    for _ in range(max_epochs):
        successors = nature.kernel[state]
        if not successors:
            break
        state = int(rng.choice([j for j, _ in successors], p=[p for _, p in successors]))
        if state == DONE:
            break
        visited.append(state)
    return visited

@dataclass
class InactiveRoadmap:
    """Roadmap graph spanning the joint subspace of inactive robots during a skill mode"""
    mode: Mode
    robots: List[str]
    indices: np.ndarray
    vertices: np.ndarray
    adjacency: List[List[Tuple[int, np.ndarray]]]
    seeds: Dict[str, List[int]] = field(default_factory=dict)
    n_sampled: int = 0
    n_stranded_seeds: int = 0
    max_vertices: Optional[int] = None
    build_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return sum(len(a) for a in self.adjacency) // 2

def inactive_robot_slices(env, robots: Sequence[str]) -> List[Tuple[int, int]]:
    """Offsets of each inactive robot inside the packed inactive subspace vector"""
    slices, offset = [], 0
    for robot in robots:
        slices.append((offset, offset + env.robot_dims[robot]))
        offset += env.robot_dims[robot]
    return slices

def inactive_speed(config: "ReactiveRoadmapConfig", env) -> float:
    """How fast an inactive robot moves inside a skill window, in configuration units per second"""
    speed = config.inactive_speed
    if speed is not None:
        return float(speed)

    return float(config.inactive_max_vel)

def inactive_step_radius(config: "ReactiveRoadmapConfig", env) -> float:
    """
    How far an inactive robot travels in ONE decision epoch, and therefore the maximum length of an
    inactive-roadmap edge: one edge is one action is one epoch
    """
    return inactive_speed(config, env) / config.decision_frequency

def inactive_reach_radius(config: "ReactiveRoadmapConfig", env) -> float:
    """
    How far the roadmap BUILDER may reach in one growth/connection step -- a search radius, not a
    speed, and deliberately several epochs long
    """
    return config.inactive_reach_epochs * inactive_step_radius(config, env)

def inactive_edge_length(diff, slices) -> float:
    """
    Length of an inactive-subspace displacement in the ONLY metric that is consistent with what
    prices it and what executes it: the maximum PER-ROBOT L2 norm
    """
    return max((float(np.linalg.norm(diff[s:e])) for s, e in slices), default=0.0)

def inactive_edge_lengths(deltas, slices) -> np.ndarray:
    """Vectorised `inactive_edge_length` over a stack of displacements, shape (n, d) -> (n,)"""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim == 1:
        deltas = deltas[None, :]
    if not slices:
        return np.zeros(len(deltas))
    return np.max(
        np.stack([np.linalg.norm(deltas[:, s:e], axis=1) for s, e in slices], axis=1), axis=1
    )

def _lattice_p_max(env, robots) -> int:
    """
    Largest per-robot DOF count in the inactive subspace. The lattice's diagonal edge spans
    `h*sqrt(d)` in the joint norm but only `h*sqrt(p_max)` in the per-robot max norm, where p_max is
    the widest single robot -- so `h = step_radius/sqrt(d)` under-steps by `sqrt(d/p_max)`, i.e
    """
    return max((env.robot_dims[r] for r in robots), default=1)

def inactive_feasibility(env, mode, robots, indices, template=None):
    """
    Returns (free, edge_free) for the inactive subspace of a skill mode. Lifted out of
    build_inactive_roadmap so every inactive-roadmap operation uses the same feasibility test
    """
    if template is None:
        template = np.asarray(env.get_start_pos().state(), dtype=np.float64)

    def free(q_sub) -> bool:
        q = template.copy()
        q[indices] = q_sub
        return bool(env.is_collision_free_for_robot(robots, q, mode))

    def edge_free(a, b) -> bool:
        n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / env.collision_resolution)))
        return all(free(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n + 1)[1:-1])

    return free, edge_free

def _available_memory_bytes() -> Optional[float]:
    """MemAvailable from /proc/meminfo, or None where that is not readable"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except Exception:
        return None
    return None

def shape_bounding_radius(shape_type: str, size: np.ndarray) -> float:
    """
    Radius of a sphere centred on the frame origin that contains the whole shape. ||getSize()||_2
    was the original, layout-agnostic answer, and it is sound for every rai type -- but it is
    roughly a factor of two too large on the two types these environments are built from, because a
    size vector holds FULL extents while the shape is centred on its frame
    """
    size = np.asarray(size, dtype=float).ravel()

    if shape_type == "ST.capsule" and size.size >= 2:
        return 0.5 * float(size[0]) + float(size[-1])
    if shape_type == "ST.cylinder" and size.size >= 2:
        return float(np.hypot(0.5 * float(size[0]), float(size[-1])))

    if shape_type in ("ST.box", "ST.ssBox") and size.size >= 3:
        corner = 0.5 * float(np.linalg.norm(size[:3]))
        return corner + (float(size[3]) if size.size > 3 else 0.0)
    if shape_type == "ST.sphere" and size.size >= 1:
        return float(size[0])
    return float(np.linalg.norm(size))

def robot_frame_spheres(env, robots: Sequence[str], q: Optional[np.ndarray] = None):
    """Conservative per-frame bounding spheres for `robots`, as (rai frame handles, radii)"""
    shaped = []
    for name in env.C.getFrameNames():
        frame = env.C.getFrame(name)
        try:
            shape_type = str(frame.getShapeType())
            if shape_type in ("ST.none", "ST.marker"):
                continue
            size = np.asarray(frame.getSize(), dtype=float).ravel()
        except Exception:
            continue
        if size.size == 0:
            continue
        radius = shape_bounding_radius(shape_type, size)
        if radius <= 0.0:
            continue
        shaped.append((name, frame, radius))
    if not shaped:
        return [], np.zeros(0)

    keep = np.array([env._frame_owner(name) in robots for name, _, _ in shaped])

    if q is not None:
        indices = subspace_indices(env, robots)
        if len(indices):
            handles = [f for _, f, _ in shaped]
            q = np.asarray(q, dtype=np.float64)
            env.C.setJointState(q)
            base = np.asarray([f.getPosition() for f in handles], dtype=np.float64)
            for k in indices:
                probe = q.copy()
                probe[k] += 1e-3
                env.C.setJointState(probe)
                moved = np.asarray([f.getPosition() for f in handles], dtype=np.float64)
                keep |= np.linalg.norm(moved - base, axis=1) > 1e-9
            env.C.setJointState(q)

    frames = [f for (name, f, r), k in zip(shaped, keep) if k]
    radii = np.asarray([r for (name, f, r), k in zip(shaped, keep) if k], dtype=np.float64)
    return frames, radii

def frame_positions(env, frames: Sequence, q: np.ndarray) -> np.ndarray:
    """
    World positions of `frames` (handles from robot_frame_spheres) at full configuration q, shape
    (len(frames), 3)
    """
    env.C.setJointState(q)
    return np.asarray([f.getPosition() for f in frames], dtype=np.float64)

def enclosing_sphere(positions: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    One sphere containing all the given frame spheres: centred on their centroid, with a radius that
    reaches the furthest frame surface. Cheap and conservative -- the first (and for a single-shape
    robot, only) level of the broad-phase
    """
    if len(positions) == 0:
        return np.zeros(3), 0.0
    centre = positions.mean(axis=0)
    return centre, float(np.max(np.linalg.norm(positions - centre, axis=1) + radii))

def sample_inactive_task_goals(env, mode, robots, indices) -> List[np.ndarray]:
    """Goal configurations for the tasks of this mode that belong ENTIRELY to inactive robots"""
    goals, inactive = [], set(robots)
    template = np.asarray(env.get_start_pos().state(), dtype=np.float64)
    for task_id in dict.fromkeys(mode.task_ids):
        task = env.tasks[task_id]
        if task.goal is None or not set(task.robots).issubset(inactive):
            continue
        sample = np.asarray(task.goal.sample(mode), dtype=np.float64)
        q, offset = template.copy(), 0
        for robot in task.robots:
            dim = env.robot_dims[robot]
            q[subspace_indices(env, [robot])] = sample[offset : offset + dim]
            offset += dim
        goals.append(q[indices])
    return goals

def _subdivide_edges(vertices, adjacency, array, slices, epoch_step, free, stats=None):
    """
    Cuts every edge longer than `epoch_step` into equal one-epoch pieces, inserting the interior
    points as new vertices on the straight segment
    """
    n0 = len(vertices)
    pairs = {(i, j) for i, nbrs in enumerate(adjacency) for j, _ in nbrs if i < j}
    keep = [[] for _ in range(n0)]
    new_pts: List[np.ndarray] = []
    new_links: List[Tuple[int, int]] = []

    def step_of(a, b):
        d = b - a
        return np.asarray([float(np.linalg.norm(d[s:e])) for s, e in slices])

    for i, j in pairs:
        a, b = array[i], array[j]
        length = inactive_edge_length(b - a, slices)
        k = int(np.ceil(length / epoch_step - 1e-9))
        if k <= 1:
            keep[i].append(j)
            continue
        interior = [a + (b - a) * (t / k) for t in range(1, k)]
        if not all(free(q) for q in interior):

            if stats is not None:
                stats["subdiv_edges_dropped"] = stats.get("subdiv_edges_dropped", 0) + 1
            continue
        if stats is not None:
            stats["subdiv_edges_split"] = stats.get("subdiv_edges_split", 0) + 1
            stats["subdiv_vertices"] = stats.get("subdiv_vertices", 0) + (k - 1)
        base = n0 + len(new_pts)
        new_pts.extend(interior)
        chain = [i] + list(range(base, base + k - 1)) + [j]
        new_links.extend(zip(chain[:-1], chain[1:]))

    vertices = list(vertices) + new_pts
    array = np.vstack([array, np.stack(new_pts)]) if new_pts else array
    adjacency = [[] for _ in range(len(vertices))]
    for i, nbrs in enumerate(keep):
        for j in nbrs:
            new_links.append((i, j))
    for i, j in new_links:
        st = step_of(array[i], array[j])
        adjacency[i].append((j, st))
        adjacency[j].append((i, st))
    return vertices, adjacency, array

def _lattice_is_affordable(env, indices, config, robots=None) -> bool:
    """Whether a regular lattice over this inactive subspace fits the configured budget"""
    d = len(indices)
    if d == 0 or d > config.inactive_lattice_max_dim:
        return False
    p_max = _lattice_p_max(env, robots) if robots else d
    h = inactive_step_radius(config, env) / np.sqrt(p_max)
    span = env.limits[1, indices] - env.limits[0, indices]
    n = float(np.prod(np.floor(span / h) + 1.0))
    return n <= config.inactive_lattice_max_vertices

def build_inactive_lattice(env, mode, active_tasks, seed_configs, config) -> InactiveRoadmap:
    """
    Regular-lattice roadmap over the inactive subspace, for the low-dimensional case. Same contract
    as build_inactive_roadmap -- one edge is one action is one decision epoch, seeds are exact, the
    result is pruned to what a mode entry can reach -- but the vertices lie on lines instead of
    being scattered, which is the only way to get a straight executed path
    """
    robots = inactive_robots(env, active_tasks)
    indices = subspace_indices(env, robots)
    slices = inactive_robot_slices(env, robots)
    step_radius = inactive_step_radius(config, env)
    d = len(indices)
    h = step_radius / np.sqrt(_lattice_p_max(env, robots))
    free, edge_free = inactive_feasibility(env, mode, robots, indices)
    lower, upper = env.limits[0, indices], env.limits[1, indices]
    axes = [np.arange(lower[k], upper[k] + 1e-9, h) for k in range(d)]
    shape = tuple(len(a) for a in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    occupied = np.array([free(q) for q in grid], dtype=bool)
    n_grid = len(grid)
    vertices = [grid[i] for i in range(n_grid)]
    adjacency: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(n_grid)]
    seeds: Dict[str, List[int]] = {"entry": [], "task_goal": []}

    def link(i: int, j: int) -> None:
        diff = vertices[j] - vertices[i]
        step = np.asarray([float(np.linalg.norm(diff[a:b])) for a, b in slices])
        adjacency[i].append((j, step))
        adjacency[j].append((i, step))

    offsets = [o for o in itertools.product((-1, 0, 1), repeat=d) if any(o)]
    strides = np.array([int(np.prod(shape[k + 1:])) for k in range(d)], dtype=np.int64)
    for flat in range(n_grid):
        if not occupied[flat]:
            continue
        idx = np.array(np.unravel_index(flat, shape), dtype=np.int64)
        for off in offsets:
            nb = idx + np.asarray(off, dtype=np.int64)
            if np.any(nb < 0) or np.any(nb >= np.asarray(shape)):
                continue
            j = int(np.dot(nb, strides))
            if j <= flat or not occupied[j]:
                continue
            if edge_free(vertices[flat], vertices[j]):
                link(flat, j)

    array = np.stack(vertices)

    def add_seed(q_sub, kind: str) -> None:
        q_sub = np.asarray(q_sub, dtype=np.float64)
        if not free(q_sub):
            return

        near = int(np.argmin(np.linalg.norm(array[:n_grid] - q_sub, axis=1)))
        if occupied[near] and float(np.linalg.norm(array[near] - q_sub)) < 1e-9:
            seeds[kind].append(near)
            return
        vertices.append(q_sub)
        adjacency.append([])
        i = len(vertices) - 1
        seeds[kind].append(i)

        distances = inactive_edge_lengths(array[:n_grid] - q_sub, slices)
        cand = np.nonzero((distances <= step_radius) & occupied)[0]
        for j in map(int, cand[np.argsort(distances[cand])]):
            if len(adjacency[i]) >= config.inactive_k_max:
                break
            if edge_free(q_sub, array[j]):
                link(i, j)

    for q_sub in seed_configs:
        add_seed(q_sub, "entry")
    for q_sub in sample_inactive_task_goals(env, mode, robots, indices):
        add_seed(q_sub, "task_goal")

    array = np.stack(vertices)
    n_sampled = int(occupied.sum()) + (len(vertices) - n_grid)
    kept = reachable_from(adjacency, seeds["entry"]) if seeds["entry"] else [
        i for i in range(len(vertices)) if adjacency[i] or i >= n_grid]
    remap = {old: new for new, old in enumerate(kept)}
    return InactiveRoadmap(
        mode=mode,
        robots=robots,
        indices=indices,
        vertices=array[kept],
        adjacency=[[(remap[j], w) for j, w in adjacency[i] if j in remap] for i in kept],
        seeds={k: [remap[i] for i in v if i in remap] for k, v in seeds.items()},
        n_sampled=n_sampled,
        n_stranded_seeds=sum(1 for i in seeds["entry"] + seeds["task_goal"] if i not in remap),
    )

def build_inactive_roadmap(env, mode, active_tasks, seed_configs, config,
                           rrg_subtree=None, seams=None) -> InactiveRoadmap:
    """
    Builds the inactive roadmap for a skill mode. - If 2D / lattice is affordable: uses
    build_inactive_lattice
    """
    robots = inactive_robots(env, active_tasks)
    indices = subspace_indices(env, robots)
    if len(indices) == 0:
        return InactiveRoadmap(
            mode, robots, indices, np.zeros((1, 0)), [[]], seeds={"entry": [], "task_goal": []}
        )

    structure = config.inactive_roadmap_structure
    if structure == "lattice" or (
            structure == "auto" and _lattice_is_affordable(env, indices, config, robots)):
        return build_inactive_lattice(env, mode, active_tasks, seed_configs, config)

    return build_inactive_roadmap_from_rrg(
        env, mode, active_tasks, seed_configs, config,
        rrg_subtree=rrg_subtree, seams=seams
    )

def build_inactive_roadmap_from_rrg(
    env,
    mode: Mode,
    active_tasks: List[Task],
    seed_configs: Sequence[np.ndarray],
    config: "ReactiveRoadmapConfig",
    rrg_subtree: Optional[Any] = None,
    seams: Optional[Dict] = None,
) -> InactiveRoadmap:
    """
    Builds an inactive roadmap by extracting vertices, continuous rollout waypoints, and edges
    directly from the multi-robot RRG subtree constructed in Phase A, augmented with entry seeds,
    exit seams, and collision-free cross connections
    """
    robots = inactive_robots(env, active_tasks)
    indices = subspace_indices(env, robots)
    if len(indices) == 0:
        return InactiveRoadmap(
            mode, robots, indices, np.zeros((1, 0)), [[]], seeds={"entry": [], "task_goal": []}
        )

    slices = inactive_robot_slices(env, robots)
    epoch_step = inactive_step_radius(config, env)
    step_radius = inactive_reach_radius(config, env)
    connect_radius = max(step_radius, config.inactive_connect_epochs * epoch_step)
    free, edge_free = inactive_feasibility(env, mode, robots, indices)
    vertices: List[np.ndarray] = []
    adjacency: List[List[Tuple[int, np.ndarray]]] = []
    seeds: Dict[str, List[int]] = {"entry": [], "task_goal": []}
    build_stats: Dict[str, int] = {}

    def link(i: int, j: int) -> None:
        if i == j or i < 0 or j < 0:
            return
        if any(nb == j for nb, _ in adjacency[i]):
            return
        diff = vertices[j] - vertices[i]
        step = np.asarray([float(np.linalg.norm(diff[s:e])) for s, e in slices])
        adjacency[i].append((j, step))
        adjacency[j].append((i, step))

    def add_point(q_sub, kind: Optional[str] = None) -> int:
        q_sub = np.asarray(q_sub, dtype=np.float64)
        if not free(q_sub):
            return -1

        check_start = max(0, len(vertices) - 100)
        for existing_idx in range(check_start, len(vertices)):
            if np.linalg.norm(vertices[existing_idx] - q_sub) < 1e-4:
                if kind:
                    seeds[kind].append(existing_idx)
                return existing_idx
        idx = len(vertices)
        vertices.append(q_sub)
        adjacency.append([])
        if kind:
            seeds[kind].append(idx)
        return idx

    for q_sub in seed_configs:
        add_point(q_sub, "entry")
    for q_sub in sample_inactive_task_goals(env, mode, robots, indices):
        add_point(q_sub, "task_goal")

    node_to_vtx: Dict[int, int] = {}
    if rrg_subtree is not None and hasattr(rrg_subtree, "nodes") and rrg_subtree.size > 0:
        nodes = rrg_subtree.nodes[: rrg_subtree.size]
        for node in nodes:

            if (node.skill_edge is not None and hasattr(node.skill_edge, "waypoints")
                    and len(node.skill_edge.waypoints) > 0):
                wps = node.skill_edge.waypoints
                prev_idx = node_to_vtx.get(id(node.parent)) if node.parent is not None else None
                for k in range(len(wps)):
                    wp_sub = np.asarray(wps[k, indices], dtype=np.float64)
                    curr_idx = add_point(wp_sub)
                    if curr_idx >= 0 and prev_idx is not None and curr_idx != prev_idx:
                        link(prev_idx, curr_idx)
                    if curr_idx >= 0:
                        prev_idx = curr_idx
                if prev_idx is not None:
                    node_to_vtx[id(node)] = prev_idx
            else:
                q_sub = np.asarray(node.state.q.state()[indices], dtype=np.float64)
                curr_idx = add_point(q_sub)
                if curr_idx >= 0:
                    node_to_vtx[id(node)] = curr_idx
                    if node.parent is not None and id(node.parent) in node_to_vtx:
                        parent_idx = node_to_vtx[id(node.parent)]
                        link(parent_idx, curr_idx)

        for node in nodes:
            u = node_to_vtx.get(id(node))
            if u is None:
                continue
            for nb, _, kind in node.edges:
                v = node_to_vtx.get(id(nb))
                if v is not None and u != v:
                    link(u, v)

    if seams is not None:
        for (src, dst), qs in seams.items():
            if src == mode:
                for q_seam in qs:
                    q_sub = np.asarray(q_seam, dtype=np.float64)[indices]
                    add_point(q_sub)

    if len(vertices) > 1:
        array = np.stack(vertices)
        k_max = config.inactive_k_max
        for i in range(len(vertices)):

            dists = inactive_edge_lengths(array - array[i], slices)
            near = np.nonzero((dists <= connect_radius) & (dists > 1e-9))[0]
            for j in map(int, near[np.argsort(dists[near])]):
                if len(adjacency[i]) >= k_max:
                    break
                if any(nb == j for nb, _ in adjacency[i]):
                    continue
                if edge_free(array[i], array[j]):
                    link(i, j)

        all_seeds = seeds["entry"] + seeds["task_goal"]
        if all_seeds:
            labels, _ = component_labels(adjacency)
            sizes = np.bincount(np.asarray(labels, dtype=int))
            root = max(all_seeds, key=lambda i: sizes[labels[i]])
            main_label = labels[root]
            for s in all_seeds:
                if labels[s] != main_label:
                    main_pool = [idx for idx in range(len(vertices)) if labels[idx] == main_label]
                    if main_pool:
                        sub_dists = inactive_edge_lengths(array[main_pool] - array[s], slices)
                        order = np.argsort(sub_dists)
                        for cand_pos in order[:min(10, len(order))]:
                            cand_idx = main_pool[cand_pos]
                            if edge_free(array[s], array[cand_idx]):
                                link(s, cand_idx)
                                labels, _ = component_labels(adjacency)
                                main_label = labels[root]
                                break

        vertices, adjacency, array = _subdivide_edges(
            vertices, adjacency, array, slices, epoch_step, free, stats=build_stats
        )
    else:
        array = np.stack(vertices) if vertices else np.zeros((0, len(indices)))

    n_sampled = len(vertices)
    kept = reachable_from(adjacency, seeds["entry"]) if seeds["entry"] else list(range(n_sampled))
    remap = {old: new for new, old in enumerate(kept)}
    return InactiveRoadmap(
        mode=mode,
        robots=robots,
        indices=indices,
        vertices=array[kept] if len(kept) > 0 else np.zeros((0, len(indices))),
        adjacency=[[(remap[j], w) for j, w in adjacency[i] if j in remap] for i in kept],
        seeds={k: [remap[i] for i in v if i in remap] for k, v in seeds.items()},
        n_sampled=n_sampled,
        n_stranded_seeds=sum(1 for i in seeds["entry"] + seeds["task_goal"] if i not in remap),
        build_stats=build_stats,
    )

_LEGAL_SUCCESSOR_CACHE: Dict[int, set] = {}

def is_legal_mode_successor(env, parent: Mode, child: Mode) -> bool:
    """True if `child` is a mode the environment allows to follow `parent`"""
    key = id(parent)
    allowed = _LEGAL_SUCCESSOR_CACHE.get(key)
    if allowed is None:
        try:
            allowed = {tuple(ids) for ids in env.get_valid_next_task_combinations(parent)}
        except Exception:
            allowed = set()
        _LEGAL_SUCCESSOR_CACHE[key] = allowed
    if not allowed:
        return True
    return tuple(child.task_ids) in allowed



def add_inactive_vertices(env, graph: InactiveRoadmap, points, config, k_connect: int = 10) -> int:
    """
    Adds collision-free configurations to an existing inactive roadmap and links them into it,
    returning how many were kept
    """
    if graph.n_vertices == 0 or len(graph.indices) == 0:
        return 0

    slices = inactive_robot_slices(env, graph.robots)
    step_radius = inactive_step_radius(config, env)
    free, edge_free = inactive_feasibility(env, graph.mode, graph.robots, graph.indices)
    array = graph.vertices
    adjacency = graph.adjacency
    added = 0
    cap = getattr(graph, "max_vertices", None) or getattr(config, "inactive_max_vertices", None)
    for point in points:
        if cap is not None and len(graph.vertices) >= cap:
            break
        q_sub = np.asarray(point, dtype=np.float64)
        distances = inactive_edge_lengths(array - q_sub, slices)
        if float(distances.min()) < config.inactive_dedup_frac * step_radius or not free(q_sub):
            continue

        i = len(array)
        adjacency.append([])

        near = np.nonzero(distances <= step_radius)[0]
        for j in map(int, near[np.argsort(distances[near])]):
            if len(adjacency[i]) >= k_connect:
                break
            if not edge_free(q_sub, array[j]):
                continue
            diff = array[j] - q_sub
            step = np.asarray([float(np.linalg.norm(diff[s:e])) for s, e in slices])
            adjacency[i].append((j, step))
            adjacency[j].append((i, step))

        if not adjacency[i]:
            adjacency.pop()
            continue
        array = np.vstack([array, q_sub])
        added += 1

    graph.vertices = array
    graph.n_sampled += added
    return added

def add_inactive_entry_bridges(env, graph: InactiveRoadmap, points, config) -> int:
    """Adds exact seam entries and subdivided one-epoch bridges to an inactive roadmap"""
    if graph.n_vertices == 0 or len(graph.indices) == 0:
        return 0

    slices = inactive_robot_slices(env, graph.robots)
    step = inactive_step_radius(config, env)
    tolerance = max(1e-8, 1e-6 * step)
    free, edge_free = inactive_feasibility(env, graph.mode, graph.robots, graph.indices)
    added = 0

    def distance(a, b):
        return inactive_edge_length(np.asarray(b) - np.asarray(a), slices)

    def append_vertex(point):
        nonlocal added
        graph.vertices = np.vstack([graph.vertices, np.asarray(point, dtype=np.float64)])
        graph.adjacency.append([])
        graph.n_sampled += 1
        added += 1
        return graph.n_vertices - 1

    for point in points:
        point = np.asarray(point, dtype=np.float64)
        distances = np.asarray([distance(vertex, point) for vertex in graph.vertices])
        exact = int(np.argmin(distances)) if len(distances) else -1
        if exact >= 0 and float(distances[exact]) <= tolerance:
            if exact not in graph.seeds.setdefault("entry", []):
                graph.seeds["entry"].append(exact)
            continue
        if not free(point):
            continue

        target = int(np.argmin(distances))
        target_point = graph.vertices[target].copy()
        n_steps = max(1, int(np.ceil(float(distances[target]) / step - 1e-9)))
        chain = [point + (target_point - point) * (k / n_steps)
                 for k in range(n_steps)]
        if not all(free(candidate) and edge_free(chain[k], chain[k + 1])
                   for k, candidate in enumerate(chain[:-1])):
            continue
        if not edge_free(chain[-1], target_point):
            continue

        previous = append_vertex(chain[0])
        graph.seeds.setdefault("entry", []).append(previous)
        for candidate in chain[1:]:
            current = append_vertex(candidate)
            diff = graph.vertices[current] - graph.vertices[previous]
            weights = np.asarray([float(np.linalg.norm(diff[a:b])) for a, b in slices])
            graph.adjacency[previous].append((current, weights))
            graph.adjacency[current].append((previous, weights))
            previous = current

        diff = graph.vertices[target] - graph.vertices[previous]
        weights = np.asarray([float(np.linalg.norm(diff[a:b])) for a, b in slices])
        graph.adjacency[previous].append((target, weights))
        graph.adjacency[target].append((previous, weights))
    return added

class ReactiveRoadmap:
    """Multi-modal roadmap representation for the reactive planner"""
    def __init__(self, env: BaseProblem, config: ReactiveRoadmapConfig):
        """Documentation"""
        self.env = env
        self.config = config
        self.planner: Optional[RRTSkills] = None
        self.modes: List[Mode] = []
        self.skill_modes: List[Mode] = []
        self.seams: Dict[Tuple[Mode, Mode], List[np.ndarray]] = {}
        self.inactive_roadmaps: Dict[Mode, InactiveRoadmap] = {}
        self.nature_sets: Dict[str, NatureSet] = {}
        self.joint_nature_sets: Dict[Mode, NatureSet] = {}
        self.build_times: Dict[str, float] = {}
        self.phase_a_times: List[float] = []
        self.phase_a_costs: List[float] = []

    def _build_composite_rrg(self, stop_at_first_solution: bool = False):
        """Runs standard composite RRG to discover reachable modes and deterministic roadmaps"""
        rrt_config = RRTSkillsConfig()
        rrt_config.build_mode = "rrg"
        rrt_config.try_shortcutting = self.config.build_shortcut
        rrt_config.inactive_max_vel = self.config.inactive_max_vel
        rrt_config.kinodynamic_steps = self.config.rrg_kinodynamic_steps
        rrt_config.skill_duration_model = "sampled_per_branch"
        rrt_config.inactive_transition_source = self.config.rrg_inactive_transition_source
        self.planner = RRTSkills(self.env, rrt_config)
        if self.config.rrg_iters is not None:
            termination = IterationTerminationCondition(self.config.rrg_iters)
        else:
            termination = RuntimeTerminationCondition(self.config.rrg_runtime)
        _, phase_a_info = self.planner.plan(termination, optimize=not stop_at_first_solution)

        self.phase_a_times = [float(t) for t in phase_a_info.get("times", [])]
        self.phase_a_costs = [float(c) for c in phase_a_info.get("costs", [])]
        self.modes = list(self.planner.reached_modes)
        self.skill_modes = [m for m in self.modes if active_skill_tasks(self.env, m)]

    def _extract_mode_transitions(self):
        """Extracts directed mode-transition boundary configurations from the RRG"""
        self.seams = {}
        _LEGAL_SUCCESSOR_CACHE.clear()
        for mode in self.modes:
            subtree = self.planner.tree.subtrees[mode]
            for node in subtree.nodes[: subtree.size]:
                for child in node.children:
                    if child.state.mode != node.state.mode:

                        if not is_legal_mode_successor(
                            self.env, node.state.mode, child.state.mode
                        ):
                            continue
                        key = (node.state.mode, child.state.mode)
                        q = np.asarray(child.state.q.state(), dtype=np.float64)
                        self.seams.setdefault(key, []).append(q)
    def get_mode_entry_configs(self, mode: Mode) -> List[np.ndarray]:
        """Retrieves composite configurations at which a given mode was entered"""
        entries = [q for (_, dst), qs in self.seams.items() if dst == mode for q in qs]
        if not entries and mode == self.env.get_start_mode():
            entries = [np.asarray(self.env.get_start_pos().state(), dtype=np.float64)]
        return entries

    def mode_active_tasks(self, mode: Mode) -> List[Task]:
        """The mode's active skill tasks, ordered by their robots' position in env.robots"""
        order = {r: i for i, r in enumerate(self.env.robots)}
        return sorted(active_skill_tasks(self.env, mode),
                      key=lambda t: min(order[r] for r in t.robots))

    def _joint_entry_states(self, mode: Mode, tasks: Sequence[Task]) -> List[Tuple[int, ...]]:
        """The joint nature states this mode can be ENTERED at"""
        predecessors = [src for (src, dst) in self.seams if dst == mode]
        fresh_only = [(tuple(self.nature_sets[t.name].initial for t in tasks))]
        if not predecessors:
            return fresh_only

        cap = max(1, int(self.config.joint_entry_states_per_component))
        entries: List[Tuple[int, ...]] = []
        for src in predecessors:
            running = {t.name for t in active_skill_tasks(self.env, src)}
            choices = []
            for task in tasks:
                nature = self.nature_sets[task.name]
                if task.name not in running:
                    choices.append([nature.initial])
                    continue
                free = np.argsort(nature.epochs, kind="stable")
                if len(free) > cap:
                    free = free[np.linspace(0, len(free) - 1, cap).round().astype(int)]
                choices.append([int(x) for x in free])
            combos: List[Tuple[int, ...]] = [()]
            for choice in choices:
                combos = [c + (x,) for c in combos for x in choice]
            entries.extend(combos)

        entries.extend(e for e in fresh_only if e not in entries)
        return list(dict.fromkeys(entries)) or fresh_only

    def _build_joint_nature_sets(self) -> None:
        """One product NatureSet per mode that runs more than one skill at once"""
        for mode in self.skill_modes:
            tasks = self.mode_active_tasks(mode)
            if len(tasks) < 2:
                continue
            natures = [self.nature_sets[t.name] for t in tasks]
            entries = self._joint_entry_states(mode, tasks)
            joint = product_nature_set(natures, [t.name for t in tasks], entries)
            self.joint_nature_sets[mode] = joint
    def _find_dead_end_modes(self) -> List[Mode]:
        """
        Reached modes from which the mode graph has NO route to a terminal mode. This is the single
        cause of V(start) = inf found so far, and it is a phase-A SAMPLING accident, not a modelling
        error
        """
        terminal = [m for m in self.modes if self.env.is_terminal_mode(m)]
        if not terminal:
            return []
        incoming: Dict[Mode, List[Mode]] = {}
        for a, b in self.seams:
            incoming.setdefault(b, []).append(a)
        alive, frontier = set(terminal), list(terminal)
        while frontier:
            for predecessor in incoming.get(frontier.pop(), []):
                if predecessor not in alive:
                    alive.add(predecessor)
                    frontier.append(predecessor)
        dead = [m for m in self.modes if m not in alive]
        return dead

    def build(self) -> "ReactiveRoadmap":
        """Executes the full 3-phase roadmap construction pipeline"""
        t0 = time.time()
        self._build_composite_rrg()
        self._extract_mode_transitions()
        self.dead_end_modes = self._find_dead_end_modes()
        self.build_times["phase_a"] = time.time() - t0
        t0 = time.time()
        for mode in self.skill_modes:
            for task in active_skill_tasks(self.env, mode):
                if task.name not in self.nature_sets:
                    self.nature_sets[task.name] = build_nature_set(self.env, task, self.config)
        self._build_joint_nature_sets()
        self.build_times["nature"] = time.time() - t0
        t0 = time.time()
        build_modes = list(self.skill_modes)
        n_samples = self.config.inactive_n_samples
        if self.config.inactive_sample_budget and build_modes:
            n_samples = int(np.clip(self.config.inactive_sample_budget // len(build_modes),
                                    self.config.inactive_n_samples_min,
                                    self.config.inactive_n_samples))
        mode_config = replace(self.config, inactive_n_samples=n_samples)
        for mode in build_modes:
            tasks = active_skill_tasks(self.env, mode)
            indices = subspace_indices(self.env, inactive_robots(self.env, tasks))
            self.inactive_roadmaps[mode] = build_inactive_roadmap(
                self.env, mode, tasks, [q[indices] for q in self.get_mode_entry_configs(mode)], mode_config,
                rrg_subtree=self.planner.tree.subtrees.get(mode),
                seams=self.seams
            )

        available = _available_memory_bytes()
        for mode, graph in self.inactive_roadmaps.items():
            explicit = self.config.inactive_max_vertices
            if explicit:
                graph.max_vertices = int(explicit)
                continue
            nature = self._mode_nature_set(mode)
            if nature is None or available is None:
                continue
            per_vertex = max(1, nature.n_states * 24)
            graph.max_vertices = max(
                graph.n_vertices,
                int(self.config.inactive_memory_fraction * available / per_vertex))
        self.build_times["inactive"] = time.time() - t0
        return self

    def mode_coverage_complete(self) -> bool:
        """Reports whether the roadmap has an initialized composite planner"""
        return self.planner is not None

    def grow(self, seconds: float,
             stop_at_first_solution: bool = False,
             inactive_fraction: float = 0.25) -> Dict[str, int]:
        """Extend the roadmap by ONE sampling batch, then fold whatever appeared into the model"""
        first = self.planner is None
        before_modes = 0 if first else len(self.modes)
        before_seams = sum(len(v) for v in self.seams.values())
        before_skill = 0 if first else len(self.skill_modes)
        existing_skill = set(self.inactive_roadmaps)
        inactive_fraction = float(np.clip(inactive_fraction, 0.0, 0.8))
        densifiable = bool(existing_skill) and any(
            not _lattice_is_affordable(
                self.env, self.inactive_roadmaps[mode].indices,
                self.config, self.inactive_roadmaps[mode].robots)
            for mode in existing_skill
        )
        effective_inactive_fraction = inactive_fraction if densifiable else 0.0

        if first:

            original = self.config
            self.config = replace(original, rrg_runtime=seconds)
            try:
                self._build_composite_rrg(stop_at_first_solution=stop_at_first_solution)
            finally:
                self.config = original
        else:
            planning_seconds = seconds * (1.0 - effective_inactive_fraction)
            self.planner.plan(RuntimeTerminationCondition(planning_seconds), optimize=True)
            self.modes = list(self.planner.reached_modes)
            self.skill_modes = [m for m in self.modes if active_skill_tasks(self.env, m)]

        self._extract_mode_transitions()
        self.dead_end_modes = self._find_dead_end_modes()

        for mode in self.skill_modes:
            for task in active_skill_tasks(self.env, mode):
                if task.name not in self.nature_sets:
                    self.nature_sets[task.name] = build_nature_set(self.env, task, self.config)

        for mode in self.skill_modes:
            if len(self.mode_active_tasks(mode)) < 2 or mode in self.joint_nature_sets:
                continue
            tasks = self.mode_active_tasks(mode)
            natures = [self.nature_sets[t.name] for t in tasks]
            entries = self._joint_entry_states(mode, tasks)
            self.joint_nature_sets[mode] = product_nature_set(
                natures, [t.name for t in tasks], entries)

        fresh = [m for m in self.skill_modes if m not in self.inactive_roadmaps]
        if fresh:
            mode_config = replace(self.config,
                                  inactive_n_samples=self.config.inactive_n_samples)
            for mode in fresh:
                tasks = active_skill_tasks(self.env, mode)
                indices = subspace_indices(self.env, inactive_robots(self.env, tasks))
                self.inactive_roadmaps[mode] = build_inactive_roadmap(
                    self.env, mode, tasks,
                    [q[indices] for q in self.get_mode_entry_configs(mode)], mode_config,
                    rrg_subtree=self.planner.tree.subtrees.get(mode),
                    seams=self.seams)
            self._cap_inactive_vertices(fresh)

        added_inactive = 0
        if not first and effective_inactive_fraction > 0.0:
            added_inactive = self.densify_inactive_roadmaps(
                seconds * effective_inactive_fraction,
                modes=[m for m in self.skill_modes if m in existing_skill],
            )

        for (src, dst), seam_configs in self.seams.items():
            graph = self.inactive_roadmaps.get(dst)
            if graph is None or dst not in self.skill_modes:
                continue
            graph_indices = graph.indices
            add_inactive_entry_bridges(
                self.env, graph,
                [q[graph_indices] for q in seam_configs],
                self.config,
            )

        return {"modes": len(self.modes) - before_modes,
                "seams": sum(len(v) for v in self.seams.values()) - before_seams,
                "skill_modes": len(self.skill_modes) - before_skill,
                "inactive": added_inactive}

    def densify_inactive_roadmaps(self, seconds: float,
                                  modes: Optional[Sequence[Mode]] = None) -> int:
        """Add collision-free one-epoch vertices to existing inactive roadmaps"""
        requested = self.skill_modes if modes is None else modes
        candidates = [m for m in requested if m in self.inactive_roadmaps]
        if not candidates or seconds <= 0.0:
            return 0

        rng = getattr(self, "_inactive_growth_rng", None)
        if rng is None:
            rng = np.random.default_rng(self.config.seed + 104729)
            self._inactive_growth_rng = rng

        deadline = time.time() + float(seconds)
        added = 0
        cursor = int(getattr(self, "_inactive_growth_cursor", 0))
        while time.time() < deadline:
            eligible = [m for m in candidates
                        if not _lattice_is_affordable(
                            self.env, self.inactive_roadmaps[m].indices,
                            self.config, self.inactive_roadmaps[m].robots)
                        and (self.inactive_roadmaps[m].n_vertices
                             < (self.inactive_roadmaps[m].max_vertices
                                or self.config.inactive_max_vertices
                                or float("inf")))]
            if not eligible:
                break
            mode = eligible[cursor % len(eligible)]
            cursor += 1
            graph = self.inactive_roadmaps[mode]
            if graph.n_vertices == 0 or len(graph.indices) == 0:
                continue

            points = self._sample_inactive_growth_points(graph, rng, 64)
            added += add_inactive_vertices(
                self.env, graph, points, self.config,
                k_connect=self.config.inactive_k_max,
            )
        self._inactive_growth_cursor = cursor % len(candidates)
        return added

    def _sample_inactive_growth_points(self, graph: InactiveRoadmap, rng,
                                       n_points: int) -> np.ndarray:
        """Sample local and global candidates around an existing inactive roadmap"""
        slices = inactive_robot_slices(self.env, graph.robots)
        step = inactive_step_radius(self.config, self.env)
        lower = self.env.limits[0, graph.indices]
        upper = self.env.limits[1, graph.indices]
        points = []
        for _ in range(n_points):
            if rng.random() < 0.1:
                q_sub = rng.uniform(lower, upper)
            else:
                base = graph.vertices[int(rng.integers(graph.n_vertices))]
                q_sub = np.array(base, dtype=np.float64, copy=True)
                for start, end in slices:
                    direction = rng.normal(size=end - start)
                    norm = float(np.linalg.norm(direction))
                    if norm <= 1e-12:
                        continue
                    q_sub[start:end] += direction / norm * step * np.sqrt(rng.random())
                q_sub = np.clip(q_sub, lower, upper)
            points.append(q_sub)
        return np.asarray(points, dtype=np.float64)

    def _mode_nature_set(self, mode: Mode) -> Optional["NatureSet"]:
        """
        The nature set whose size a mode's MDP tables scale with: joint if concurrent, else the single
        active skill's
        """
        joint = self.joint_nature_sets.get(mode)
        if joint is not None:
            return joint
        tasks = active_skill_tasks(self.env, mode)
        return self.nature_sets.get(tasks[0].name) if tasks else None

    def _cap_inactive_vertices(self, modes: Sequence[Mode]) -> None:
        """Per-mode ceiling on anything the shortcut loop injects later, for the given modes"""
        available = _available_memory_bytes()
        for mode in modes:
            graph = self.inactive_roadmaps.get(mode)
            if graph is None:
                continue
            explicit = self.config.inactive_max_vertices
            if explicit:
                graph.max_vertices = int(explicit)
                continue
            nature = self._mode_nature_set(mode)
            if nature is None or available is None:
                continue
            per_vertex = max(1, nature.n_states * 24)
            graph.max_vertices = max(
                graph.n_vertices,
                int(self.config.inactive_memory_fraction * available / per_vertex))
