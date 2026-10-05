from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import path_cost

from .reactive_mdp import ReactiveMDP
from .rrt_skills_reactive import (
    decision_epoch_indices,
    inactive_step_radius,
    inactive_robot_slices,
    nature_key,
    subspace_indices,
)

def build_nature_index(nature):
    """
    Lookup tables for projecting an online observation onto a nature state: exact key -> index, and
    the same keys as a float array for nearest-neighbour fallback
    """
    return ({key: i for i, key in enumerate(nature.keys)},
            np.asarray(nature.keys, dtype=np.float64))

def project_observation(nature, index, features, q_active, epoch):
    """
    Maps a continuous active-robot observation to the closest discrete nature state.
    """
    key = nature_key(q_active, epoch, nature.bin_tol, nature.time_indexed)
    exact = index.get(key)
    if exact is not None:
        return exact

    candidates = None
    if nature.time_indexed:
        layer = min(epoch, int(nature.epochs.max()))
        candidates = np.nonzero(nature.epochs == layer)[0]
        features = features[candidates][:, :-1]
        key = key[:-1]
    distances = np.linalg.norm(features - np.asarray(key, dtype=np.float64), axis=1)
    nearest = int(np.argmin(distances))
    if candidates is not None:
        nearest = int(candidates[nearest])
    return nearest

@dataclass
class Execution:
    """Results from one closed-loop execution"""
    path: List[State] = field(default_factory=list)
    reached_goal: bool = False
    cost: float = float("inf")
    skill_epochs: int = 0
    skill_seconds: float = 0.0
    skill_seconds_by_task: Dict[str, float] = field(default_factory=dict)
    failure: Optional[str] = None

class ReactiveExecutor:
    """Runs the policy once per call, each time against a fresh draw of the skill"""
    MAX_EPOCHS = 5000

    def __init__(self, mdp: ReactiveMDP, shortcut_iters: int = 1000):
        """
        shortcut_iters is the budget for shortcutting each non-skill leg at execution time; 0 disables
        it
        """
        self.shortcut_iters = shortcut_iters
        self.mdp = mdp
        self.env = mdp.env
        self.roadmap = mdp.roadmap
        self.config = mdp.roadmap.config
        tables = {name: build_nature_index(nature)
                  for name, nature in self.roadmap.nature_sets.items()}
        self._index = {name: table[0] for name, table in tables.items()}
        self._features = {name: table[1] for name, table in tables.items()}

    def _quantize_observation(self, name, nature, q_active, epoch):
        """Projects an online observation onto a discrete nature state; see project_observation"""
        return project_observation(nature, self._index[name], self._features[name],
                                   q_active, epoch)

    def _advance_mode(self, q: np.ndarray, mode, completed=None):
        """
        Queries environment transition logic and advances the mode state machine. Returns (mode, hops),
        where hops lists every DISTINCT mode entered along the way, in order (empty if the mode did not
        change)
        """
        config = self.env.get_start_pos().from_flat(q)
        pending = list(completed) if completed else []
        hops = []
        for _ in range(len(self.env.tasks)):
            if self.env.is_terminal_mode(mode):
                break
            try:
                if pending:
                    nxt = self.env.get_next_modes(config, mode, completed_task_ids=[pending.pop(0)])
                elif self.env.is_transition(config, mode):
                    nxt = self.env.get_next_modes(config, mode)
                else:
                    break
            except ValueError:
                break
            if not nxt:
                break
            mode = nxt[0]
            hops.append(mode)
        return mode, hops

    def run(self) -> Execution:
        """Executes a complete closed-loop trial against a fresh stochastic realization"""
        env = self.env
        result = Execution()
        mode = env.get_start_mode()
        q = np.asarray(env.get_start_pos().state(), dtype=np.float64)
        result.path.append(State(env.get_start_pos().from_flat(q), mode))

        skills = {}
        skill_vertices = {}

        for _ in range(self.MAX_EPOCHS):
            if env.done(env.get_start_pos().from_flat(q), mode):
                result.reached_goal = True
                break

            tasks = self.roadmap.mode_active_tasks(mode)
            if tasks:
                q, mode, skills = self._step_skill_epoch(
                    q, mode, tasks, skills, skill_vertices, result)
            else:
                skills = {}
                skill_vertices.clear()
                q, mode = self._execute_deterministic_mode(q, mode, result)
            if result.failure:
                break

        if not result.failure and not result.reached_goal:
            result.failure = "step limit reached"
        if len(result.path) > 1:
            result.cost = float(path_cost(result.path, env.batch_config_cost, env=env))
        return result

    def _joint_nature_state(self, mode, nature, components, q, epoch):
        """The joint nature state, from the per-skill states the observation projected onto"""
        if not nature.is_joint:
            return components[0]
        key = tuple(components)
        exact = nature.index.get(key)
        if exact is not None:
            return exact
        layer = np.nonzero(nature.epochs == min(epoch, int(nature.epochs.max())))[0]
        if len(layer) == 0:
            layer = np.arange(nature.n_states)
        distances = np.linalg.norm(nature.configs[layer] - q, axis=1)
        nearest = int(layer[int(np.argmin(distances))])
        return nearest

    def _resolve_skill_vertex(self, q, mode, graph, result):
        """Resolve a skill-mode entry configuration to an inactive-roadmap entry vertex"""
        q_inactive = q[graph.indices]
        candidates = graph.seeds.get("entry", [])
        candidates = np.asarray(list(candidates), dtype=int)
        if len(candidates) == 0:
            result.failure = f"skill mode {mode.task_ids} has no inactive-roadmap entry vertex"
            return None, 0.0
        distances = np.linalg.norm(graph.vertices[candidates] - q_inactive, axis=1)
        nearest = int(np.argmin(distances))
        vertex = int(candidates[nearest])
        gap = float(distances[nearest])
        return vertex, gap

    def _step_skill_epoch(self, q, mode, tasks, skills, skill_vertices, result):
        """
        Executes one decision epoch of a skill mode, with ANY number of skills running at once. Every
        running skill advances by exactly one decision epoch, which is what makes the joint nature model
        well defined: the product kernel moves all components together
        """
        env = self.env
        graph = self.roadmap.inactive_roadmaps.get(mode)
        if graph is None or mode not in self.mdp.pi_skill:
            result.failure = (f"no policy for skill mode {mode.task_ids}: phase A never explored "
                              f"it, so it has no inactive roadmap")
            return q, mode, skills
        if mode not in skill_vertices:
            vertex, gap = self._resolve_skill_vertex(q, mode, graph, result)
            if vertex is None:
                return q, mode, skills
            tolerance = max(1e-8, 1e-6 * inactive_step_radius(self.config, env))
            if gap > tolerance:
                result.failure = (f"skill mode {mode.task_ids} entered away from its modeled "
                                  f"roadmap entry by {gap:.6f}; refusing an execution-time "
                                  "alignment motion")
                return q, mode, skills
            skill_vertices[mode] = vertex
        current_vertex = int(skill_vertices[mode])
        _, nature, _ = self.mdp._mode_nature(mode)

        for task in tasks:
            if task.name in skills:
                continue
            active_idx = subspace_indices(env, task.robots)
            trajectory = np.asarray(
                task.skill.rollout(q[active_idx], task, env.get_joint_names(), env, t0=0.0).trajectory,
                dtype=np.float64)
            skills[task.name] = {
                "traj": trajectory,
                "idx": decision_epoch_indices(len(trajectory), task.skill.dt,
                                              self.config.decision_frequency),
                "epoch": 0,
                "active_idx": active_idx,
                "task": task,
            }
            realized_seconds = (len(trajectory) - 1) * task.skill.dt
            result.skill_seconds = max(result.skill_seconds, realized_seconds)
            result.skill_seconds_by_task[task.name] = realized_seconds

        live = [skills[t.name] for t in tasks]
        components = []
        for entry in live:
            task, epoch = entry["task"], entry["epoch"]
            component_nature = self.roadmap.nature_sets[task.name]
            nu_i = self._quantize_observation(
                task.name, component_nature, entry["traj"][entry["idx"][epoch]], epoch)
            components.append(nu_i)

        joint_epoch = max(entry["epoch"] for entry in live)
        q_active = np.concatenate([entry["traj"][entry["idx"][entry["epoch"]]] for entry in live])
        nu = self._joint_nature_state(mode, nature, components, q_active, joint_epoch)

        spans = []
        for entry in live:
            epoch, indices, trajectory = entry["epoch"], entry["idx"], entry["traj"]
            low = indices[epoch]
            high = indices[epoch + 1] if epoch + 1 < len(indices) else low
            spans.append((low, high, trajectory, entry["active_idx"]))

        n_sub = max(high - low for low, high, _, _ in spans)
        start = q[graph.indices].copy()
        reach = inactive_step_radius(self.config, env)
        slices = inactive_robot_slices(env, graph.robots)
        current_gap = float(np.linalg.norm(graph.vertices[current_vertex] - start))
        tolerance = max(1e-8, 1e-6 * reach)
        if current_gap > tolerance:
            result.failure = (f"skill mode {mode.task_ids} lost inactive roadmap state: current "
                              f"vertex {current_vertex} is {current_gap:.6f} from the realized "
                              f"configuration")
            return q, mode, skills

        def epoch_motion(target_vertex):
            """Micro-step states for steering towards target_vertex"""
            end = graph.vertices[target_vertex]

            states = []
            for step in range(1, n_sub + 1):
                alpha = step / n_sub
                q_step = q.copy()
                steps = {}
                for (low, high, trajectory, active_idx), entry in zip(spans, live):
                    idx = min(low + int(round(alpha * (high - low))), len(trajectory) - 1)
                    q_step[active_idx] = trajectory[idx]
                    steps[entry["task"].name] = idx
                q_step[graph.indices] = start + alpha * (end - start)
                states.append((q_step, steps))
            return states

        adjacency = {int(neighbour) for neighbour, _ in graph.adjacency[current_vertex]}
        nominal_target = int(self.mdp.pi_skill[mode][current_vertex, nu])
        targets = [nominal_target]
        if self.mdp.config.allow_wait and current_vertex not in targets:
            targets.append(current_vertex)

        env.set_to_mode(mode)
        target, tried, motion = -1, set(), []
        for cand_target in targets:
            if cand_target < 0 or cand_target in tried:
                continue
            if cand_target != current_vertex and cand_target not in adjacency:
                result.failure = (f"invalid reactive policy action in mode {mode.task_ids}: "
                                  f"vertex {current_vertex} targets non-neighbor {cand_target}")
                return q, mode, skills
            tried.add(cand_target)
            policy_edge_distance = max(
                (float(np.linalg.norm(graph.vertices[cand_target][a:b] -
                                      graph.vertices[current_vertex][a:b]))
                 for a, b in slices),
                default=0.0,
            )
            if policy_edge_distance > reach + tolerance:
                result.failure = (f"inactive roadmap edge exceeds one decision epoch in mode "
                                  f"{mode.task_ids}: {policy_edge_distance:.6f} > {reach:.6f}")
                return q, mode, skills
            motion = epoch_motion(cand_target)
            if all(env.is_collision_free_np(q_step, mode, set_mode=False) for q_step, _ in motion):
                target = cand_target
                break

        if target < 0:
            if tried:
                result.failure = (f"no collision-free action from the realized configuration at "
                                  f"mode {mode.task_ids}, epoch {joint_epoch}")
            else:
                result.failure = f"no feasible action at mode {mode.task_ids}, epoch {joint_epoch}"
            return q, mode, skills

        result.skill_epochs += 1

        for q_step, steps in motion:
            q = q_step
            result.path.append(State(env.get_start_pos().from_flat(q), mode,
                                     is_skill_waypoint=True, skill_steps=steps))

        q[graph.indices] = graph.vertices[target]
        skill_vertices[mode] = target

        completed = []
        for entry, (low, high, trajectory, _) in zip(live, spans):
            entry["epoch"] += 1
            if high >= len(trajectory) - 1:
                completed.append(env.tasks.index(entry["task"]))
                skills.pop(entry["task"].name, None)

        new_mode, hops = self._advance_mode(q, mode, completed=completed)
        boundary_steps = {entry["task"].name: high
                          for entry, (_, high, _, _) in zip(live, spans)}
        for hop_mode in hops:
            continuing_steps = {
                task.name: boundary_steps[task.name]
                for task in self.roadmap.mode_active_tasks(hop_mode)
                if task.name in boundary_steps
            }
            result.path.append(State(
                env.get_start_pos().from_flat(q), hop_mode,
                is_skill_waypoint=bool(continuing_steps), skill_steps=continuing_steps,
            ))
        if new_mode != mode:
            skill_vertices.pop(mode, None)
        return q, new_mode, skills

    def _append_interpolated_states(self, q: np.ndarray, mode, result: Execution,
                                   interpolate_from: Optional[np.ndarray] = None, resolution: float = 0.05):
        """Appends configuration(s) to result.path with linear edge interpolation and state deduplication"""
        env = self.env
        if interpolate_from is not None and len(result.path) > 0:
            start = interpolate_from
            dist = float(np.linalg.norm(q - start))
            n_steps = max(1, int(np.ceil(dist / resolution)))
            for s in range(1, n_steps + 1):
                alpha = s / n_steps
                q_interp = start + alpha * (q - start)
                if len(result.path) > 0 and np.allclose(result.path[-1].q.state(), q_interp, atol=1e-6):
                    self._replace_last(result, q_interp, mode)
                else:
                    result.path.append(State(env.get_start_pos().from_flat(q_interp), mode))
        else:
            if len(result.path) > 0 and np.allclose(result.path[-1].q.state(), q, atol=1e-6):
                self._replace_last(result, q, mode)
            else:
                result.path.append(State(env.get_start_pos().from_flat(q), mode))

    def _replace_last(self, result: Execution, q: np.ndarray, mode) -> None:
        """Collapses a within-1e-6 duplicate onto the NEWER configuration"""
        result.path[-1].q = self.env.get_start_pos().from_flat(q)
        result.path[-1].mode = mode

    def _execute_deterministic_mode(self, q, mode, result):
        """
        Follows the deterministic policy across one non-skill mode, shortcutting the chain before
        driving it
        """
        env = self.env
        _, node = self.mdp.evaluate_connection_cost(q, mode, wide=True)
        if node < 0:
            result.failure = f"cannot enter mode {mode.task_ids} from the realized configuration"
            return q, mode

        chain = self.mdp.validated_chain(mode, node)
        nodes = self.mdp.det_nodes[mode]
        mode_states = [State(env.get_start_pos().from_flat(q), mode)]
        last_i = node
        for i in chain:
            mode_states.append(nodes[i].state)
            last_i = i

        for s in mode_states[1:]:
            q_target = np.asarray(s.q.state(), dtype=np.float64)
            self._append_interpolated_states(q_target, mode, result, interpolate_from=q)
            q = q_target

        child = self.mdp.det_exit[mode].get(last_i)
        if child is not None:
            q_target = np.asarray(child.state.q.state(), dtype=np.float64)
            self._append_interpolated_states(q_target, mode, result, interpolate_from=q)
            mode = child.state.mode

            result.path.append(State(env.get_start_pos().from_flat(q_target), mode))
            q = q_target
        elif not env.done(nodes[last_i].state.q, mode):
            result.failure = f"policy stalled inside mode {mode.task_ids}"

        return q, mode
