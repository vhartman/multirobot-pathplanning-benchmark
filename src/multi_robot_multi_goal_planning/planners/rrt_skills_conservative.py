import copy
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseProblem,
    Mode,
    State, 
    Task
)
from multi_robot_multi_goal_planning.problems.core.configuration import Configuration
from multi_robot_multi_goal_planning.problems.skills import (
    BaseStochasticTimedSkill,
    StochasticBaseSkill
)
from multi_robot_multi_goal_planning.planners import shortcutting
from .rrt_skills import (
    RRTSkills,
    RRTSkillsConfig,
    Node
)


# =====================================================================
# Config and Data Structures
# =====================================================================
@dataclass
class RRTSkillsConservativeConfig(RRTSkillsConfig):
    """
    Hyperparameters for the conservative multi-modal RRT with stochastic skills
    Inherits all core RRT* and skill parameters from RRTSkillsConfig
    """
    skill_duration_model: str = "max"
    tube_rollouts: int = 200  # MC rollouts to estimate the uncertainty tube


# =====================================================================
# Conservative Stochastic RRT* Planner
# =====================================================================
class RRTSkillsConservative(RRTSkills):
    """
    Conservative / tube planner for stochastic skills
    Inherits from RRTSkills and overrides skill stepping, horizon, and validation hooks to check
    safety against the union of Monte-Carlo rollouts across all stochastic branches.
    Commits the tree to follow the nominal (mean) trajectory of the longest branch
    """

    def __init__(self, env: BaseProblem, config: RRTSkillsConservativeConfig):
        super().__init__(env, config)
        # Uncertainty tubes for stochastic skills, grouping rollouts by their branch_idx
        self._skill_tubes: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def _use_nominal_tube(self, skill) -> bool:
        """
        Determines if a skill should be executed using the robust Tube-RRT strategy
        """
        return isinstance(skill, (BaseStochasticTimedSkill, StochasticBaseSkill))

    def _committed_branch(self, task_name: str, tubes: Dict) -> int:
        """
        The tube branch the tree commits to (open-loop): the branch with the longest
        unpadded nominal trajectory length, or the first branch if equal.
        """
        return max(tubes.keys(), key=lambda b: tubes[b].get("unpadded_len", len(tubes[b]["nominal"])))

    def _is_skill_done(self, node: Node, task: Task) -> bool:
        """
        Evaluates if an active skill has finished its execution.
        - For stochastic skills using nominal tubes, it finishes when base_step reaches
          the nominal tube length - 1 (the global_max - 1 budgeted steps)
        - For deterministic skills, fallback to base class RRTSkills
        """
        skill = task.skill
        if self._use_nominal_tube(skill) and task.name in self._skill_tubes:
            branch_idx = self._committed_branch(task.name, self._skill_tubes[task.name])
            nominal_traj = self._skill_tubes[task.name][branch_idx]["nominal"]
            base_step = node.state.skill_steps.get(task.name, 0)
            return base_step >= len(nominal_traj) - 1
        return super()._is_skill_done(node, task)
        
    def _get_skill_tubes(self, skill_task, q_subspace: np.ndarray, skill_step: int):
        """
        Runs tube_rollouts Monte Carlo executions of the stochastic skill and groups them by their 
        branch_idx. Per branch we keep the raw (collision checked) rollouts and their mean as the 
        nominal trajectory the tree follows. All branches are padded to global_max across all rollouts.
        """
        skill = skill_task.skill

        # Select the configuration from which the tube is generated
        if skill_step == 0:
            q_init = np.asarray(q_subspace)
        else:
            q_init = np.asarray(skill_task.initiation_goal.sample(None))

        key = skill_task.name

        # Reuse the cached tube
        if key in self._skill_tubes:
            return self._skill_tubes[key]

        all_joints = self.env.get_joint_names()

        # Generate Monte Carlo rollouts
        results = [
            skill.rollout(q_init, skill_task, all_joints, self.env, t0=0.0)
            for _ in range(self.config.tube_rollouts)
        ]

        labels = np.array([r.branch_idx if r.branch_idx is not None else 0 for r in results])

        # Find global maximum length across ALL rollouts in all branches
        global_max = max(len(r.trajectory) for r in results)

        # Group by branch and pad to a common horizon
        tubes = {}
        for b in np.unique(labels):
            b_trajs = [r.trajectory for r, lab in zip(results, labels) if lab == b]
            
            # Pad every trajectory up to global_max
            b_rollouts = np.stack([
                traj if len(traj) == global_max
                else np.vstack([traj, np.repeat(traj[-1:], global_max - len(traj), axis=0)])
                for traj in b_trajs
            ])
            # Duplicate trajectories cannot change the union collision test
            unique_rollouts = np.unique(b_rollouts, axis=0)
            tubes[b] = {
                "nominal": b_rollouts[0].copy(),
                "raw_rollouts": unique_rollouts,
                "unpadded_len": float(np.mean([len(traj) for traj in b_trajs])),
            }

        # Cache the tube for later expansions
        self._skill_tubes[key] = tubes
        return tubes

    def _get_skill_horizon(self, task: Task, q_subspace: np.ndarray, base_step: int, skill_steps: Optional[Dict[str, int]] = None) -> Tuple[int, bool]:
        """
        Returns (n_total_steps, is_bounded) for an active skill
        - For stochastic skills using nominal tubes, returns (len(nominal_traj) - 1, True)
        - For deterministic skills, fallback to base class RRTSkills
        """
        if self._use_nominal_tube(task.skill):
            # Tubes are precomputed on the first call and cached
            tubes = self._get_skill_tubes(task, q_subspace, base_step)
            branch_idx = self._committed_branch(task.name, tubes)
            nominal = tubes[branch_idx]["nominal"]
            return len(nominal) - 1, True
        return super()._get_skill_horizon(task, q_subspace, base_step, skill_steps)

    def _step_active_skill(self, task: Task, q_subspace: np.ndarray, step_idx: int, n_total: int) -> Tuple[np.ndarray, bool]:
        """
        Advances one active skill by 1 step
        - For stochastic skills, follows the precomputed nominal trajectory of the committed branch
        - For deterministic skills, fallback to base class RRTSkills
        """
        if self._use_nominal_tube(task.skill):
            tubes = self._skill_tubes[task.name] # Already cached by _get_skill_horizon
            branch_idx = self._committed_branch(task.name, tubes)
            nominal = tubes[branch_idx]["nominal"]
            q_new = nominal[min(step_idx, len(nominal) - 1)].copy()
            return q_new, step_idx >= n_total
        return super()._step_active_skill(task, q_subspace, step_idx, n_total)

    def _is_safe_against_tube(
        self,
        tube_tasks: List,
        tubes_by_task: Dict[str, Dict[str, Any]],
        steps: Dict[str, int],
        q_flat: np.ndarray,
        mode: Mode
    ) -> bool:
        """
        Evaluates collisions using the Monte Carlo realizations of the stochastic skill.
        Instead of a Cartesian margin, we directly check the cached rollouts at the current timestep
        """
        for t in tube_tasks:
            active_indices = self._get_active_subspace_indices([t])
            # Save original joint positions of the active robot
            q_orig = q_flat[active_indices].copy()
            
            # Check inactive robot against each realization at step k across all branches 
            for b in tubes_by_task[t.name].values():
                raw_rollouts = b["raw_rollouts"] # shape (N_rollouts, N_steps, N_joints)
                step_idx = min(steps[t.name], raw_rollouts.shape[1] - 1)
                for rollout_idx in range(raw_rollouts.shape[0]):
                    q_flat[active_indices] = raw_rollouts[rollout_idx, step_idx]
                    
                    q_cfg = self.env.get_start_pos().from_flat(q_flat)
                    if not self.env.is_collision_free(q_cfg, mode):
                        # Restore original before returning
                        q_flat[active_indices] = q_orig
                        return False
                        
            # Restore original after checking all branches
            q_flat[active_indices] = q_orig
            
        return True

    def _validate_skill_step(self, state_next: State, q_prev_cfg: Configuration, skill_tasks: List[Task], steps: Dict[str, int]) -> bool:
        """
        Stochastic override: first runs the base static-obstacle check, then additionally
        checks whether the inactive robots are safe against every MC rollout realization
        across all stochastic branches at the current step
        """
        # 1. Standard static-obstacle collision check
        if not super()._validate_skill_step(state_next, q_prev_cfg, skill_tasks, steps):
            return False

        # 2. MC tube check: inactive robot must be clear of all stochastic branches
        tube_tasks = [t for t in skill_tasks if self._use_nominal_tube(t.skill)]
        if tube_tasks:
            if not self._is_safe_against_tube(tube_tasks, self._skill_tubes, steps, state_next.q.state().copy(), state_next.mode):
                return False
        return True

    def _tube_state_validator(self, state: State) -> bool:
        """
        Tube clearance check for shortcutting: inactive robots may be moved by the shortcutter 
        through skill modes, but must stay outside the stochastic skills uncertainty tubes
        """
        mode = state.mode
        tube_tasks = []

        # 1. Identify which active tasks have stochastic tubes
        for t_id in dict.fromkeys(mode.task_ids):
            task = self.env.tasks[t_id]
            if (getattr(task, "skill", None) is not None 
                    and self._use_nominal_tube(task.skill) 
                    and task.name in self._skill_tubes):
                tube_tasks.append(task)

        # If no stochastic skills are active, the state is valid (trivial)
        if not tube_tasks:
            return True

        q = state.q.state()
        steps = {}

        # 2. Extract the current timestep for each active skill
        for t in tube_tasks:
            steps[t.name] = getattr(state, "skill_steps", {}).get(t.name, 0)

        # 3. Ensure the newly shortcutted state doesn't violate the inflated margins
        # (using the union check over all branches)
        return self._is_safe_against_tube(tube_tasks, self._skill_tubes, steps, np.asarray(q).copy(), mode)

    def _shortcut(self, path: List[State], shortcutting_iters: int) -> List[State]:
        """
        Post-processes a path with robot_mode_shortcut with tube_state_validator to protect tubes
        """
        shortcut_path, _ = shortcutting.robot_mode_shortcut(
            self.env, path, shortcutting_iters,
            resolution=self.env.collision_resolution,
            tolerance=self.env.collision_tolerance,
            robot_choice=self.config.shortcutting_mode,
            interpolation_resolution=self.config.shortcutting_interpolation_resolution,
            state_validator=self._tube_state_validator,
            deadline=getattr(self, "deadline", None),
        )

        # Remove interpolated points used in shortcutting (collision check)
        return shortcutting.remove_interpolated_nodes(shortcut_path)

    def plan(self, ptc, optimize: bool = False):
        """
        Runs RRT* planning with conservative tube checks and returns the solution path and metadata
        """
        path, info = super().plan(ptc, optimize=optimize)
        info["skill_tubes"] = self._skill_tubes
        return path, info


# =====================================================================
# Open-Loop Stochastic Replay
# =====================================================================

def stochastic_task_indices(env, task) -> np.ndarray:
    """Return the joint indices controlled by a task during replay"""
    idx = []
    end = 0
    for robot in env.robots:
        dim = env.robot_dims[robot]
        if robot in task.robots:
            idx.extend(range(end, end + dim))
        end += dim
    return np.array(idx)

def _rollout_fresh(env, task, q_init: np.ndarray) -> np.ndarray:
    """
    Executes one fresh noisy realization of task.skill from q_init
    Replay only checks the realized endpoint against the plan (see replay_stochastic_path),
    so no branch targeting is needed here - any realization is accepted as-is
    """
    skill = task.skill
    result = skill.rollout(q_init, task, env.get_joint_names(), env, t0=0.0)
    return result.trajectory

@dataclass
class ReplayResult:
    path: List[State]
    valid: bool = True
    break_reason: Optional[str] = None
    skill_steps: Dict[str, int] = None

_ENDPOINT_TOL = 1e-2
def replay_stochastic_path(env, path: List[State]) -> ReplayResult:
    """Replay one fresh stochastic realization through a conservative open-loop plan"""
    stoc_tasks = [
        t for t in env.tasks
        if getattr(t, "skill", None) is not None
        and isinstance(t.skill, (BaseStochasticTimedSkill, StochasticBaseSkill))
    ]
    replayed = copy.deepcopy(path)
    realized: Dict[str, int] = {}

    if not stoc_tasks:
        return ReplayResult(path=replayed, valid=True, break_reason=None, skill_steps=realized)

    for task in stoc_tasks:
        idx = stochastic_task_indices(env, task)
        seg_indices = [
            i for i, s in enumerate(replayed)
            if getattr(s, "is_skill_waypoint", False)
            and task.name in getattr(s, "skill_steps", {})
        ]
        if not seg_indices:
            continue

        # Interpolation can leave unmarked states inside the skill segment, include the full range
        seg_range = list(range(seg_indices[0], seg_indices[-1] + 1))
        seg_states = [replayed[i] for i in seg_range]

        # Remove duplicate marked waypoints while preserving unmarked interpolated states
        deduped, last_val = [], None
        for s in seg_states:
            val = getattr(s, "skill_steps", {}).get(task.name) if getattr(s, "is_skill_waypoint", False) else None
            if val is not None and val == last_val:
                continue
            deduped.append(s)
            if val is not None:
                last_val = val
        seg_states = deduped

        init_state = replayed[seg_indices[0] - 1] if seg_indices[0] > 0 else seg_states[0]
        q_init = np.asarray(init_state.q.state(), dtype=np.float64)[idx]

        fresh_traj = _rollout_fresh(env, task, q_init)
        realized[task.name] = len(fresh_traj) - 1

        fresh_segment = []
        for j, s_plan in enumerate(seg_states):
            # Use positional indexing: fresh_traj[0] is the pre-skill configuration
            active_q = fresh_traj[min(j + 1, len(fresh_traj) - 1)]
            q = np.asarray(s_plan.q.state(), dtype=np.float64).copy()
            q[idx] = active_q
            fresh_segment.append(
                State(
                    env.get_start_pos().from_flat(q),
                    s_plan.mode,
                    is_skill_waypoint=True,
                    # Keep interpolated states associated with the active skill
                    skill_steps=dict(getattr(s_plan, "skill_steps", {})) or {task.name: j},
                )
            )

        prefix = replayed[:seg_range[0]]
        suffix = replayed[seg_range[-1] + 1:]

        # A rollout fits when its number of motion steps does not exceed the reserved segment
        time_exceeded = (len(fresh_traj) - 1) > len(seg_states)

        # Check whether the realized endpoint matches the committed plan endpoint
        committed_endpoint = np.asarray(seg_states[-1].q.state(), dtype=np.float64)[idx]
        endpoint_dist = np.linalg.norm(fresh_traj[-1] - committed_endpoint)
        endpoint_mismatch = endpoint_dist > _ENDPOINT_TOL

        if time_exceeded or endpoint_mismatch:
            reason = (
                f"time_exceeded:{task.name}" if time_exceeded
                else f"endpoint_mismatch:{task.name}:dist={endpoint_dist:.4f}"
            )
            print(f"[REPLAY] '{task.name}' open-loop plan failed: {reason}. Halting execution.")
            return ReplayResult(path=prefix + fresh_segment, valid=False, break_reason=reason,
                                skill_steps=realized)

        # Snap the following transition state to the realized endpoint when it is acceptable
        if suffix and not endpoint_mismatch:
            q_next = np.asarray(suffix[0].q.state(), dtype=np.float64).copy()
            q_next[idx] = fresh_traj[-1]
            suffix[0] = State(
                env.get_start_pos().from_flat(q_next), suffix[0].mode,
                getattr(suffix[0], "is_skill_waypoint", False),
                dict(getattr(suffix[0], "skill_steps", {})),
            )

        replayed = prefix + fresh_segment + suffix if suffix else prefix + fresh_segment

    # Distinguish terminal failures from collision/constraint failures
    if not env.is_valid_plan(replayed):
        if not env.done(replayed[-1].q, replayed[-1].mode):
            print("[REPLAY] Terminal mode/goal not reached in physical execution realization")
            reason = "terminal_not_reached"
        else:
            print("[REPLAY] Collision detected in physical execution realization")
            reason = "collision"
        return ReplayResult(path=replayed, valid=False, break_reason=reason,
                            skill_steps=realized)

    return ReplayResult(path=replayed, valid=True, break_reason=None, skill_steps=realized)
