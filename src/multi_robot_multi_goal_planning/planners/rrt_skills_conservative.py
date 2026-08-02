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
    BaseDeterministicTimedSkill,
    BaseStochasticTimedSkill,
    StochasticBaseSkill
)
from multi_robot_multi_goal_planning.planners import shortcutting
from .rrt_skills import (
    RRTSkills,
    RRTSkillsConfig,
    Node,
    SkillEdge
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
    tube_rollouts: int = 200  # MC rollouts to estimate the uncertainty tube


# =====================================================================
# Conservative Stochastic RRT* Planner
# =====================================================================
class RRTSkillsConservative(RRTSkills):
    """
    Conservative / tube planner for stochastic skills
    Inherits from RRTSkills and overrides skill expansion and shortcutting to check
    safety against the union of Monte-Carlo rollouts across all stochastic branches.
    Commits the tree to follow the nominal (mean) trajectory of the first branch.
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
        The tube branch the tree commits to (open-loop): the first available branch
        """
        return next(iter(tubes))
        
    def _get_skill_tubes(self, skill_task, q_subspace: np.ndarray, skill_step: int):
        """
        Runs tube_rollouts Monte Carlo executions of the stochastic skill and groups them by their 
        branch_idx. Per branch we keep the raw (collision checked) rollouts and their mean as the 
        nominal trajectory the tree follows. Rollouts are padded only within a branch
        """
        key = skill_task.name

        # 1. Check cache if tube already calculated for this task
        if key in self._skill_tubes:
            return self._skill_tubes[key]
        
        skill = skill_task.skill

        # 2. Starting point
        if skill_step == 0:
            q_init = np.asarray(q_subspace)
        else: # TODO double check my logic...
            # If first request not at skill start (e.g., resumed mid-skill across a mode 
            # boundary before any step-0 expansion) -> use initiation config to compute full 
            # tube from step 0
            q_init = np.asarray(skill_task.initiation_goal.sample(None))

        # We only need the joints belonging to this specific skill task
        task_joints = skill_task.skill.joints

        # 3. Monte Carlo rollouts: run N noisy executions
        results = [
            skill.rollout(q_init, skill_task, task_joints, self.env, t0=0.0)
            for _ in range(self.config.tube_rollouts)
        ]

        labels = np.array([r.branch_idx if r.branch_idx is not None else 0 for r in results])

        # 4. Group by branch first, then pad within the branch only
        tubes = {}
        for b in np.unique(labels):
            b_trajs = [r.trajectory for r, lab in zip(results, labels) if lab == b]
            
            # Find the maximum length in this branch
            b_max = max(len(traj) for traj in b_trajs)
            # Pad every trajectory up to b_max
            b_rollouts = np.stack([
                traj if len(traj) == b_max
                else np.vstack([traj, np.repeat(traj[-1:], b_max - len(traj), axis=0)])
                for traj in b_trajs
            ])
            tubes[b] = {
                "nominal": np.mean(b_rollouts, axis=0),
                "raw_rollouts": b_rollouts,
            }

        # 5. Cache it
        self._skill_tubes[key] = tubes
        return tubes

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

    def _expand_single_step(self, n_near: Node, q_target: Configuration, mode: Mode, skill_tasks: List) -> List[Node]:
        """
        Rolls out multiple concurrent skills by one step, with optional concurrent steering for the inactive robots.
        Inactive robots motions are bounded by max_vel*dt. Follows nominal path and checks against tube.
        """
        if not skill_tasks:
            return []
            
        for skill_task in skill_tasks:
            if self._is_skill_done(n_near, skill_task):
                return []

        dt = skill_tasks[0].skill.dt

        # 1. Get positions from all robots
        q_full = n_near.state.q.state().copy()
        q_target_vec = q_target.state().copy()

        # 2. Inactive robots
        active_indices = self._get_active_subspace_indices(skill_tasks)
        q_base = self._steer_inactive(q_full, q_target_vec, active_indices, dt)
        
        all_joints = self.env.get_joint_names()

        # 3. Stochastic tube preparation 
        tube_tasks = [t for t in skill_tasks if self._use_nominal_tube(t.skill)]
        tubes_by_task = {
            t.name: self._get_skill_tubes(
                t, q_full[self._get_active_subspace_indices([t])],
                n_near.state.skill_steps.get(t.name, 0)
            ) for t in tube_tasks
        }

        # 4. Apply skills (using nominal path for active stochastic robots)
        q_base_choice = q_base.copy()
        for skill_task in skill_tasks:
            skill = skill_task.skill
            task_name = skill_task.name
            task_indices = self._get_active_subspace_indices([skill_task])

            # Isolate just the active robot's joints
            q_subspace = q_full[task_indices]
            self.env.C.selectJoints(skill.joints)

            base_step = n_near.state.skill_steps.get(task_name, 0)

            if isinstance(skill, (BaseDeterministicTimedSkill, BaseStochasticTimedSkill)):
                n_steps = max(1, round(skill.duration / dt))
                
                # Avoid step past horizon
                if base_step >= n_steps:
                    self.env.C.selectJoints(all_joints)
                    continue
                # Stochastic: don't call skill.step() -> adds random noise
                # Instead force robot to follow center of uncertainty tube
                if self._use_nominal_tube(skill):
                    branch_idx = self._committed_branch(task_name, tubes_by_task[task_name])
                    nominal_traj = tubes_by_task[task_name][branch_idx]["nominal"]
                    q_subspace_new = nominal_traj[min(base_step + 1, len(nominal_traj) - 1)].copy()
                else:
                    # Deterministic skills just step normally
                    t_norm = min((base_step + 1) / n_steps, 1.0)
                    q_subspace_new = skill.step(t_norm, q_subspace, self.env)
            else: 
                q_subspace_new = skill.step(q_subspace, self.env)
            q_base_choice[task_indices] = q_subspace_new
        self.env.C.selectJoints(all_joints)

        # 5. Assemble the new state
        q_new = self.env.get_start_pos().from_flat(q_base_choice)
        
        # Keep track of how many steps each skill has taken so far
        new_skill_steps = dict(n_near.state.skill_steps)
        for skill_task in skill_tasks:
            new_skill_steps[skill_task.name] = new_skill_steps.get(skill_task.name, 0) + 1

        state_new = State(q_new, mode, is_skill_waypoint=True,
                        skill_steps=new_skill_steps)
     
        # 6. Collision checking (checking if nominal center path hit anything)
        if not self._validate(state_new, n_near.state.q, is_skill=True):
            return []
        
        # If nominal safe, check inflated path against the union of all branches
        if tube_tasks:
            steps = {t.name: n_near.state.skill_steps.get(t.name, 0) + 1 for t in tube_tasks}
            if not self._is_safe_against_tube(tube_tasks, tubes_by_task, steps, q_base_choice, mode):
                return []

        # 7. Add to tree
        n_new = self._create_and_add_node(state_new, n_near, mode, is_skill=True)
        n_new.state.skill_steps = new_skill_steps

        return [n_new]
 
    def _expand_kinodynamic(self, n_near: Node, q_target: Configuration, mode: Mode, skill_tasks: List) -> List[Node]:
        """
        Unrolls a skill execution for N stepy, constructing intermediate waypoints (stored in SkillEdge), 
        and evaluating collisions before appending the final state (only the end node enters the subtree 
        for NN search) to the RRT tree. 
        """
        if not skill_tasks:
            return []
            
        for skill_task in skill_tasks:
            if self._is_skill_done(n_near, skill_task):
                return []
                
        dt = skill_tasks[0].skill.dt
        n_kino = self.config.kinodynamic_steps
        active_indices = self._get_active_subspace_indices(skill_tasks)

        q_target_vec = q_target.state().copy()

        rollout_steps = n_kino
        skill_infos = []
        
        # 1. Setup: precompute bounds to make sure we don't try to unroll past the skill's end
        for task in skill_tasks:
            skill = task.skill
            base_step = n_near.state.skill_steps.get(task.name, 0)
            is_timed = isinstance(skill, (BaseDeterministicTimedSkill, BaseStochasticTimedSkill))
            n_total = max(1, round(skill.duration / dt)) if is_timed else 0
            
            if is_timed:
                if n_total - base_step <= 0: return []
                rollout_steps = min(rollout_steps, n_total - base_step)
                
            skill_infos.append({
                'skill': skill,
                'name': task.name,
                'indices': self._get_active_subspace_indices([task]),
                'base_step': base_step,
                'is_timed': is_timed,
                'n_total': n_total
            })

        all_joints = self.env.get_joint_names()

        # Fetch stochastic tubes for all active stochastic tasks
        tube_tasks = [t for t in skill_tasks if self._use_nominal_tube(t.skill)]
        tubes_by_task = {
            t.name: self._get_skill_tubes(
                t, n_near.state.q.state()[self._get_active_subspace_indices([t])],
                n_near.state.skill_steps.get(t.name, 0)
            ) for t in tube_tasks
        }

        q_curr = n_near.state.q.state().copy()
        waypoints = [q_curr.copy()]
        t_norms_list = [0.0]
        skill_done = False
        actual_steps = 0

        # 2. The unrolling loop
        for i in range(1, rollout_steps + 1):

            # Move inactive robots
            q_next = self._steer_inactive(q_curr, q_target_vec, active_indices, dt)
            
            # Move active robots
            for info in skill_infos:
                skill = info['skill']
                q_sub = q_curr[info['indices']]
                self.env.C.selectJoints(skill.joints)
                
                if info['is_timed']:
                    t_norm = min((info['base_step'] + i) / info['n_total'], 1.0)
                    
                    # Force stochastic skill to follow nominal baseline
                    if self._use_nominal_tube(skill):
                        branch_idx = self._committed_branch(info['name'], tubes_by_task[info['name']])
                        nominal_traj = tubes_by_task[info['name']][branch_idx]["nominal"]
                        q_sub_new = nominal_traj[min(info['base_step'] + i, len(nominal_traj) - 1)].copy()
                    else:
                        q_sub_new = skill.step(t_norm, q_sub, self.env)
                    if skill.done(t_norm, q_sub_new, self.env): skill_done = True
                else:
                    q_sub_new = skill.step(q_sub, self.env)
                    if skill.done(q_sub_new, self.env): skill_done = True
                
                q_next[info['indices']] = q_sub_new
            
            self.env.C.selectJoints(all_joints)

            # 3. Intermediate collision checking
            q_next_cfg = self.env.get_start_pos().from_flat(q_next)
            q_curr_cfg = self.env.get_start_pos().from_flat(q_curr)
            
            state_next = State(q_next_cfg, mode)
            if not self._validate(state_next, q_curr_cfg, is_skill=True):
                break

            # Inflated stochastic collision check (union over all branches)
            if tube_tasks:
                steps = {t.name: n_near.state.skill_steps.get(t.name, 0) + i for t in tube_tasks}
                if not self._is_safe_against_tube(tube_tasks, tubes_by_task, steps, q_next, mode):
                    break
            
            # Passed collision check -> record waypoint
            actual_steps = i
            waypoints.append(q_next.copy())
            t_norms_list.append(float(i))
            q_curr = q_next

            if skill_done:
                break # Skill finished early
        
        # 4. Final node creation
        if len(waypoints) < 2:
            return [] 
        
        q_end_cfg = self.env.get_start_pos().from_flat(waypoints[-1])
        
        # Dict for step tracking
        end_step_dict = dict(n_near.state.skill_steps)
        for skill_task in skill_tasks:
            end_step_dict[skill_task.name] = end_step_dict.get(skill_task.name, 0) + actual_steps
        state_new = State(q_end_cfg, mode, is_skill_waypoint=True, skill_steps=end_step_dict)

        # Put all itermediate waypoints into an Edge and compute true distance
        skill_edge = SkillEdge(waypoints=np.array(waypoints), t_norms=np.array(t_norms_list))
        edge_cost = self._skill_edge_cost(np.asarray(waypoints), mode)

        # Add single node to tree
        n_new = self._create_and_add_node(state_new, n_near, mode, is_skill=True, edge_cost_override=edge_cost)
        n_new.state.skill_steps = end_step_dict
        n_new.skill_edge = skill_edge
        
        return [n_new]

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

        # If no stochastic skills are active, that stat is valid (trivial)
        if not tube_tasks:
            return True

        q = state.q.state()
        steps = {}

        # 2. Extract the current timestep for each active skill
        for t in tube_tasks:
            steps[t.name] = getattr(state, "skill_steps", {}).get(t.name, 0)

        # 3. Ensure the newly shortcutted state doesn't violate the inflated margins
        # (using the union check over all branches)
        return self._is_safe_against_tube(tube_tasks, self._skill_tubes, steps, np.asarray(q), mode)

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
        )

        # Remove interpolated points used in shortcutting (collision check)
        return shortcutting.remove_interpolated_nodes(shortcut_path)

    def plan(self, ptc, optimize: bool = False):
        """
        Runs RRT* planning with conservative tube checks and returns the solution path and metadata
        """
        path, info = super().plan(ptc, optimize=optimize)
        info["skill_tubes"] = self._skill_tubes
        info["skill_branch_commitments"] = {
            name: self._committed_branch(name, tubes)
            for name, tubes in self._skill_tubes.items() if tubes
        }
        return path, info
