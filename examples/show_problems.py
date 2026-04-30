from multi_robot_multi_goal_planning.problems import (
    get_env_by_name,
    get_all_environments,
)
import multi_robot_multi_goal_planning.problems as problems
from multi_robot_multi_goal_planning.problems.rai.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import Mode, State
from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration
from multi_robot_multi_goal_planning.problems.util import compute_reachable_modes

import numpy as np
import argparse
import time
import random


def _mode_annotation(env, q_config, m, task) -> str:
    """Build a markdown annotation string for a single mode step."""
    conf_type = type(env.get_start_pos())
    lines = []

    if task.name is not None:
        lines.append(f"**Task:** {task.name}")
    lines.append(f"**Robots:** {', '.join(task.robots)}")

    collision_free = env.is_collision_free(q_config, m)
    lines.append(f"**Collision free:** {'✓' if collision_free else '✗'}")

    if task.constraints:
        lines.append("**Task constraints:**")
        task_ok = True
        for constraint in task.constraints:
            ok = constraint.is_fulfilled(q_config, m, env)
            res = constraint.F(q_config.state(), m, env)
            task_ok = task_ok and ok
            lines.append(f"- {'✓' if ok else '✗'} residual: `{np.round(res, 4).tolist()}`")
        lines.append(f"**Task constraints fulfilled:** {'✓' if task_ok else '✗'}")

    if env.constraints:
        lines.append("**Env constraints:**")
        env_ok = True
        for constraint in env.constraints:
            ok = constraint.is_fulfilled(q_config, m, env)
            res = constraint.F(q_config.state(), m, env)
            env_ok = env_ok and ok
            lines.append(f"- {'✓' if ok else '✗'} residual: `{np.round(res, 4).tolist()}`")
        lines.append(f"**Env constraints fulfilled:** {'✓' if env_ok else '✗'}")

    return "  \n".join(lines)


def visualize_modes(env: rai_env, export_images: bool = False, use_viser: bool = False):
    
    if not use_viser:
        env.show()

    q_home = env.start_pos
    conf_type = type(env.get_start_pos())

    # Accumulated for viser display
    viser_states = []
    viser_annotations = []

    if use_viser:
        viser_states.append(State(env.start_pos, env.start_mode))
        viser_annotations.append("Home")

    m = env.start_mode
    while True:
        print("--------")
        print("Mode", m)

        q = []
        next_task_combos = env.get_valid_next_task_combinations(m)
        if len(next_task_combos) > 0:
            idx = random.randint(0, len(next_task_combos) - 1)
            task = env.get_active_task(m, next_task_combos[idx])
        else:
            task = env.get_active_task(m, None)
        switching_robots = task.robots

        if task.name is not None:
            print("Active Task name:", task.name)
        print("Involved robots: ", task.robots)

        if task.is_skill:
            task.skill.joints = []
            for r in task.robots:
                task.skill.joints.extend(env.robot_joints[r])
            
            all_joints = []
            for r in env.robots:
                all_joints.extend(env.robot_joints[r])
            
            env.C.selectJoints(task.skill.joints)
            skill_result = task.skill.rollout(env.C.getJointState(), task, all_joints, env, 0)
            env.C.selectJoints(all_joints)

            goal_sample = skill_result.trajectory[-1]
        else:
            goal_sample = task.goal.sample(m)
    
        print("Goal state:")
        print(goal_sample)

        print("switching robots: ", switching_robots)

        for j, r in enumerate(env.robots):
            if r in switching_robots:
                # TODO: need to check all goals here
                # figure out where robot r is in the goal description
                offset = 0
                for _, task_robot in enumerate(task.robots):
                    if task_robot == r:
                        q.append(
                            goal_sample[offset : offset + env.robot_dims[task_robot]]
                        )
                        break
                    offset += env.robot_dims[task_robot]
                # q.append(goal_sample)
            else:
                q.append(q_home.robot_state(j))

        print("Goal state (all robots)")
        print(q)

        q_config = conf_type.from_list(q)

        print(
            "Is collision free: ",
            env.is_collision_free(q_config, m),
        )
        task_constraints_fulfilled = True
        for constraint in task.constraints:
            if not constraint.is_fulfilled(q_config, m, env):
                task_constraints_fulfilled = False
            print("Residual:", constraint.F(q_config.state(), m, env))

        print(
            "Fulfills task constraints: ", task_constraints_fulfilled
        )

        env_constraints_fulfilled = True
        for constraint in env.constraints:
            if not constraint.is_fulfilled(q_config, m, env):
                env_constraints_fulfilled = False
            print("Residual:", constraint.F(q_config.state(), m, env))

        print(
            "Fulfills env constraints: ", env_constraints_fulfilled
        )

        # colls = env.C.getCollisions()
        # for c in colls:
        #     if c[2] < 0:
        #         print(c)

        if use_viser:
            viser_states.append(State(q_config, m))
            viser_annotations.append(_mode_annotation(env, q_config, m, task))
        elif export_images:
            env.show(False)
            env.C.view_savePng("./z.img/")
        else:
            env.show_config(q_config)

        if env.is_terminal_mode(m):
            break

        ms = env.get_next_modes(q_config, m)
        assert len(ms) == 1
        m = ms[0]

    if use_viser:
        env.display_path_viser(
            viser_states,
            primitives_only=True,
            step_annotations=viser_annotations,
        )
    elif hasattr(env, "close"):
        env.close()
    
def benchmark_collision_checking(env: rai_env, N=10000):
    print("Make mode list")
    reachable_modes = compute_reachable_modes(env)
    print("Found", len(reachable_modes), "reachable modes")
    
    is_collision_free = env.is_collision_free

    # actually do the benchmarking
    print("Starting benchmark")
    start = time.time()
    for _ in range(N):
        q = env.sample_config_uniform_in_limits()
        m = random.choice(reachable_modes)

        is_collision_free(q, m)

    end = time.time()

    print(f"Took on avg. {(end - start) / N * 1000} ms for a collision check.")


def main():
    # problems.rai_envs.rai_hallway_two_dim_dependency_graph()
    # print()
    # problems.rai_envs.rai_two_dim_three_agent_env_dependency_graph()

    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "show", "modes", "list_all"],
        required=True,
        help="Select the mode of operation",
    )
    parser.add_argument(
        "--show_coll_config",
        action="store_true",
        help="Display the configuration used for collision checking. (default: False)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export images of modes. (default: False)",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Use viser for visualization (rai envs only). (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    args = parser.parse_args()

    if args.mode == "list_all":
        all_envs = get_all_environments()
        print("\n".join([name for name in all_envs.keys()]))
        return

    # check_all_modes()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env_by_name(args.env_name)

    # make use of the original config
    if not args.show_coll_config and isinstance(env, rai_env):
        env.C_base = env.C_orig
        env.C = env.C_orig

    if args.mode == "show":
        print("Environment starting position")
        env.show()
    elif args.mode == "benchmark":
        benchmark_collision_checking(env)
    elif args.mode == "modes":
        print("Environment modes/goals")
        visualize_modes(env, args.export, use_viser=args.viser)


if __name__ == "__main__":
    main()
