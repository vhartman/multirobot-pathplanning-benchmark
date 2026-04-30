from multi_robot_multi_goal_planning.problems import (
    get_env_by_name,
    get_all_environments,
)
import multi_robot_multi_goal_planning.problems as problems
from multi_robot_multi_goal_planning.problems.rai_envs import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import Mode, State
from multi_robot_multi_goal_planning.problems.configuration import NpConfiguration

import numpy as np
import argparse
import time
import random


def display_disagreements(envs, states_per_env, labels, base_port=8080):
    """Display disagreement states in viser, one server per env on consecutive ports."""
    import threading

    threads = []
    for i, (env, states) in enumerate(zip(envs, states_per_env)):
        port = base_port + i
        print(f"Launching viser for env {i} on port {port} ({len(states)} paths)")
        t = threading.Thread(
            target=env.display_path_viser,
            kwargs=dict(
                paths=states,
                port=port,
                step_annotations=labels,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def benchmark_collision_checking(envs, N_config=10000, N_edge=10000, disp=False):
    conf_type = type(envs[0].get_start_pos())

    def sample_next_modes(env, mode: Mode):
        if env.is_terminal_mode(mode):
            return None
        failed_attemps = 0
        while True:
            if failed_attemps > 1000:
                print("Failed to sample next mode")
                return None
            possible_next_task_combinations = env.get_valid_next_task_combinations(mode)
            if len(possible_next_task_combinations) > 0:
                ind = random.randint(0, len(possible_next_task_combinations) - 1)
                active_task = env.get_active_task(
                    mode, possible_next_task_combinations[ind]
                )
            else:
                active_task = env.get_active_task(mode, None)

            goals_to_sample = active_task.robots

            goal_sample = active_task.goal.sample(mode)

            q = []
            for i in range(len(env.robots)):
                r = env.robots[i]
                if r in goals_to_sample:
                    offset = 0
                    for _, task_robot in enumerate(active_task.robots):
                        if task_robot == r:
                            q.append(
                                goal_sample[
                                    offset : offset + env.robot_dims[task_robot]
                                ]
                            )
                            break
                        offset += env.robot_dims[task_robot]
                else:  # uniform sample
                    lims = env.limits[:, env.robot_idx[r]]
                    if lims[0, 0] < lims[1, 0]:
                        qr = (
                            np.random.rand(env.robot_dims[r])
                            * (lims[1, :] - lims[0, :])
                            + lims[0, :]
                        )
                    else:
                        qr = np.random.rand(env.robot_dims[r]) * 6 - 3

                    q.append(qr)

            q = conf_type.from_list(q)

            if env.is_collision_free(q, mode):                
                next_modes = env.get_next_modes(q, mode)
                # assert len(next_modes) == 1
                # next_mode = next_modes[0]
                return next_modes
            else:
                failed_attemps += 1

    # Disagreement states: [env_0_states, env_1_states], [labels]
    disagree_states = [[] for _ in envs]
    disagree_labels = []

    modes_for_envs = []

    for env in envs:
        # create list of modes that we can reach
        print("Make mode list")
        reachable_modes = set([env.get_start_mode()])
        max_iter = 500
        for _ in range(max_iter):
            m_rnd = random.choice(tuple(reachable_modes))
            next_modes = sample_next_modes(env, m_rnd)
            # print(next_modes)
            if next_modes is not None:
                reachable_modes.update(next_modes)
        reachable_modes = tuple(reachable_modes)
        print("Found", len(reachable_modes), "reachable modes")

        modes_for_envs.append(reachable_modes)
    
    def config_benchmark():
        is_collision_free_rai = envs[0].is_collision_free
        is_collision_free_vamp = envs[1].is_collision_free

        rai_time = 0
        vamp_time = 0

        rai_config_coll_counter = 0
        vamp_config_coll_counter = 0

        # actually do the benchmarking
        print("Starting single sample benchmark")
        for _ in range(N_config):

            idx = random.randint(0, len(modes_for_envs[0])-1)

            m_rai = modes_for_envs[0][idx]
            
            for i in range(len(modes_for_envs[1])):
                if modes_for_envs[1][i].task_ids == m_rai.task_ids:
                    m_vamp = modes_for_envs[1][i]
                    break

            q = env.sample_config_uniform_in_limits()
        
            start = time.time()
            rai_res = is_collision_free_rai(q, m_rai)
            end = time.time()

            rai_time += end - start

            start = time.time()
            vamp_res = is_collision_free_vamp(q, m_vamp)
            end = time.time()

            vamp_time += end - start

            rai_config_coll_counter += rai_res
            vamp_config_coll_counter += vamp_res

            if rai_res != vamp_res:
                disagree_states[0].append(State(q, m_rai))
                disagree_states[1].append(State(q, m_vamp))
                disagree_labels.append(
                    f"cfg rai={'free' if rai_res else 'coll'} vamp={'free' if vamp_res else 'coll'}"
                )

        print(rai_config_coll_counter, vamp_config_coll_counter)

        print(f"Took on avg. {(rai_time) / N_config * 1000} ms for a rai collision check.")
        print(f"Took on avg. {(vamp_time) / N_config * 1000} ms for a vamp collision check.")

        

    def edge_benchmark():
        is_collision_free_rai = envs[0].is_collision_free
        is_collision_free_vamp = envs[1].is_collision_free
        
        is_edge_collision_free_rai = envs[0].is_edge_collision_free
        is_edge_collision_free_vamp = envs[1].is_edge_collision_free

        rai_time = 0
        vamp_time = 0

        rai_edge_coll_counter = 0
        vamp_edge_coll_counter = 0

        # actually do the benchmarking
        print("Starting edge sample benchmark")
        for _ in range(N_edge):

            idx = random.randint(0, len(modes_for_envs[0])-1)

            m_rai = modes_for_envs[0][idx]
            m_vamp = modes_for_envs[1][idx]

            while True:
                q1 = env.sample_config_uniform_in_limits()
                if is_collision_free_vamp(q1, m_vamp):
                    break

            while True:
                q2 = env.sample_config_uniform_in_limits()
                if is_collision_free_vamp(q2, m_vamp):
                    break
            
            start = time.time()
            rai_res = is_edge_collision_free_rai(q1, q2, m_rai)
            end = time.time()

            rai_time += end - start

            start = time.time()
            vamp_res = is_edge_collision_free_vamp(q1, q2, m_vamp)
            end = time.time()

            vamp_time += end - start

            rai_edge_coll_counter += rai_res
            vamp_edge_coll_counter += vamp_res

        print(rai_edge_coll_counter, vamp_edge_coll_counter)

        print(f"Took on avg. {(rai_time) / N_edge * 1000} ms for a rai collision check.")
        print(f"Took on avg. {(vamp_time) / N_edge * 1000} ms for a vamp collision check.")

    config_benchmark()
    edge_benchmark()

    if disp and any(disagree_states[0]):
        print(f"\nFound {len(disagree_states[0])} disagreements total. Launching viser...")
        display_disagreements(
            envs,
            disagree_states,
            disagree_labels,
            base_port=8080,
        )
    elif disp:
        print("No disagreements found — nothing to display.")

def main():

    np.random.seed(0)
    random.seed(0)
    env_rai = get_env_by_name("rai.ur5_box_stacking")
    
    np.random.seed(0)
    random.seed(0)
    env_vampmr = get_env_by_name("vampmr.ur5_box_stacking")

    benchmark_collision_checking([env_rai, env_vampmr], 10_000, 100, disp=True)


if __name__ == "__main__":
    main()
