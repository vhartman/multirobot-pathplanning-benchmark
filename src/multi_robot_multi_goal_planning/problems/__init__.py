import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import multi_robot_multi_goal_planning.problems.rai_envs as re
import multi_robot_multi_goal_planning.problems.rai_single_goal_envs as rsge


def get_all_environments():
    # fmt: off
    environment_configs = {
        # 2D Environments
        "piano": lambda: re.rai_two_dim_simple_manip(),
        "simple_2d": lambda: re.rai_two_dim_env(),
        "simple_2d_no_rot": lambda: re.rai_two_dim_env(agents_can_rotate=False),
        "hallway": lambda: re.rai_hallway_two_dim(),
        "hallway_no_rot": lambda: re.rai_hallway_two_dim(agents_can_rotate=False),
        "other_hallway": lambda: re.rai_alternative_hallway_two_dim(),
        "other_hallway_no_rot": lambda: re.rai_alternative_hallway_two_dim(),
        "random_2d": lambda: re.rai_random_two_dim(),
        "random_2d_three_goals": lambda: re.rai_random_two_dim(num_goals=3),
        "random_2d_two_goals": lambda: re.rai_random_two_dim(num_goals=2),
        "random_2d_one_goals": lambda: re.rai_random_two_dim(num_goals=1),
        "random_2d_no_rot": lambda: re.rai_random_two_dim(agents_can_rotate=False),
        "2d_handover": lambda: re.rai_two_dim_handover(),
        "three_agents": lambda: re.rai_two_dim_three_agent_env(),

        # 2D with neighborhood
        "single_agent_mover": lambda: re.rai_two_dim_single_agent_neighbourhood(),
        
        # Envs without obstacles, used to test optimality convergence
        "one_agent_many_goals": lambda: re.rai_two_dim_env_no_obs(),
        "one_agent_many_goals_no_rot": lambda: re.rai_two_dim_env_no_obs(agents_can_rotate=False),
        "three_agent_many_goals": lambda: re.rai_two_dim_env_no_obs_three_agents(),
        "three_agent_many_goals_no_rot": lambda: re.rai_two_dim_env_no_obs_three_agents(agents_can_rotate=False),

        # Arm Environments
        "box_sorting": lambda: re.rai_ur10_arm_pick_and_place_env(),
        "eggs": lambda: re.rai_ur10_arm_egg_carton_env(),
        "eggs_five_only": lambda: re.rai_ur10_arm_egg_carton_env(5),
        "bottles": lambda: re.rai_ur10_arm_bottle_env(),
        "handover": lambda: re.rai_ur10_handover_env(),        
        "triple_waypoints": lambda: re.rai_multi_panda_arm_waypoint_env(num_robots=3, num_waypoints=5),
        "welding": lambda: re.rai_quadruple_ur10_arm_spot_welding_env(),
        "simplified_welding": lambda: re.rai_quadruple_ur10_arm_spot_welding_env(num_robots=2, num_pts=2),
        "box_stacking": lambda: re.rai_ur10_arm_box_stack_env(),
        "box_stacking_two_robots": lambda: re.rai_ur10_arm_box_stack_env(num_robots=2),
        "box_stacking_three_robots": lambda: re.rai_ur10_arm_box_stack_env(num_robots=3),
        "box_reorientation": lambda: re.rai_ur10_box_pile_cleanup_env(),
        "box_reorientation_multi_handover": lambda: re.rai_ur10_box_pile_cleanup_env(make_many_handover_poses=True),

        "box_rearrangement": lambda: re.rai_ur10_arm_box_rearrangement_env(), # 2 robots, 9 boxes
        "box_rearrangement_only_five": lambda: re.rai_ur10_arm_box_rearrangement_env(num_boxes=5),
        "box_rearrangement_four_robots": lambda: re.rai_ur10_arm_box_rearrangement_env(num_robots=4),
        "crl_four": lambda: re.rai_ur10_arm_box_rearrangement_env(num_robots=4, logo=True), # 2 robots, 9 boxes
        "crl_two": lambda: re.rai_ur10_arm_box_rearrangement_env(num_robots=2, logo=True), # 2 robots, 9 boxes

        # mobile
        "mobile_wall_four": lambda: re.rai_mobile_manip_wall(num_robots=4),
        "mobile_wall_three": lambda: re.rai_mobile_manip_wall(num_robots=3),
        "mobile_wall_two": lambda: re.rai_mobile_manip_wall(num_robots=2),

        # single goal envs
        "two_dim_single_goal": lambda: rsge.rai_two_dim_env(),
        "two_dim_single_goal_no_rot": lambda: rsge.rai_two_dim_env(agents_can_rotate=False),
        "random_2d_single_goal": lambda: rsge.rai_random_two_dim(),
        "random_2d_single_goal_no_rot": lambda: rsge.rai_random_two_dim(agents_can_rotate=False),
        "hallway_single_goal": lambda: rsge.rai_hallway_two_dim(),
        "hallway_single_goal_no_rot": lambda: rsge.rai_hallway_two_dim(agents_can_rotate=False),

        # single robot, single goal: debugging
        "two_dim_single_robot_single_goal": lambda: rsge.rai_random_two_dim_single_agent(),
        "single_panda_arm_single_goal": lambda: rsge.rai_single_panda_arm_single_goal_env(),
        "single_agent_box_stacking": lambda: re.rai_ur10_arm_box_rearrangement_env(num_robots=1, num_boxes=2),

        # 3d single goal envs
        "multi_agent_panda_single_goal": lambda: rsge.rai_multi_panda_arm_single_goal_env(),
        "handover_single_goal": lambda: rsge.rai_ur10_handover_env(),

        ##### DEPENDENCY GRAPHS
        "hallway_dep": lambda: re.rai_hallway_two_dim_dependency_graph(),
        "other_hallway_dep": lambda: re.rai_alternative_hallway_two_dim_dependency_graph(),
        "piano_dep": lambda: re.rai_two_dim_simple_manip_dependency_graph(),
        "2d_handover_dep": lambda: re.rai_two_dim_handover_dependency_graph(),
        "two_agents_many_goals_dep": lambda: re.rai_two_dim_env_no_obs_dep_graph(),
        "two_agents_many_goals_dep_no_rot": lambda: re.rai_two_dim_env_no_obs_dep_graph(agents_can_rotate=False),
        "three_agent_many_goals_dep": lambda: re.rai_two_dim_three_agent_env_dependency_graph(),
        "mobile_dep": lambda: re.rai_mobile_manip_wall_dep(),
        "mobile_five_dep": lambda: re.rai_mobile_manip_wall_dep(num_robots=5),
        "mobile_four_dep": lambda: re.rai_mobile_manip_wall_dep(num_robots=4),
        "mobile_three_dep": lambda: re.rai_mobile_manip_wall_dep(num_robots=3),
        "mobile_two_dep": lambda: re.rai_mobile_manip_wall_dep(num_robots=2),
        "box_stacking_dep": lambda: re.rai_ur10_arm_box_stack_env_dep(),
        "box_stacking_three_robots_dep": lambda: re.rai_ur10_arm_box_stack_env_dep(num_robots=3),
        "box_reorientation_dep": lambda: re.rai_ur10_box_pile_cleanup_env_dep(),
        "box_reorientation_handover_set_dep": lambda: re.rai_ur10_box_pile_cleanup_env_dep(make_many_handover_poses=True),

    }
    # fmt: on

    return environment_configs


def get_env_by_name(name):
    environment_configs = get_all_environments()

    if name not in environment_configs:
        raise ValueError(f"Unknown environment name: {name}")

    return environment_configs[name]()
