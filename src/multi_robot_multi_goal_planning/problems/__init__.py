import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import src.multi_robot_multi_goal_planning.problems.rai_envs as re
import src.multi_robot_multi_goal_planning.problems.rai_single_goal_envs as rsge

def get_env_by_name(name):
    # fmt: off
    environment_configs = {
        # 2D Environments
        "piano": lambda: re.rai_two_dim_simple_manip(),
        "simple_2d": lambda: re.rai_two_dim_env(),
        "simple_2d_no_rot": lambda: re.rai_two_dim_env(agents_can_rotate=False),
        "hallway": lambda: re.rai_hallway_two_dim(),
        "hallway_no_rot": lambda: re.rai_hallway_two_dim(agents_can_rotate=False),
        "random_2d": lambda: re.rai_random_two_dim(),
        "random_2d_no_rot": lambda: re.rai_random_two_dim(agents_can_rotate=False),
        "2d_handover": lambda: re.rai_two_dim_handover(),
        "three_agents": lambda: re.rai_two_dim_three_agent_env(),
        
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

        "box_rearrangement": lambda: re.rai_ur10_arm_box_rearrangement_env(), # 2 robots, 9 boxes
        "box_rearrangement_only_five": lambda: re.rai_ur10_arm_box_rearrangement_env(num_boxes=5),
        "box_rearrangement_four_robots": lambda: re.rai_ur10_arm_box_rearrangement_env(num_robots=4),
    
        # single goal envs
        "two_dim_single_goal": lambda: rsge.rai_two_dim_env(),
        "two_dim_single_goal_no_rot": lambda: rsge.rai_two_dim_env(agents_can_rotate=False),
        "random_2d_single_goal": lambda: rsge.rai_random_two_dim(),
        "random_2d_single_goal_no_rot": lambda: rsge.rai_random_two_dim(agents_can_rotate=False),
        "hallway_single_goal": lambda: rsge.rai_hallway_two_dim(),
        "hallway_single_goal_no_rot": lambda: rsge.rai_hallway_two_dim(agents_can_rotate=False),
    }
    # fmt: on

    if name not in environment_configs:
        raise ValueError(f"Unknown environment name: {name}")

    return environment_configs[name]()