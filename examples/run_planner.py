import argparse
from matplotlib import pyplot as plt

from multi_robot_multi_goal_planning.problems.rai_envs import get_env_by_name
from multi_robot_multi_goal_planning.problems.rai_envs import display_path

# planners
from multi_robot_multi_goal_planning.planners.prioritized_planner import *

# np.random.seed(100)


def main():
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env", nargs="?", default="default", help="env to show")

    args = parser.parse_args()

    env = get_env_by_name(args.env)

    env.show()

    path, info = prioritized_planning(env)
    print("Checking original path for validity")
    print(env.is_valid_plan(path))

    print('cost', info['costs'])
    print('comp_time', info['times'])

    print("displaying original path")
    # discretized_path = discretize_path(path)
    display_path(env, path, stop=False)

if __name__ == "__main__":
    # make_2d_rai_env_3_agents()

    main()
