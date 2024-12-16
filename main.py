import os
import json
import cProfile
from planner_rrtstar import *
from planner_bi_rrtstar import *
from planner_bi_rrtstar_parallelized import *
from rai_envs import *
from util import *

def execute_planner(env, args, config_manager):
    if args.planner == "rrtstar":
        rrt_star = RRTstar(env, config_manager)
        path = rrt_star.Plan()

    if args.planner == "bi_rrtstar":
        bi_rrt_star = BidirectionalRRTstar(env, config_manager)
        path = bi_rrt_star.Plan()

    if args.planner == "bi_rrtstar_par":
        bi_rrt_star = ParallelizedBidirectionalRRTstar(env, config_manager)
        path = bi_rrt_star.Plan()
    print('Path is collision free:', env.is_path_collision_free(path))
    if config_manager.debug_mode:
        return
    path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(path)}
    config_manager.logger.info("Path: %s", json.dumps(path_dict, indent=4))
    

def main():
    parser = argparse.ArgumentParser(description="Environment Viewer")
    parser.add_argument("env_name", nargs="?", default="default", help="Environment to show")
    parser.add_argument("--planner", choices=["rrtstar", "bi_rrtstar", "bi_rrtstar_par"], required=True, help="Planner type")
    args = parser.parse_args()

    config_manager = ConfigManager('config.yaml')
    if config_manager.amount_of_runs < 2 and not config_manager.debug_mode:
        config_manager.log_params(args)
    env = get_env_by_name(args.env_name)
    for run in range(config_manager.amount_of_runs):
        if config_manager.amount_of_runs > 1 and not config_manager.debug_mode:
            config_manager.output_dir = os.path.join(os.path.expanduser("~"), f'output/{args.env_name}_{config_manager.timestamp}/{run}')
            config_manager.log_params(args)
        if config_manager.cprofiler:
            with cProfile.Profile() as profiler:
                execute_planner(env, args, config_manager)
            profiler.dump_stats(os.path.join(config_manager.output_dir, 'results.prof'))
            os.system(f"snakeviz {os.path.join(config_manager.output_dir, 'results.prof')}")
        else:
            execute_planner(env, args, config_manager)

        print(f'======= Run {run+1}/{config_manager.amount_of_runs} terminated =======')
        print()
if __name__ == "__main__":
    main()