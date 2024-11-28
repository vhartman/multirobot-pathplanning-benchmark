import os
import json
import cProfile
from rrtstar_planner import *
from rai_envs import *
from util import *

def execute_planner(args, config_manager):
    env = get_env_by_name(args.env_name)
    env.show()
    
    if args.planner == "rrtstar":
        rrt_star = RRTstar(env, config_manager)
        path = rrt_star.Plan()
        path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(path)}
        config_manager.logger.info("Path: %s", json.dumps(path_dict, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Environment Viewer")
    parser.add_argument("env_name", nargs="?", default="default", help="Environment to show")
    parser.add_argument("--planner", choices=["rrtstar"], required=True, help="Planner type")
    args = parser.parse_args()

    config_manager = ConfigManager('config.yaml')
    config_manager.log_params(args)

    if config_manager.cprofiler:
        with cProfile.Profile() as profiler:
            execute_planner(args, config_manager)
        profiler.dump_stats(os.path.join(config_manager.output_dir, 'results.prof'))
        os.system(f"snakeviz {os.path.join(config_manager.output_dir, 'results.prof')}")
    else:
        execute_planner(args, config_manager)

if __name__ == "__main__":
    main()