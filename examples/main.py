import cProfile
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # Current file's directory
project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
src_path = os.path.abspath(os.path.join(project_root, "../src"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from multi_robot_multi_goal_planning.planners.planner_rrtstar import *
from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar import *
from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_parallelized import *
from multi_robot_multi_goal_planning.planners.planner_rrtstar_parallelized import *
from multi_robot_multi_goal_planning.planners.planner_q_rrtstar import *
from multi_robot_multi_goal_planning.problems.rai_envs import *
from multi_robot_multi_goal_planning.problems import get_env_by_name



def execute_planner(env, args, config_manager):
    if args.planner == "rrtstar":
        rrt_star = RRTstar(env, config_manager)
        path = rrt_star.Plan()
    
    if args.planner == "rrtstar_par":
        rrt_star = ParallelizedRRTstar(env, config_manager)
        path = rrt_star.Plan()
    
    if args.planner == "q_rrtstar":
        bi_rrt_star = QRRTstar(env, config_manager)
        path = bi_rrt_star.Plan()

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
    parser.add_argument("--planner", choices=["rrtstar", "bi_rrtstar", "bi_rrtstar_par", "q_rrtstar", "rrtstar_par"], required=True, help="Planner type")
    args = parser.parse_args()

    config_manager = ConfigManager('config.yaml')
    if config_manager.use_seed and config_manager.amount_of_runs == 1:
        np.random.seed(config_manager.seed)
        random.seed(config_manager.seed)
    config_manager.planner = args.planner
    if config_manager.amount_of_runs == 1 and not config_manager.debug_mode:
        config_manager.log_params(args)
    env = get_env_by_name(args.env_name)
    for run in range(config_manager.amount_of_runs):
        if config_manager.amount_of_runs > 1 and not config_manager.debug_mode:
            seed = random.randint(0, 2**32 - 1)
            np.random.seed(seed)
            random.seed(seed)
            config_manager.seed = seed
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
        
        gc.collect()
        torch.cuda.empty_cache()
        config_manager.reset_logger()  # Reset the logger for the next run
        print()
if __name__ == "__main__":
    main()