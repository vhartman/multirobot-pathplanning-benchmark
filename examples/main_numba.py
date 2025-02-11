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

from multi_robot_multi_goal_planning.planners.rrtstar_base_numba import *
from multi_robot_multi_goal_planning.planners.planner_rrtstar_numba import RRTstar
from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_numba import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.planner_bi_rrtstar_parallelized_numba import ParallelizedBidirectionalRRTstar
# from multi_robot_multi_goal_planning.planners.planner_rrtstar_parallelized import ParallelizedRRTstar
# from multi_robot_multi_goal_planning.planners.planner_q_rrtstar import QRRTstar
from multi_robot_multi_goal_planning.planners.planner_drrtstar_numba import dRRTstar
from multi_robot_multi_goal_planning.problems.rai_envs import *
from multi_robot_multi_goal_planning.problems import get_env_by_name
from analysis.postprocessing import cost_single, final_path_animation
from analysis.analysis_util import save_env_as_mesh
import multiprocessing

def execute_planner(env, args, config_manager):
    if args.planner == "rrtstar":
        planner = RRTstar(env, config_manager)
    
    # if args.planner == "rrtstar_par":
    #     planner = ParallelizedRRTstar(env, config_manager)
    
    # if args.planner == "q_rrtstar":
    #     planner = QRRTstar(env, config_manager)

    if args.planner == "bi_rrtstar":
        planner = BidirectionalRRTstar(env, config_manager)

    if args.planner == "bi_rrtstar_par":
        planner = ParallelizedBidirectionalRRTstar(env, config_manager)
    
    if args.planner == "drrtstar":
        planner = dRRTstar(env, config_manager)
    
    path = planner.Plan()
    if path == []:
        print('No path was found')
        return

    print('Path is collision free:', env.is_path_collision_free(path))
    path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(path)}
    config_manager.logger.info("Path: %s", json.dumps(path_dict, indent=4))
    pkl_folder = os.path.join(config_manager.output_dir, 'FramesFinalData')
    output_dir = os.path.join(config_manager.output_dir, 'Cost.png')
    cost_single(env, pkl_folder, output_dir)
    output_html = os.path.join(config_manager.output_dir, 'final_path_animation_3d.html')
    env_path = os.path.join(os.path.expanduser("~"), f'env/{args.env_name}')
    if len(path[0].q.state())/(len(env.robots)) <= 3: # Only applicable up to 2D env with orientation
        save_env_as_mesh(env, env_path)
        final_path_animation(env, env_path, pkl_folder, output_html)  


def single_run(run_id, args, config_manager):
    env = get_env_by_name(args.env_name)

    #logic for seeding and output directory setup
    if config_manager.use_seed and config_manager.amount_of_runs == 1:
        seed = config_manager.seed
    elif config_manager.use_seed and config_manager.amount_of_runs < 20: 
        seed = run_id +1
    else:
        seed = int(time.time() * 1000000) % (2**32) + run_id
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    config_manager.seed = seed
    if config_manager.amount_of_runs > 1:
        config_manager.output_dir = os.path.join(
            os.path.expanduser("~"), 
            f'output/{args.env_name}_{config_manager.timestamp}/{run_id}'
        )


    config_manager.log_params(args)
    print(f"======= Run {run_id+1}/{config_manager.amount_of_runs} started =======")
    if config_manager.cprofiler:
        with cProfile.Profile() as profiler:
            execute_planner(env, args, config_manager)
        profiler.dump_stats(os.path.join(config_manager.output_dir, 'results.prof'))
        os.system(f"snakeviz {os.path.join(config_manager.output_dir, 'results.prof')}")
    else:
        execute_planner(env, args, config_manager)

    print(f"======= Run {run_id+1}/{config_manager.amount_of_runs} terminated =======")
    gc.collect()
    torch.cuda.empty_cache()
    config_manager.reset_logger()
    time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Environment Viewer")
    parser.add_argument("env_name", nargs="?", default="default", help="Environment to show")
    parser.add_argument("--planner", choices=["rrtstar", "bi_rrtstar", "bi_rrtstar_par", "q_rrtstar", "rrtstar_par", "drrtstar"], required=True, help="Planner type")
    args = parser.parse_args()

    #initilize config manager
    config_manager = ConfigManager('config.yaml')
    config_manager.planner = args.planner
    multiprocessing.set_start_method("spawn")
    for run_id in range(config_manager.amount_of_runs):
        process = multiprocessing.Process(target=single_run, args=(run_id, args, config_manager))
        process.start()
        process.join() 

if __name__ == "__main__":
    main()