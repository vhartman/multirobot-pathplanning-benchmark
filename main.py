from rai_envs import *
import cProfile
from datetime import datetime
from planner import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument("env_name", nargs="?", default="default", help="env to show")
    parser.add_argument(
        "--planner",
        choices=["RRTstar"],
        required=True,
        help="Select the type of planner",
    )
    args = parser.parse_args()
    time = datetime.now().strftime("%H%M%S")
    timestamp = datetime.now().strftime("%d%m%y") + "_" + time
    directory = f'./output/evaluation/{timestamp}/'
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    with cProfile.Profile() as profiler:
        env = get_env_by_name(args.env_name)
        if args.planner == "RRTstar":
            print("RRTstar")
            user_defined_metrics
            rrt_star = RRTstar(env, user_defined_metrics)
            env.show()
    profiler.dump_stats(directory + 'results.prof')
    os.system(f"snakeviz {directory + 'results.prof'}")