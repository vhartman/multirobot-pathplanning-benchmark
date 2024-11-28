import argparse
import sys
import os
import dill
from analysis_util import *
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # One level up
sys.path.append(parent_folder)
from rai_envs import *
from planning_env import *
from util import *
from rai_config import *
from planning_env import *
from util import *


def interpolation(waypoints:  List[List], num_points: int=50):
    waypoints = np.array(waypoints)
    interpolated = []
    for i in range(len(waypoints) - 1):
        segment = np.linspace(waypoints[i], waypoints[i+1], num_points)
        interpolated.extend(segment)
    return np.array(interpolated)    

def path_simulation(env: base_env, pkl_folder, interpolate:bool=False, shortcutt: List[List] = None):
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    with open(pkl_files[-1], 'rb') as file:
        data = dill.load(file)
        results = data["result"]
        path_data = results["path"]
    C = ry.Config()
    C.addConfigurationCopy(env.C)
    if interpolate and shortcutt is None:
        path_data = interpolation(path_data)
        sleep_time = 0.05
    elif shortcutt is not None:
        path_data = interpolation(shortcutt, 2)
        sleep_time = 0.05
    else:
        sleep_time = 2
    for q in path_data:
        C.setJointState(q)
        C.view()
        time.sleep(sleep_time)
    time.sleep(10)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--sim",
        choices=["path", "waypoints", "path_shortcutt"],
        required=True,
        help="Select the mode of operation",
    )
    args = parser.parse_args()
    home_dir = os.path.expanduser("~")
    directory = os.path.join(home_dir, 'output')
    path = get_latest_folder(directory)
    env_name, config_params, path_data, path_post_data = get_config(path)
    pkl_folder = os.path.join(path, 'FramesData')
    

    env = get_env_by_name(env_name)    
    


    if args.sim == "path":
        print("Simulating interpolated path")
        path_simulation(env, pkl_folder, interpolate =True)
    if args.sim == "waypoints":
        print("Simulating path waypoints")
        path_simulation(env, pkl_folder)
    if args.sim == "path_shortcutt":
        print("Simulating path waypoints")
        path_simulation(env, pkl_folder, shortcutt = path_post_data )