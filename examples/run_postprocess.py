import argparse
import dill
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
import os
import numpy as np
import webbrowser
import time as time
import subprocess
from datetime import datetime
from analysis_util import(
    colors_plotly,
    count_files_in_folder,
    mesh_traces_env,
    get_latest_folder,
    save_env_as_mesh
    )
from typing import List
from multi_robot_multi_goal_planning.problems.planning_env import (
    State,
    BaseProblem
)

from multi_robot_multi_goal_planning.problems.rai_config import (
    get_robot_joints
)

# from analysis.check import(
#     nearest_neighbor, 
#     interpolation_check,
#     tree, 
#     graph
# )
from multi_robot_multi_goal_planning.problems import get_env_by_name
from run_experiment import load_experiment_config
from multi_robot_multi_goal_planning.problems.util import interpolate_path



def path_as_png(
    env,
    path: List[State],
    stop: bool = True,
    export: bool = False,
    pause_time: float = 0.05,
    dir: str = "./z.vid/", 
    framerate = 1
) -> None:
    for i in range(len(path)):
        env.set_to_mode(path[i].mode)
        for k in range(len(env.robots)):
            q = path[i].q[k]
            env.C.setJointState(q, get_robot_joints(env.C, env.robots[k]))
        if i == 0:
            env.C.view(True)
        else:
            env.C.view(False)

        if export:
            env.C.view_savePng(dir)

        time.sleep(pause_time)

    for i in range(2*framerate):
        if export:
            env.C.view_savePng(dir)

def path_vis(env: BaseProblem, vid_path:str, framerate:int = 1, generate_png:bool = True, path_original:bool = False):
    pkl_files = None
    if generate_png:
        with open(pkl_files[-1], 'rb') as file:
            data = dill.load(file)
            results = data["result"]
            path_ = results["path"]
            intermediate_tot = results["intermediate_tot"]
            transition = results["is_transition"]
            transition[0] = True
            modes = results["modes"]
            mode_sequence = []
            m = env.start_mode
            indices  = [env.robot_idx[r] for r in env.robots]
            while True:
                mode_sequence.append(m)
                if env.is_terminal_mode(m):
                    break
                m = env.get_next_mode(None, m)
            if path_original:
                discretized_path, _, _, _ = interpolate_path(path_, intermediate_tot, modes, indices, transition, resolution=None)
            else:
                discretized_path, _, _, _ = interpolate_path(path_, intermediate_tot, modes, indices, transition, resolution=0.01)  
            path_as_png(env, discretized_path, export = True, dir =  vid_path, framerate = framerate)
    # Generate a gif
    palette_file = os.path.join(vid_path, 'palette.png')
    output_gif = os.path.join(vid_path, 'out.gif')
    for file in [palette_file, output_gif]:
        if os.path.exists(file):
            os.remove(file)
    palette_command = [
        "ffmpeg",
        "-framerate", f"{framerate}",
        "-i", os.path.join(vid_path, "%04d.png"),
        "-vf", "scale=iw:-1:flags=lanczos,palettegen",
        palette_file
    ]

    # Command 2: Use palette.png to generate the out.gif
    gif_command = [
        "ffmpeg",
        "-framerate", f"{framerate}",
        "-i", os.path.join(vid_path, "%04d.png"),
        "-i", palette_file,
        "-lavfi", "scale=iw:-1:flags=lanczos [scaled]; [scaled][1:v] paletteuse=dither=bayer:bayer_scale=5",
        output_gif
    ]
    subprocess.run(palette_command, check=True)
    subprocess.run(gif_command, check=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--path",
        required=False,
        help="Select the path of the folder to be processed",
    )
    parser.add_argument(
        "--do",
        choices=["dev", "cost_single", "cost_multiple", "shortcutting_cost", "path", "sum", "interpolation", "nn", "tree", "graph", "final_path_animation", "ellipse"],
        required=True,
        help="Select postprocess to be done",
    )
   
    args = parser.parse_args()
    if args.path is not None:
        dir = args.path
    else:
        home_dir = os.path.expanduser("~")
        directory = os.path.join(home_dir, 'multirobot-pathplanning-benchmark/out')
        dir = get_latest_folder(directory)

    config_dir = os.path.join(dir, 'config.json')
    config = load_experiment_config(config_dir)
    env = get_env_by_name(config["environment"])    
    path =   os.path.join(home_dir, 'output')
    pkl_folder = get_latest_folder(path)
    env_path = os.path.join(home_dir, f'env/{config["environment"]}')  
    save_env_as_mesh(env, env_path) 

    # if args.do == "dev":
    #     print("Development")
    #     with_tree = True
    #     if with_tree:
    #         output_html = os.path.join(path, 'tree_animation_3d.html')
    #         reducer = 100
    #     else:
    #         output_html = os.path.join(path, 'path_animation_3d.html')
    #         reducer = 100
    #     developement_animation(env, env_path, pkl_folder, output_html, with_tree, reducer)    
    #     webbrowser.open('file://' + os.path.realpath(output_html))
    # if args.do == 'final_path_animation':
    #     save_env_as_mesh(env, env_path)
    #     pkl_folder = os.path.join(path, 'FramesFinalData')
    #     output_html = os.path.join(path, 'final_path_animation_3d.html')
    #     final_path_animation(env, env_path, pkl_folder, output_html)    
    #     webbrowser.open('file://' + os.path.realpath(output_html))
        
    # if args.do == "cost_single":
    #     pkl_folder = os.path.join(path, 'FramesFinalData')
    #     output_filename_cost = os.path.join(path, 'Cost.png')
    #     cost_single(env, pkl_folder, output_filename_cost)

    # if args.do == "cost_multiple":
    #     dir_planner = []
    #     for planner in config["planners"]:
    #         dir_planner.append(os.path.join(dir, planner['name']))
    #     with_inital_confidence_interval = False
    #     pkl_folder = os.path.join(path, 'FramesFinalData')
    #     output_filename_cost = os.path.join(os.path.dirname(dir), f'Cost_{datetime.now().strftime("%d%m%y_%H%M%S")}.png')
    #     cost_multiple(env, config, pkl_folder ,dir_planner ,output_filename_cost, with_inital_confidence_interval)
    
    # if args.do == "sum":
    #     sum(dir, False)
    
    # if args.do == "shortcutting_cost":
    #     fix_axis = False
    #     output_filename_cost = os.path.join(path, 'ShortcuttingCost.png')
    #     shortcutting_cost(env, path, output_filename_cost)
    
    if args.do == "final_path_animation":
        path_original = False
        generate_png = True
        if path_original:
            output_filename_path = os.path.join(path,"PathOriginal/")
            vid_path = os.path.join(path,"PathOriginal/")
        else:
            output_filename_path = os.path.join(path,"Path/")
            vid_path = os.path.join(path,"Path/")
        os.makedirs(vid_path, exist_ok=True)
        path_vis(env, vid_path, framerate=63, generate_png = generate_png, path_original= path_original)

#     if args.do == "ellipse":
#         ellipse(env, env_path, pkl_folder)    

# #TO CHECK
#     if args.do == "interpolation":
#         interpolation_check(env)
#     if args.do == "nn":
#         with_tree = True
#         if with_tree:
#             output_html = os.path.join(path, 'tree_animation_3d.html')
#             reducer = 50
#         else:
#             output_html = os.path.join(path, 'path_animation_3d.html')
#             reducer = 400
#         nearest_neighbor(env, env_path, pkl_folder, output_html, with_tree, reducer)    
#     if args.do == "tree":
#         tree(env, env_path, pkl_folder)  
#     if args.do == "graph":
#         output_html = os.path.join(path, 'graph.html')
#         graph(env, env_path, pkl_folder, output_html)    
#         webbrowser.open('file://' + os.path.realpath(output_html))
 


