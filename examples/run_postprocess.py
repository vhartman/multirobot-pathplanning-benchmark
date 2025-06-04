import argparse
import plotly.graph_objects as go
import os
import numpy as np
import webbrowser
import time as time
import subprocess
import random
import re
import glob
from analysis_util import(
    colors_plotly,
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

from multi_robot_multi_goal_planning.problems import get_env_by_name
from run_experiment import load_experiment_config
from multi_robot_multi_goal_planning.problems.util import interpolate_path
from display_single_path import (
    load_path,
    convert_to_path
)

def sort_key(file_path):
    file_name = os.path.basename(file_path)  # e.g., "path_01.json"
    # Remove the prefix "path_" and the suffix ".json", then convert to an integer.
    number_str = file_name.replace("path_", "").replace(".json", "")
    return int(number_str)

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
    env.C.view_close()

def path_vis(env: BaseProblem, dir:str, path: List[State], framerate:int = 1, generate_png:bool = True, path_original:bool = False):
    if generate_png:
            if path_original:
                interpolated_path = path
            else:
                interpolated_path = interpolate_path(path, resolution=0.25)  
            path_as_png(env, interpolated_path, export = True, dir =  dir, framerate = framerate)
    # Generate a gif
    palette_file = os.path.join(dir, 'palette.png')
    output_gif = os.path.join(dir, 'out.gif')
    for file in [palette_file, output_gif]:
        if os.path.exists(file):
            os.remove(file)
    palette_command = [
        "ffmpeg",
        "-framerate", f"{framerate}",
        "-i", os.path.join(dir, "%04d.png"),
        "-vf", "scale=iw:-1:flags=lanczos,palettegen",
        palette_file
    ]

    # Command 2: Use palette.png to generate the out.gif
    gif_command = [
        "ffmpeg",
        "-framerate", f"{framerate}",
        "-i", os.path.join(dir, "%04d.png"),
        "-i", palette_file,
        "-lavfi", "scale=iw:-1:flags=lanczos [scaled]; [scaled][1:v] paletteuse=dither=bayer:bayer_scale=5",
        output_gif
    ]
    subprocess.run(palette_command, check=True)
    subprocess.run(gif_command, check=True)

def single_path_html(env, env_path, path, output_html):
    
    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
        try:
            task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  
        except Exception:
            task_sequence_text = "No sequence given"


    # Initialize figure and static elements
    fig = go.Figure()
    frames = []
    all_frame_traces = []
    colors = colors_plotly()
    # static traces
    static_traces = mesh_traces_env(env_path)
    static_traces.append(
            go.Mesh3d(
                x=[0],  # Position outside the visible grid
                y=[0],  # Position outside the visible grid
                z=[0],
                color = "white",
                name='',
                legendgroup='',
                showlegend=True  # This will create a single legend entry for each mode
            )
        )
    modes = []
    for s in path:
        mode = s.mode
        if mode.task_ids not in modes:
            modes.append(mode.task_ids)
        if env.is_terminal_mode(mode):
            break

    for robot_idx, robot in enumerate(env.robots):
        legend_group = robot
        static_traces.append(
        go.Mesh3d(
            x=[0],  # X-coordinates of the exterior points
            y=[0],  # Y-coordinates of the exterior points
            z=[0] ,  # Flat surface at z = 0
            color=colors[len(modes)+robot_idx],  # Fill color from the agent's properties
            opacity=1,  # Transparency level
            name=legend_group,
            legendgroup=legend_group,
            showlegend=True
        )
    )
    frame_traces = []
    for robot_idx, robot in enumerate(env.robots):
        indices = env.robot_idx[robot]
        legend_group = robot
        if path:
            path_x = [state.q[robot_idx][0] for state in path]
            path_y = [state.q[robot_idx][1] for state in path]
            path_z = [1 for state in path]

            
        else:
            start = env.start_pos.q[indices]
            path_x = [start[0]]
            path_y = [start[1]]
            path_z = [1]


        frame_traces.append(
            go.Scatter3d(
                x=path_x, 
                y=path_y,
                z=path_z,
                mode="lines+markers",
                line=dict(color=colors[len(modes)+robot_idx], width=6),
                marker=dict(
                    size=5,  # Very small markers
                    color=colors[len(modes)+robot_idx],  # Match marker color with line
                    opacity=1
                ),
                opacity=1,
                name=legend_group,
                legendgroup=legend_group,
                showlegend=False
            )
        )
    all_frame_traces.append(frame_traces)

    # Determine the maximum number of dynamic traces needed
    max_dynamic_traces = max(len(frame_traces) for frame_traces in all_frame_traces)

    fig.add_traces(static_traces)

    for _ in range(max_dynamic_traces):
        fig.add_trace(go.Mesh3d(x=[], y=[], z =[]))

    frames = []
    for idx, frame_traces in enumerate(all_frame_traces):
        while len(frame_traces) < max_dynamic_traces:
            frame_traces.append(go.Mesh3d(x=[], y=[], z= []))
        
        frame = go.Frame(
            data=frame_traces,
            traces=list(range(len(static_traces), len(static_traces) + max_dynamic_traces)),
            name=f"frame_{idx}"
        )
        frames.append(frame)

    fig.frames = frames

    # Animation settings
    animation_settings = dict(
        frame=dict(duration=100, redraw=True),  # Sets duration only for when Play is pressed
        fromcurrent=True,
        transition=dict(duration=0)
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, animation_settings]  # Starts animation only on button press
                ),
                dict(
                    label="Stop",
                    method="animate",
                    args=[
                        [None],  # Stops the animation
                        dict(frame=dict(duration=0, redraw=True), mode="immediate")
                    ]
                )
            ],
            direction="left",
            pad={"t": 10},
            showactive=False, #set auto-activation to false
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Frame: "),
            pad=dict(t=60),
            steps=[dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(i)
            ) for i, f in enumerate(fig.frames)]
        )],
        showlegend=True, 
        annotations=[
        dict(
            x=1,  # Centered horizontally
            y=-0.03,  # Adjusted position below the slider
            xref="paper",
            yref="paper",
            text=task_sequence_text,
            showarrow=False,
            font=dict(size=14),
            align="center",
        )
    ]
    )

    fig.write_html(output_html)
    print(f"Animation saved to {output_html}")

def path_evolution_html(env, env_path, dir_run, output_html, args):
    
    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
        try:
            task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  
        except Exception:
            task_sequence_text = "No sequence given" 

    # Initialize figure and static elements
    fig = go.Figure()
    frames = []
    all_frame_traces = []
    colors = colors_plotly()
    # static traces
    static_traces = mesh_traces_env(env_path)
    static_traces.append(
            go.Mesh3d(
                x=[0],  # Position outside the visible grid
                y=[0],  # Position outside the visible grid
                z=[0],
                color = "white",
                name='',
                legendgroup='',
                showlegend=True  # This will create a single legend entry for each mode
            )
        )
    files = glob.glob(os.path.join(dir_run, 'path_*.json'))

    sorted_files = sorted(files, key=sort_key)
    if len(sorted_files) > 50:
        indices = np.linspace(0, len(sorted_files) - 1, num=50, dtype=int)
        files_to_process = [sorted_files[i] for i in indices]
        if files_to_process[-1] != sorted_files[-1]:
            files_to_process.append(sorted_files[-1])
        sorted_files = files_to_process

    modes = []
    paths = []
    for dir_path_json in sorted_files:
        path_data = load_path(dir_path_json)
        path = convert_to_path(env, path_data)
        if args.insert_transition_nodes:
            # Add transition nodes to the path
            path = add_transition_nodes(path)
        paths.append(path)
        for s in path:
            mode = s.mode
            if mode.task_ids not in modes:
                modes.append(mode.task_ids)
            if env.is_terminal_mode(mode):
                break
    for robot_idx, robot in enumerate(env.robots):
        legend_group = robot
        static_traces.append(
        go.Mesh3d(
            x=[0],  # X-coordinates of the exterior points
            y=[0],  # Y-coordinates of the exterior points
            z=[0] ,  # Flat surface at z = 0
            color=colors[len(modes)+robot_idx],  # Fill color from the agent's properties
            opacity=1,  # Transparency level
            name=legend_group,
            legendgroup=legend_group,
            showlegend=True
        )
    )
    
    for path in paths:
        frame_traces = []
        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            legend_group = robot
            if path:
                path_x = [state.q[robot_idx][0] for state in path]
                path_y = [state.q[robot_idx][1] for state in path]
                path_z = [1 for state in path]

                
            else:
                start = env.start_pos.q[indices]
                path_x = [start[0]]
                path_y = [start[1]]
                path_z = [1]


            frame_traces.append(
                go.Scatter3d(
                    x=path_x, 
                    y=path_y,
                    z=path_z,
                    mode="lines+markers",
                    line=dict(color=colors[len(modes)+robot_idx], width=6),
                    marker=dict(
                        size=3,  # Very small markers
                        color=colors[len(modes)+robot_idx],  # Match marker color with line
                        opacity=1
                    ),
                    opacity=1,
                    name=legend_group,
                    legendgroup=legend_group,
                    showlegend=False
                )
            )
        all_frame_traces.append(frame_traces)

    # Determine the maximum number of dynamic traces needed
    max_dynamic_traces = max(len(frame_traces) for frame_traces in all_frame_traces)

    fig.add_traces(static_traces)

    for _ in range(max_dynamic_traces):
        fig.add_trace(go.Mesh3d(x=[], y=[], z =[]))

    frames = []
    for idx, frame_traces in enumerate(all_frame_traces):
        while len(frame_traces) < max_dynamic_traces:
            frame_traces.append(go.Mesh3d(x=[], y=[], z= []))
        
        frame = go.Frame(
            data=frame_traces,
            traces=list(range(len(static_traces), len(static_traces) + max_dynamic_traces)),
            name=f"frame_{idx}"
        )
        frames.append(frame)

    fig.frames = frames

    # Animation settings
    animation_settings = dict(
        frame=dict(duration=100, redraw=True),  # Sets duration only for when Play is pressed
        fromcurrent=True,
        transition=dict(duration=0)
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, animation_settings]  # Starts animation only on button press
                ),
                dict(
                    label="Stop",
                    method="animate",
                    args=[
                        [None],  # Stops the animation
                        dict(frame=dict(duration=0, redraw=True), mode="immediate")
                    ]
                )
            ],
            direction="left",
            pad={"t": 10},
            showactive=False, #set auto-activation to false
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Frame: "),
            pad=dict(t=60),
            steps=[dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(i)
            ) for i, f in enumerate(fig.frames)]
        )],
        showlegend=True, 
        annotations=[
        dict(
            x=1,  # Centered horizontally
            y=-0.03,  # Adjusted position below the slider
            xref="paper",
            yref="paper",
            text=task_sequence_text,
            showarrow=False,
            font=dict(size=14),
            align="center",
        )
    ]
    )

    fig.write_html(output_html)
    print(f"Animation saved to {output_html}")

def add_transition_nodes(path: List[State]) -> List[State]:
    """
    Adds transition nodes to the path.
    """
    path_w_doubled_modes = []
    for i in range(len(path)):
        path_w_doubled_modes.append(path[i])

        if i + 1 < len(path) and path[i].mode != path[i + 1].mode:
            path_w_doubled_modes.append(State(path[i].q, path[i + 1].mode))

    return path_w_doubled_modes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "do",
        choices=["single_path_html", "path_evolution_html", "final_paths", "single_path"],
        help="Select postprocess to be done",
    )
    parser.add_argument(
        "path",
        help="Select path to be processed",
    )
    parser.add_argument(
        "--insert_transition_nodes",
        action="store_true",
        help="Shortcut the path. (default: False)",
    )

   
    args = parser.parse_args()
    if args.path.endswith('.json'):
        assert args.do == "single_path_html" or args.do == "single_path", "Please give a json file for single path"
        dir_path_json = args.path
        dir = '/'.join(dir_path_json.split('/')[:-3]) 
        
        print("-----------------------------------------------------------------")
        print(dir)
        run = re.search(r'/(\d+)(?=/)', dir_path_json).group(1)
        dir_run = os.path.join(dir, run)
    elif re.search(r"/\d$", args.path): 
        assert args.do == "path_evolution_html", "Please give a folder for path evolution"
        dir_run = args.path
        dir = '/'.join(dir_run.split('/')[:-2]) 
        run = os.path.basename(dir_run)
        dir_path_json = None
    elif args.path is not None:
        assert args.do == "final_paths", "Please give a folder for final paths"
        dir = args.path
        dir_path_json = None
    else:
        #get latest created folder
        home_dir = os.path.expanduser("~")
        directory = os.path.join(home_dir, 'multirobot-pathplanning-benchmark/out')
        dir = get_latest_folder(directory)
        dir_path_json = None

    config_dir = os.path.join(dir, 'config.json')
    config = load_experiment_config(config_dir)
    seed = config["seed"] 
    np.random.seed(seed)
    random.seed(seed)
    env = get_env_by_name(config["environment"])    
    home_dir = os.path.expanduser("~")
    env_path = os.path.join(home_dir, f'env/{config["environment"]}')  
    save_env_as_mesh(env, env_path) 
    
    if args.do == "final_paths":
        #all final paths in one folder
        #only path_folder needs to be specified
        path_original = False
        generate_png = True
        framerate = 63
        if path_original:
            folder_name = "PathOriginal"
        else:
            folder_name = "Path"
        
        for planner in config['planners']:
            dir_planner = os.path.join(dir, planner['name'])
            dir_out = os.path.join(dir_planner, folder_name)
            os.makedirs(dir_out, exist_ok=True)
            for run in range(config['num_runs']):
                dir_run = os.path.join(dir_planner, str(run))
                dir_path_json = os.path.join(dir_run, sorted(os.listdir(dir_run))[-1])
                path_data = load_path(dir_path_json)
                path = convert_to_path(env, path_data)
                if args.insert_transition_nodes:
                    # Add transition nodes to the path
                    path = add_transition_nodes(path)
                dir_out_run = os.path.join(dir_out, str(run))
                os.makedirs(dir_out_run, exist_ok=True)
                path_vis(env, dir_out_run +"/", path, framerate=framerate, generate_png = generate_png, path_original= path_original)
    
    if args.do == "single_path":
        #one single path in a folder
        #path_filename needs to be specified
        path_original = False
        generate_png = True
        framerate = 15
        if path_original:
            folder_name = "PathOriginal"
        else:
            folder_name = "Path"
        try: 
            # planner_name = re.match(r'.*/out/[^/]+/([^/]+)(?:/|$)', dir_path_json).group(1)
            planner_name = ''.join(dir_path_json.split('/')[-3:-2]) 
            print(planner_name)
        except Exception:
            print("Please give dir to json file")
        dir_planner = os.path.join(dir, planner_name)
        dir_out = os.path.join(dir_planner, folder_name)
        print(dir_out)
        os.makedirs(dir_out, exist_ok=True)
        path_data = load_path(dir_path_json)
        path = convert_to_path(env, path_data)
        if args.insert_transition_nodes:
            # Add transition nodes to the path
            path = add_transition_nodes(path)
        run = re.search(r'/(\d+)(?=/)', dir_path_json).group(1)
        dir_out_run = os.path.join(dir_out, run)
        os.makedirs(dir_out_run, exist_ok=True)
        path_vis(env, dir_out_run +"/", path, framerate=framerate, generate_png = generate_png, path_original= path_original)

    if args.do == "single_path_html":
        #path_filename needs to be specified
        try: 
            planner_name = re.match(r'.*/out/[^/]+/([^/]+)(?:/|$)', dir_path_json).group(1)
        except Exception:
            print("Please give dir to json file")
        dir_planner = os.path.join(dir, planner_name)
        path_data = load_path(dir_path_json)
        path = convert_to_path(env, path_data)
        if args.insert_transition_nodes:
            # Add transition nodes to the path
            path = add_transition_nodes(path)
        file_name = os.path.basename(dir_path_json)
        modified_file_name = file_name.replace('.', '_')
        output_html = os.path.join(dir_planner, f'run_{run}_{modified_file_name}.html')
        single_path_html(env, env_path, path, output_html)    
        webbrowser.open('file://' + os.path.realpath(output_html))

    if args.do == "path_evolution_html":
        planner_name = os.path.basename('/'.join(dir_run.split('/')[:7])) 
        dir_planner = os.path.join(dir, planner_name)
        output_html = os.path.join(dir_planner, f'run_{run}_path_evolution.html')
        path_evolution_html(env, env_path, dir_run, output_html, args)    
        webbrowser.open('file://' + os.path.realpath(output_html))