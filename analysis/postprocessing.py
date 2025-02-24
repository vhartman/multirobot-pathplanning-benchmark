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
from analysis.analysis_util import(
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
from analysis.shortcutting import(
    init_discretization
)
from analysis.check import(
    nearest_neighbor, 
    interpolation_check,
    tree, 
    graph
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from examples.run_experiment import load_experiment_config


def get_cost_label(cost_function, number_agents):
    if cost_function == 1:
        cost_label = 'sum'
    if cost_function == 2:
        cost_label = 'max'
    return cost_label

def cost_single(env, pkl_folder, output_filename):
    """Plot cost of single run"""
    # Get the list of .pkl files
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    num_agents = len(env.robots)
    cost_function= 2 # TODO hardcoded
    cost_label = get_cost_label(cost_function, num_agents-1)
    costs = []
    time = []
    init_sol = False
    for pkl_file in pkl_files:
        fig = go.Figure()
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
        result = data["result"]
        time.append(data["time"])
        
        all_init_path = data["all_init_path"]
        if all_init_path and not init_sol:
            init_sol = True # iteration where first iteration was found
            init_path_time = time[-1]
        if init_sol:
            
            cost = result["total"]
            if cost is None:
                costs.append(1e6)
            else:
                costs.append(cost.item())
            
               
        else:
            costs.append(1e6)
            

    
    fig = go.Figure()
    costs_discretized, times_discretized,  _, _ = step_function(init_path_time, time[-1], 1, [time], [costs])



    # Plot the cumulative total cost data
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='lines',
        name= "",
        line=dict(color='white'),
        showlegend=True  # Ensure "Total Cost" is in the legend
    ))
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name= f"Initial sol.found after {np.round(init_path_time,3)}s",
        line=dict(color='white'),
        showlegend=True  # Ensure "Total Cost" is in the legend
    ))
    max_cost = np.nanmax([cost for cost in costs if cost < 1e5])
    max_time = time[-1]
   
    fig.update_layout(
        title=f"Cost({cost_label}) vs Time",
        xaxis=dict(type='log', title="Time [s]",range = [np.log10(time[0]), np.log10(max_time +100)], autorange= False),
        yaxis=dict(title="Cost [m]", range=[0, max_cost+0.5], autorange=False ), 
        margin=dict(l=0, r=50, t=50, b=50),  # Increase right margin for legend space
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1,
            traceorder="normal"  # Position legend outside the plot area on the right
        ) 
    )
    
    fig.write_image(output_filename)

def step_function(start_time, sum_times, runs, times, costs, resolution = 0.01):
    """Step function
    Returns: Median and the nonparametric 99% confidence interval"""
    stop_time = sum_times/runs # get the average stop time (all have similar/same end time)
    max_len = int((stop_time - start_time) / resolution) + 1 # +1 to include endpoint
    average_time = np.linspace(start_time, stop_time, max_len, endpoint = True)
    average_costs = []
    for idx, cost in enumerate(costs):
        cost_average = []
        for t in average_time:
            if times[idx][-1] < t:
                cost_average.append(cost[-1])
            for c_idx, _ in enumerate(cost):
                if times[idx][c_idx] == t:
                    cost_average.append(cost[c_idx])
                    break
                elif times[idx][c_idx] > t:
                    cost_average.append(cost[c_idx-1])
                    break
                    
        average_costs.append(cost_average)
    if runs == 1:
        # if average_costs[0][0] < 1e4:
        #     average_costs[0][0] = 1e10
        return average_costs[0], average_time, _, _
    #np.median calculates element-wise medians (so for each timestamp)
    return np.median(average_costs, axis=0).tolist(), average_time, np.percentile(average_costs, 1, axis=0).tolist(), np.percentile(average_costs, 99, axis=0).tolist()

def cost_multiple(env, config, pkl_folder, parent_folders, output_filename, with_inital_confidence_interval):
    """Plot cost of multiple runs with upper and lower bounds (= min/max values)""" # TODO for several different planners 
    fig = go.Figure() 
    colors = colors_plotly()
    max_cost_average = []
    min_cost_average = []
    max_time_average = []
    min_time_average = []
    
    for planner, parent_folder in enumerate(parent_folders):
        planner_costs = []
        planner_times = []
        planner_init_times = []
        planner_init_costs = []
        
        min_init_times = np.inf
        sum_terminal_times = 0
        runs = 0
        for number in range(config['num_runs']):
            runs += 1
            costs_path = os.path.join(parent_folder, 'costs.txt')
            with open(costs_path, 'r', encoding='utf-8') as file:
                for current_line, line in enumerate(file, start=0):
                    if current_line == number:
                        costs =[float(value) for value in line.strip().split(',') if value]
            time_path = os.path.join(parent_folder, 'timestamps.txt')
            with open(time_path, 'r', encoding='utf-8') as file:
                for current_line, line in enumerate(file, start=0):
                    if current_line == number:
                        time = [float(value) for value in line.strip().split(',') if value]
            if time[0] < min_init_times:
                min_init_times = time[0]     
            planner_init_times.append(time[0])
            planner_init_costs.append(costs[0])  
            costs.insert(0,1e6)
            time.insert(0,0)

            num_agents = len(env.robots)
            cost_function= 2 # TODO hardcoded
            cost_label = get_cost_label(cost_function, num_agents-1)
            planner_costs.append(costs)
            planner_times.append(time)
            sum_terminal_times += time[-1]
    
        median_cost, average_time, lb_costs, up_costs = step_function(min_init_times, sum_terminal_times, runs, planner_times, planner_costs)
        # average_time = [t+1 for t in average_time]
        median_inital_cost = np.median(planner_init_costs)
        median_inital_time = np.median(planner_init_times)
        lb_inital_time = np.percentile(planner_init_times, 1, axis=0).tolist() 
        ub_inital_time = np.percentile(planner_init_times, 99, axis=0).tolist() 
        lb_inital_cost = np.percentile(planner_init_costs, 1, axis=0).tolist() 
        ub_inital_cost = np.percentile(planner_init_costs, 99, axis=0).tolist() 
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='lines',
            name= f"{config['planners'][planner]['name']}",
            line=dict(color=colors[planner]),
            showlegend=True  
        ))
        
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            name= f"with ⌀ initial sol.found after {np.round(np.mean(planner_init_times),3)}s",
            line=dict(color='white'),
            showlegend=True 
        ))
        # max_cost = np.nanmax(costs)
        max_time_average.append(average_time[-1])
        if with_inital_confidence_interval:
            min_time_average.append(lb_inital_time)
        else:
            min_time_average.append(average_time[0])
        max_cost_ = np.nanmax([cost for cost in up_costs if cost < 1e5])
        if with_inital_confidence_interval:
            max_cost_average.append(np.nanmax([max_cost_, ub_inital_cost]))
        else:
            max_cost_average.append(np.nanmax(max_cost_))
        min_cost_average.append(np.min(lb_costs))
        # Update layout with titles, axis labels, and legend
        fill_opacity = 0.15  # Adjust this value for the desired opacity (20% in this case)
        rgb_color = tuple(int(c * 255) for c in to_rgb(colors[planner]))  # to_rgb scales to [0, 1], so multiply by 255
        rgba_fill_color = f"rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, {fill_opacity})"  # Convert hex to RGB
        fig.add_trace(go.Scatter(
            x=average_time,
            y=median_cost,
            mode='lines', 
            opacity=1,
            line=dict(color=colors[planner],width = 2),
            showlegend=False  
        ))
        fig.add_trace(go.Scatter(
            x=average_time,
            y=up_costs,
            mode='lines',
            line=dict(color=colors[planner], width = 0.5),
            opacity=0.1,
            fill=None,  # No fill above this line
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=average_time,
            y=lb_costs,
            mode='lines',
            line=dict(color=colors[planner], width = 0.5),
            opacity=0.1,
            fill='tonexty',  # Fill the area between this line and the previous one
            showlegend=False,
            fillcolor=rgba_fill_color,
        ))
        if with_inital_confidence_interval:
            fig.add_trace(go.Scatter(
            x=[median_inital_time],  # Median initial time (square dot x-coordinate)
            y=[median_inital_cost],  # Median initial cost (square dot y-coordinate)
            mode='markers',  # Only 'markers' since error bars are handled separately
            marker=dict(size=7, symbol='square', color=colors[planner]),  # Square dot
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ub_inital_time - median_inital_time],  # Upper bound for time
                arrayminus=[median_inital_time - lb_inital_time],  # Lower bound for time
                thickness=2, width=2
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ub_inital_cost - median_inital_cost],  # Upper bound for cost
                arrayminus=[median_inital_cost - lb_inital_cost],  # Lower bound for cost
                thickness=2, width=2
            ),
            name='Median Initial Solution',  # Label for the legend
            showlegend=False
        ))
        
    fig.add_trace(go.Scatter(
        x=[None], 
        y=[None], 
        mode='markers',
        name= "",
        line=dict(color='white'),
        showlegend=True 
    ))
    fig.add_trace(go.Scatter(
        x=[None], 
        y=[None],  
        mode='none',  # No line, marker, or text will be displayed
        name=cost_label,  
        showlegend=True  # Ensures this entry appears in the legend
    ))
    max_time = np.max(max_time_average)
    min_time = np.min(min_time_average)
    max_cost = np.max(max_cost_average)
    min_cost = np.min(min_cost_average)

    fig.update_layout(
        title="Cost vs Time",
        xaxis=dict(type = 'log', title="Time [s]", range=[np.log10(min_time/2), np.log10(max_time+100)], autorange=False ),
        yaxis=dict(title="Cost [m]", range=[min_cost-0.1, max_cost +5], autorange=False ), 
        margin=dict(l=0, r=50, t=50, b=50),  # Increase right margin for legend space
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1,
            traceorder="normal"  # Position legend outside the plot area on the right
        ) 
    )
    # fig.update_layout(
    #     title="Cost vs Time",
    #     xaxis=dict(type = 'log', title="Time [s]", range=[np.log10(600), np.log10(910)], autorange=False ),
    #     yaxis=dict(title="Cost [m]", range=[min_cost-0.1, 13], autorange=False ), 
    #     margin=dict(l=0, r=50, t=50, b=50),  # Increase right margin for legend space
    #     legend=dict(
    #         orientation="v",
    #         yanchor="top",
    #         y=1,
    #         xanchor="left",
    #         x=1,
    #         traceorder="normal"  # Position legend outside the plot area on the right
    #     ) 
    # )

    fig.write_image(output_filename)

def sum(parent_folder, single_folder):
    """Get average of inital time/cost and terminal time/cost over several runs"""
    init_sol_time = []
    init_sol_cost = []
    terminal_cost = []
    terminal_time = []
    if not single_folder:
        for folder_name in os.listdir(parent_folder):
            print(folder_name)
            folder_path = os.path.join(parent_folder, folder_name)
            pkl_folder =  os.path.join(folder_path, 'FramesData')
            pkl_files = sorted(
                [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )
            for pkl_file in pkl_files:
                with open(pkl_file, 'rb') as file:
                    data = dill.load(file)
                all_init_path = data["all_init_path"]
                if all_init_path:
                    init_sol_time.append(data["time"])
                    init_sol_cost.append(data["result"]["total"].item())
                    break
            pkl_file = pkl_files[-1]
            with open(pkl_file, 'rb') as file:
                    data = dill.load(file)
            terminal_time.append(data["time"])
            terminal_cost.append(data["result"]["total"].item())  
            print(data["result"]["total"].item())  
    else:
        pkl_folder =  os.path.join(parent_folder, 'FramesData')
        pkl_files = sorted(
            [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as file:
                data = dill.load(file)
            all_init_path = data["all_init_path"]
            if all_init_path:
                init_sol_time.append(data["time"])
                init_sol_cost.append(data["result"]["total"].item())
                break
        pkl_file = pkl_files[-1]
        with open(pkl_file, 'rb') as file:
                data = dill.load(file)
        terminal_time.append(data["time"])
        terminal_cost.append(data["result"]["total"].item())  
        print(data["result"]["total"].item())  

    print("Initial solution time", np.mean(init_sol_time))
    print("Initial solution cost", np.mean(init_sol_cost))
    print("Terminal time", np.mean(terminal_time))
    print("Terminal cost", np.mean(terminal_cost))
    # print("Variation of terminal cost", np.var(terminal_cost, ddof=1))

def discretization_shortcutting(sum_times, runs, max_len, times, costs):
    stop_time = sum_times/runs # get the average stop time
    average_time = np.linspace(0, stop_time, max_len, endpoint = True)
    average_costs = []
    for idx, cost in enumerate(costs):
        cost_average = []
        for t in average_time:
            if t == 0:
                cost_average.append(cost[0])
                continue
            if times[idx][-1] < t:
                cost_average.append(cost[-1])
            for c_idx, _ in enumerate(cost):
                if times[idx][c_idx] > t:
                    cost_average.append(cost[c_idx-1])
                    break
                    
        average_costs.append(cost_average)
    
    return np.mean(average_costs, axis=0).tolist(), average_time

def shortcutting_cost(env, path, output_filename):
    # Initialize lists and Plotly objects
    costs = []
    times = []
    colors_list = []
    labels_list = []
    labels_bool = []
    fig = go.Figure()
    num_agents = len(env.robots)
    colors = colors_plotly() 
    cost_function = 2 #TODO hardcoded
    cost_label = get_cost_label(cost_function, num_agents - 1) 
    labels = ["Shortcutting per mode", "Partial single dim per mode", "Partial random subset per mode", 
              "Shortcutting per task (1 robot) Prob", "Partial single dim per task (1 robot) Prob", "Partial random subset per task (1 robot) Prob", 
              "Shortcutting per task (all possible robot) Prob", "Partial single dim per task (all possible robot) Prob", "Partial random subset per task (all possible robot) Prob",
              "Shortcutting per task (1 robot) Deter", "Partial single dim per task (1 robot) Deter", "Partial random subset per task (1 robot) Deter", 
              "Shortcutting per task (all possible robot) Deter", "Partial single dim per task (all possible robot) Deter", "Partial random subset per task (all possible robot) Deter" ]
    

    # Loop through versions to process data
    for version in range(9):
        bool_legend = True
        folder = os.path.join(path, f"Post_{version}")
        # Get sorted list of .pkl files in the folder
        try:
            pkl_files = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
            print(f"Found {len(pkl_files)} .pkl files in {folder}.")
        except Exception:
            continue
        
        # Retrieve environment and configuration details
        
        sum_times = 0
        max_len = 0
        version_times = []
        version_costs = []
        # Process each .pkl file
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as file:
                data = dill.load(file)
            data["total_cost"][0] = data["total_cost"][0].item()
            # Append data for flattening later
            time = data["time"]
            times.append(time) 
            version_times.append(time)
            costs.append(data["total_cost"])
            version_costs.append(data["total_cost"])
            colors_list.append(colors[version])
            labels_list.append(labels[version])
            labels_bool.append(bool_legend)
            bool_legend = False
            sum_times += data["time"][-1]
            if len(data["time"]) > max_len:
                max_len = len(data["time"])



        average_cost, average_time = discretization_shortcutting(sum_times, len(pkl_files), max_len, version_times, version_costs)
        average_time = [t+1 for t in average_time]
        fig.add_trace(go.Scatter(
            x=np.log10(average_time),
            y=average_cost,
            mode='lines',
            name=labels[version],  
            opacity=1,
            line=dict(color=colors[version], width=5),  # Adjust 'width' to make the line thicker
            showlegend=True  
        ))

            
        
    times = [[time + 1 for time in sublist] for sublist in times]
    for time_series, cost_series, color, label, label_bool in zip(times, costs, colors_list, labels_list, labels_bool):

        fig.add_trace(go.Scatter(
            x=np.log10(time_series),
            y=cost_series,
            mode='lines',
            name=label,  
            opacity = 0.15,
            line=dict(color=color),  # Assign a single color to this series
            showlegend=False  # Ensure the legend entry is shown
        ))
    
    # Flatten the nested lists for cost and time
    flattened_costs = [item for sublist in costs for item in sublist]
    flattened_times = [item for sublist in times for item in sublist]


    # Calculate maximum values for axes range
    max_cost = np.nanmax(flattened_costs) if flattened_costs else 0
    min_cost = np.nanmin(flattened_costs) if flattened_costs else 0
    max_time = np.nanmax(flattened_times) if flattened_times else 0
    min_time = np.nanmin(flattened_times) if flattened_times else 1

    fig.add_trace(go.Scatter(
                x=[np.log10(min_time)],
                y=[min_cost],
                name=cost_label,  
                mode='markers',
                marker=dict(size=0.001, color="white"),
                showlegend=True  
            ))
    # Update layout with titles, axis labels, and legend
    fig.update_layout(
        title="Cost vs Time",
        xaxis=dict(title="Log(time) [s]",range=[np.log10(min_time), np.log10(max_time)], autorange=False, tickmode="auto",),
        yaxis=dict(title="Cost [m]", range=[min_cost -0.01, max_cost + 0.01], autorange=False),
        margin=dict(l=0, r=50, t=50, b=50), 
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1 
        ),
        
    )
    
    # Save the plot as an image
    fig.write_image(output_filename)

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
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
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
                discretized_path, _, _, _ =init_discretization(path_, intermediate_tot, modes, indices, transition, resolution=None)
            else:
                discretized_path, _, _, _ =init_discretization(path_, intermediate_tot, modes, indices, transition, resolution=0.01)  
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

# Only possible for env 2/3 DOFs
def developement_animation(env, env_path, pkl_folder, output_html, with_tree, divider = None):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  


    count = count_files_in_folder(pkl_folder)
    if count > 150 and divider is not None:
        frame = int(count / divider)
    else:
        frame = 1
    print(f'Take every {frame}th frame')
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )[::frame]
    pkl_files.append(os.path.join(pkl_folder, sorted(
        [os.path.basename(f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(x)[0])
    )[-1]))
    print(f'Take {len(pkl_files)} .pkl files')

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
    static_traces.append(
                go.Scatter3d(
                    x=[0],  # Position outside the visible grid
                    y=[0],  # Position outside the visible grid
                    z=[0],
                    mode="markers",
                    marker=dict(
                        size=0.01,  # Very small markers
                        color="red",  # Match marker color with line
                        opacity=1
                    ),
                    opacity=1,
                    name="Transitions",
                    legendgroup="Transitions",
                    showlegend=True
                )
            )
    
    modes = []
    mode = env.start_mode
    while True:     
            modes.append(mode.task_ids)
            if env.is_terminal_mode(mode):
                break
            mode = env.get_next_mode(None, mode)
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
    # if with_tree:
    legends = []
    for idx in range(len(modes)):
        name = f"Mode: {modes[idx]}"
        legends.append(name)
        static_traces.append(
            go.Mesh3d(
                x=[0],  # Position outside the visible grid
                y=[0],  # Position outside the visible grid
                z=[0],
                name=name,
                color = colors[idx],
                legendgroup=name,  # Unique legend group for each mode
                showlegend=True  # This will create a single legend entry for each mode
            )
        )
    
    # dynamic_traces
    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
            tree = data["tree"]
            results = data["result"]
            path = results["path"]
            transition = results["is_transition"]
            inter_results = data["inter_result"]
            inter_paths = inter_results[0]['path']

        frame_traces = []

        # Collect traces for the path (if any)
        if inter_paths is None:
            for robot_idx, robot in enumerate(env.robots):
                indices = env.robot_idx[robot]
                legend_group = robot
                if path:
                    path_x = [state[indices][0] for state in path]
                    path_y = [state[indices][1] for state in path]
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
                if path:
                    try:
                        transition_x = [state[indices][0] for idx, state in enumerate(path) if transition[idx]]
                        transition_y = [state[indices][1] for idx, state in enumerate(path) if transition[idx]]
                        transition_z = [1] * len(path_x)

                        frame_traces.append(
                            go.Scatter3d(
                                x=transition_x, 
                                y=transition_y,
                                z=transition_z,
                                mode="markers",
                                marker=dict(
                                    size=5,  
                                    color="red", 
                                    opacity=1
                                ),
                                opacity=1,
                                name="Transitions",
                                legendgroup="Transitions",
                                showlegend=False
                            )
                        )
                    except Exception:
                        pass
        else:
            for robot_idx, robot in enumerate(env.robots):
                indices = env.robot_idx[robot]
                legend_group = robot
                for inter_result in inter_results:
                    path = inter_result['path']
                    mode = inter_result['modes'][-1]
                    mode_idx = modes.index(mode)
                    path_x = [state[indices][0] for state in path]
                    path_y = [state[indices][1] for state in path]
                    path_z = [1 for _ in path]
                    frame_traces.append(
                        go.Scatter3d(
                            x=path_x, 
                            y=path_y,
                            z=path_z,
                            mode="lines+markers",
                            line=dict(color=colors[mode_idx], width=6),
                            marker=dict(
                                size=3,  # Very small markers
                                color=colors[mode_idx],  # Match marker color with line
                                opacity=1
                            ),
                            opacity=1,
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=False
                        )
                        )

        if with_tree:
            for robot_idx, robot in enumerate(env.robots):
                indices = env.robot_idx[robot]
                lines_by_color = {}

                for node in tree:                    
                    q = node["state"]
                    parent_q = node["parent"]  # Get the parent node (or None)
                    mode = node["mode"]
                    mode_idx = modes.index(mode)
                    # mode = node["mode"]
                    color = colors[mode_idx]
                    x0 = q[indices][0]
                    y0 = q[indices][1]

                    if parent_q is not None:
                        x1 = parent_q[indices][0]
                        y1 = parent_q[indices][1]
                    else:
                        x1 = x0
                        y1 = y0

                    if color not in lines_by_color:
                        lines_by_color[color] = {'x': [], 'y': [], 'legend_group': legends[mode_idx]}
                    lines_by_color[color]['x'].extend([x0, x1, None])
                    lines_by_color[color]['y'].extend([y0, y1, None])

                for color, line_data in lines_by_color.items():
                    legend_group = line_data['legend_group']
                    frame_traces.append(
                        go.Scatter3d(
                            x=line_data['x'],
                            y=line_data['y'],
                            z = [1] * len(line_data['x']),
                            mode='markers + lines',
                            marker=dict(size=0.8, color=color),
                            line=dict(color=color, width=8),
                            opacity=0.1,
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=False
                        )
                    )

        #informed sampling 
        # try:
        #     if config['informed_sampling'] != []:
        #         informed = data["informed_sampling"]
        #         for informed_sampling in informed:
        #             n_rand = data['n_rand']
        #             for robot_idx, robot in enumerate(env.robots):
        #                 r_indices = env.robot_idx[robot] 
        #                 if informed_sampling['L'] is not None:
        #                     if len(env.robots) > 1:
        #                         if robot_idx in informed_sampling['L'].keys():
        #                             color_informed = colors[len(modes)+robot_idx]
        #                             C = informed_sampling['C'][robot_idx]
        #                             L = informed_sampling['L'][robot_idx]
        #                             center = informed_sampling['center'][robot_idx]
        #                             focal_points = [informed_sampling['start'][robot_idx].state(), informed_sampling['goal'][robot_idx].state()]
        #                             mode = informed_sampling['mode']
        #                             mode_idx = modes.index(mode)
        #                             q_rand = n_rand[r_indices]
        #                     else: 
                                
        #                         color_informed = 'black'
        #                         C = informed_sampling['C']
        #                         L = informed_sampling['L']
        #                         center = informed_sampling['center']
        #                         focal_points = [informed_sampling['start'].state.q.state(), informed_sampling['goal'].state.q.state()]
        #                         mode = informed_sampling['mode']
        #                         mode_idx = modes.index(mode)
        #                         q_rand = n_rand


        #                     if C.shape[0] == 3:  # 3D case
        #                         theta = np.linspace(0, 2 * np.pi, 100)
        #                         phi = np.linspace(0, np.pi, 50)
        #                         theta, phi = np.meshgrid(theta, phi)

        #                         # Generate unit sphere points
        #                         x = np.sin(phi) * np.cos(theta)
        #                         y = np.sin(phi) * np.sin(theta)
        #                         z = np.cos(phi)
                                

        #                         # Transform the unit sphere into an ellipsoid
        #                         unit_sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
        #                         ellipsoid_transformed = C @ L @ unit_sphere
                                

        #                         # Translate to center
        #                         x_ellipsoid = ellipsoid_transformed[0, :] + center[0]
        #                         y_ellipsoid = ellipsoid_transformed[1, :] + center[1]
        #                         z_ellipsoid = ellipsoid_transformed[2, :] + center[2] + 1


        #                         # Add 3D ellipsoid using Mesh3d
        #                         frame_traces.append(
        #                             go.Mesh3d(
        #                                 x=x_ellipsoid,
        #                                 y=y_ellipsoid,
        #                                 z=z_ellipsoid,
        #                                 alphahull=0,  # Ensure it forms a hull
        #                                 color=color_informed,  # Grey color for the surface
        #                                 opacity=0.1,  # High transparency
        #                                 name=legends[mode_idx],
        #                                 legendgroup = legends[mode_idx],
        #                                 showlegend = False,
        #                             )
        #                         )
        #                     if C.shape[0] == 2:  # 2D case
        #                         theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles for ellipse
                                
        #                         # Generate unit circle points
        #                         x = np.cos(theta)
        #                         y = np.sin(theta)
        #                         unit_circle = np.array([x, y])
                                
        #                         # Transform the unit circle into an ellipse
        #                         ellipse_transformed = C @ L @ unit_circle
                                
        #                         # Translate to center
        #                         x_ellipse = ellipse_transformed[0, :] + center[0]
        #                         y_ellipse = ellipse_transformed[1, :] + center[1]
        #                         z_ellipse = np.ones_like(x_ellipse)  # Set z height to 1
                                
        #                         # Add 2D ellipse using Scatter3d to position it at z = 1
        #                         frame_traces.append(
        #                             go.Scatter3d(
        #                                 x=x_ellipse,
        #                                 y=y_ellipse,
        #                                 z=z_ellipse,
        #                                 mode='lines',
        #                                 line=dict(color=color_informed, width=2),
        #                                 name=legends[mode_idx],
        #                                 legendgroup=legends[mode_idx],
        #                                 showlegend=False,
        #                             )
        #                         )



        #                     # Add focal points (3D)
        #                     for i, f in enumerate(focal_points):
        #                         frame_traces.append(go.Scatter3d(
        #                             x=[f[0]],
        #                             y=[f[1]],
        #                             z = [1],
        #                             mode='markers',
        #                             name=f'Focal Point {i+1} (2D)',
        #                             marker=dict(size=8, color=color_informed),
        #                             legendgroup = legends[mode_idx],
        #                             showlegend = False,
        #                         ))

        #                     # Add center (2D)
        #                     frame_traces.append(go.Scatter3d(
        #                         x=[center[0]],
        #                         y=[center[1]],
        #                         z = [1],
        #                         mode='markers',
        #                         name='Center (2D)',
        #                         marker=dict(size=3, color=color_informed),
        #                         legendgroup = legends[mode_idx],
        #                         showlegend = False,
        #                     ))
        #                     frame_traces.append(go.Scatter3d(
        #                         x=[q_rand[0]],
        #                         y=[q_rand[1]],
        #                         z = [1],
        #                         mode='markers',
        #                         name='Center (2D)',
        #                         marker=dict(size=3, color='red'),
        #                         legendgroup = legends[mode_idx],
        #                         showlegend = False,
        #                     ))
        # except Exception:
        #     if informed is not None:
        #         if config['informed_sampling'] != []:
        #             mode = informed_sampling['mode']
        #             mode_idx = modes.index(mode)
        #             n_rand = data['n_rand']
        #             if len(env.robots) > 1:
        #                 for robot_idx, robot in enumerate(env.robots):
        #                     r_indices = env.robot_idx[robot] 
        #                     q_rand = n_rand[r_indices]
        #                     frame_traces.append(go.Scatter3d(
        #                     x=[q_rand[0]],
        #                     y=[q_rand[1]],
        #                     z = [1],
        #                     mode='markers',
        #                     name='Center (2D)',
        #                     marker=dict(size=3, color='red'),
        #                     legendgroup = legends[mode_idx],
        #                     showlegend = False,
        #                 ))
        #             else:
        #                 if n_rand is not None: 
        #                     q_rand = n_rand
        #                     frame_traces.append(go.Scatter3d(
        #                     x=[q_rand[0]],
        #                     y=[q_rand[1]],
        #                     z = [1],
        #                     mode='markers',
        #                     name='Center (2D)',
        #                     marker=dict(size=3, color='red'),
        #                     legendgroup = legends[mode_idx],
        #                     showlegend = False,
        #                     ))

                




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

def final_path_animation(env, env_path, path, output_html):
    
    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  

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
    mode = env.start_mode
    while True:     
            modes.append(mode.task_ids)
            if env.is_terminal_mode(mode):
                break
            mode = env.get_next_mode(None, mode)
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
            path_x = [state[indices][0] for state in path]
            path_y = [state[indices][1] for state in path]
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

def ellipse(env, env_path, pkl_folder):

    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  


    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
   


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
    try:
        modes = []
        mode = env.start_mode
        while True:     
                modes.append(mode.task_ids)
                if env.is_terminal_mode(mode):
                    break
                mode = env.get_next_mode(None, mode)
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
        
        legends = []
        for idx in range(len(modes)):
            name = f"Mode: {modes[idx]}"
            legends.append(name)
            static_traces.append(
                go.Mesh3d(
                    x=[0],  # Position outside the visible grid
                    y=[0],  # Position outside the visible grid
                    z=[0],
                    name=name,
                    color = colors[idx],
                    legendgroup=name,  # Unique legend group for each mode
                    showlegend=True  # This will create a single legend entry for each mode
                )
            )
    except Exception:
        modes = []
        pass
        
    
    # dynamic_traces
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as file:
            frame_traces = []
            data = dill.load(file)
            try:
                q_rand_ellipse = data["ellipse"]
                mode = data["mode"]
                if mode is None:
                    mode_idx = 0
                else:
                    try:
                        mode_idx = modes.index(mode)
                    except Exception: 
                        mode_idx = None
                        
                for q in q_rand_ellipse:
                    for robot_idx, robot in enumerate(env.robots):
                        if mode_idx is None:
                            color = colors[robot_idx]
                        else:
                            color = colors[mode_idx]
                        legend_group = robot
                        r_indices = env.robot_idx[robot] 
                        q_rand = q[r_indices]
                        frame_traces.append(go.Scatter3d(
                        x=[q_rand[0]],
                        y=[q_rand[1]],
                        z = [1],
                        mode='markers',
                        marker=dict(size=3, color=color),
                        legendgroup = legend_group,
                        showlegend = False,
                    ))
            except Exception: 
                continue
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

    # fig.write_html(output_html)
    fig.show()




        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--do",
        choices=["dev", "cost_single", "cost_multiple", "shortcutting_cost", "path", "sum", "interpolation", "nn", "tree", "graph", "final_path_animation", "ellipse"],
        required=True,
        help="Select the mode of operation",
    )
   
    args = parser.parse_args()
    home_dir = os.path.expanduser("~")
    directory = os.path.join(home_dir, 'multirobot-pathplanning-benchmark/out')
    dir = get_latest_folder(directory)
    # dir = '/home/tirza/output/simple_2d_120125_171604'
    # dir = '/home/tirza/output/one_agent_many_goals_260125_231109'
    # pattern = r'^\d{6}_\d{6}$'
    # last_part = os.path.basename(dir)
    # single_folder = False
    # if re.match(pattern, last_part):
    #     path = dir
    #     single_folder = True
    # else:
    #     path = os.path.join(dir, '0')
    
    # env_name, config_params, _, _ = get_config(path)
    config_dir = os.path.join(dir, 'config.json')
    config = load_experiment_config(config_dir)
    env = get_env_by_name(config["environment"])    
    path =   os.path.join(home_dir, 'output')
    pkl_folder = get_latest_folder(path)
    env_path = os.path.join(home_dir, f'env/{config["environment"]}')  
    save_env_as_mesh(env, env_path) 

    if args.do == "dev":
        print("Development")
        with_tree = True
        if with_tree:
            output_html = os.path.join(path, 'tree_animation_3d.html')
            reducer = 100
        else:
            output_html = os.path.join(path, 'path_animation_3d.html')
            reducer = 100
        developement_animation(env, env_path, pkl_folder, output_html, with_tree, reducer)    
        webbrowser.open('file://' + os.path.realpath(output_html))
    if args.do == 'final_path_animation':
        save_env_as_mesh(env, env_path)
        pkl_folder = os.path.join(path, 'FramesFinalData')
        output_html = os.path.join(path, 'final_path_animation_3d.html')
        final_path_animation(env, env_path, pkl_folder, output_html)    
        webbrowser.open('file://' + os.path.realpath(output_html))
        
    if args.do == "cost_single":
        pkl_folder = os.path.join(path, 'FramesFinalData')
        output_filename_cost = os.path.join(path, 'Cost.png')
        cost_single(env, pkl_folder, output_filename_cost)

    if args.do == "cost_multiple":
        dir_planner = []
        for planner in config["planners"]:
            dir_planner.append(os.path.join(dir, planner['name']))
        with_inital_confidence_interval = False
        pkl_folder = os.path.join(path, 'FramesFinalData')
        output_filename_cost = os.path.join(os.path.dirname(dir), f'Cost_{datetime.now().strftime("%d%m%y_%H%M%S")}.png')
        cost_multiple(env, config, pkl_folder ,dir_planner ,output_filename_cost, with_inital_confidence_interval)
    
    if args.do == "sum":
        sum(dir, False)
    
    if args.do == "shortcutting_cost":
        fix_axis = False
        output_filename_cost = os.path.join(path, 'ShortcuttingCost.png')
        shortcutting_cost(env, path, output_filename_cost)
    
    if args.do == "path":
        pkl_folder = os.path.join(path, 'FramesFinalData')
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

    if args.do == "ellipse":
        ellipse(env, env_path, pkl_folder)    

#TO CHECK
    if args.do == "interpolation":
        interpolation_check(env)
    if args.do == "nn":
        with_tree = True
        if with_tree:
            output_html = os.path.join(path, 'tree_animation_3d.html')
            reducer = 50
        else:
            output_html = os.path.join(path, 'path_animation_3d.html')
            reducer = 400
        nearest_neighbor(env, env_path, pkl_folder, output_html, with_tree, reducer)    
    if args.do == "tree":
        tree(env, env_path, pkl_folder)  
    if args.do == "graph":
        output_html = os.path.join(path, 'graph.html')
        graph(env, env_path, pkl_folder, output_html)    
        webbrowser.open('file://' + os.path.realpath(output_html))
 


