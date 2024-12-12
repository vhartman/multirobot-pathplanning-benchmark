import argparse
import sys
import dill
import plotly.graph_objects as go
import os
import webbrowser
from analysis_util import *
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # One level up
sys.path.append(parent_folder)
from rai_envs import *
from planning_env import *
from util import *
from rai_config import *
from planning_env import *
from util import *


def get_cost_label(cost_function, number_agents):
    if cost_function == 1:
        if number_agents == 0:    
            cost_label = f"$\\mathsf{{Cost}}: c = \\min \\sum\\limits_{{\\mathsf{{i}} \\in [0]}} \\Delta x_{{\\mathsf{{r}}^\\mathsf{{i}}}}$"
        else:
            cost_label = f"$\\mathsf{{Cost}}: c = \\min \\sum\\limits_{{\\mathsf{{i}} \\in [0 \\dots {number_agents}]}} \\Delta x_{{\\mathsf{{r}}^\\mathsf{{i}}}}$"

    if cost_function == 2:
        if number_agents == 0:             
            cost_label = f"$c = \\min \\sum\\limits_{{\\substack{{n \\in [Path]}}}} \\max\\limits_{{\\substack{{i \\in [0]}}}} \\Delta x_{{r^i}}$"
        else:              
            cost_label = f"$c = \\min \\sum\\limits_{{\\substack{{n \\in [Path]}}}} \\max\\limits_{{\\substack{{i \\in [0 \\dots {number_agents}]}}}} \\Delta x_{{r^i}}$"
               
    return cost_label

def get_cost_plot(env, config, pkl_folder, output_filename, fix_axis):
    # Get the list of .pkl files
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    print(f"Found {len(pkl_files)} .pkl files.")
    num_agents = len(env.robots)
    colors = colors_plotly()
    cost_function= config["cost_function"]
    cost_label = get_cost_label(cost_function, num_agents-1)
    agent_costs = {agent: [] for agent in range(num_agents)}
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
            print(time[-1])
            init_path_time = time[-1]
        if init_sol:
            
            cost = result["total"]
            agent_cost = result["agent_cost"]
            if cost is None:
                costs.append(float('nan'))
            else:
                costs.append(cost.item())

            for agent in range(num_agents):
                cost = agent_cost[0][agent]
                if cost is None:
                    cost = np.array(float('nan'))
                agent_costs[agent].append(cost.cpu().numpy())
        else:
            costs.append(float('nan'))
            for agent in range(num_agents):
                agent_costs[agent].append(float('nan'))

    
    fig = go.Figure()
    
    for agent in range(num_agents):
        fig.add_trace(go.Scatter(
            x=time,
            y=agent_costs[agent],
            mode='lines',
            name=f"$\\mathsf{{Agent \;{agent}}}$", 
            line=dict(color=colors[agent]),
            showlegend=True  # Show legend for each agent color
        ))

    # Plot the cumulative total cost data
    fig.add_trace(go.Scatter(
        x=time,
        y=costs,
        mode='lines',
        name=cost_label,
        line=dict(color='black'),
        showlegend=True  # Ensure "Total Cost" is in the legend
    ))
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
    max_cost = np.nanmax(costs)
    max_time = time[-1]
    print(np.nanmin(costs), max_time)
    # Update layout with titles, axis labels, and legend
    if fix_axis:
        max_time = 1200
        max_cost = 17
   
    fig.update_layout(
        title="Cost vs Time",
        xaxis=dict(title="Time [s]", range=[0, max_time+5], autorange=False ),
        yaxis=dict(title="Cost [m]", range=[0, max_cost+5], autorange=False ), 
         margin=dict(l=0, r=50, t=50, b=50),  # Increase right margin for legend space
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1  # Position legend outside the plot area on the right
        )
    )
    
    fig.write_image(output_filename)

def average(sum_times, runs, max_len, times, costs):
    stop_time = sum_times/runs
    average_time = np.linspace(0, stop_time, max_len, endpoint = True)
    average_costs = []
    for idx, cost in enumerate(costs):
        cost_average = []
        for time in average_time:
            if time == 0:
                cost_average.append(cost[0])
                continue
            if times[idx][-1] < time:
                cost_average.append(cost[-1])
            for c_idx, c in enumerate(cost):
                if times[idx][c_idx] > time:
                    cost_average.append(cost[c_idx-1])
                    break
                    
        average_costs.append(cost_average)
    
    return np.mean(average_costs, axis=0).tolist(), average_time

def get_cost_shortcutting_plot(env, config, path, output_filename):
    # Initialize lists and Plotly objects
    costs = []
    times = []
    colors_list = []
    labels_list = []
    labels_bool = []
    fig = go.Figure()
    num_agents = len(env.robots)
    colors = colors_plotly()  # Assuming this function is defined elsewhere
    cost_function = config["cost_function"]
    cost_label = get_cost_label(cost_function, num_agents - 1)  # Assuming this function is defined elsewhere
    labels = ["Shortcutting per mode", "Partial single dim per mode", "Partial random subset per mode", 
              "Shortcutting per task (1 robot) Prob", "Partial single dim per task (1 robot) Prob", "Partial random subset per task (1 robot) Prob", 
              "Shortcutting per task (all possible robot) Prob", "Partial single dim per task (all possible robot) Prob", "Partial random subset per task (all possible robot) Prob",
              "Shortcutting per task (1 robot) Deter", "Partial single dim per task (1 robot) Deter", "Partial random subset per task (1 robot) Deter", 
              "Shortcutting per task (all possible robot) Deter", "Partial single dim per task (all possible robot) Deter", "Partial random subset per task (all possible robot) Deter" ]
    

    # Loop through versions to process data
    for version in range(14):
        bool_legend = True
        folder = os.path.join(path, f"Post_{version}")
        # Get sorted list of .pkl files in the folder
        try:
            pkl_files = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
            print(f"Found {len(pkl_files)} .pkl files in {folder}.")
        except:
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



        average_cost, average_time = average(sum_times, len(pkl_files), max_len, version_times, version_costs)
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
                showlegend=True  # Ensure the l1egend entry is shown
            ))
    # Update layout with titles, axis labels, and legend
    fig.update_layout(
        title="Cost vs Time",
        xaxis=dict(title="Log(time) [s]",range=[np.log10(min_time), np.log10(max_time)], autorange=False, tickmode="auto",),
        yaxis=dict(title="Cost [m]", range=[min_cost -0.01, max_cost + 0.01], autorange=False),
        margin=dict(l=0, r=50, t=50, b=50),  # Increase right margin for legend space
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1  # Position legend outside the plot area on the right
        ),
        
    )
    
    # Save the plot as an image
    fig.write_image(output_filename)

def developement_animation(config, env, env_path, pkl_folder, output_html, with_tree, divider = None):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  


    count = count_files_in_folder(pkl_folder)
    if count > 500 and divider is not None:
        frame = int(count / divider)
    else:
        frame = 1
    print(f'Take every {frame}th frame')
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )[::frame]
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
    color_informed = 'black'
    with open(pkl_files[-1], 'rb') as file:
        data_informed = dill.load(file)
        informed_sampling = data_informed['informed_sampling']
        if informed_sampling != []:
            static_traces.append(
            go.Mesh3d(
                x=[0],  # Position outside the visible grid
                y=[0],  # Position outside the visible grid
                z=[0],
                color = color_informed,
                name='Informed Smapling - Ellipse/Ellipsoid',
                legendgroup='Informed Smapling - Ellipse/Ellipsoid',
                showlegend=True  # This will create a single legend entry for each mode
            )
        )
    modes = []
    mode = env.start_mode
    while True:     
            modes.append(mode)
            if mode == env.terminal_mode:
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
    if with_tree:
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

        frame_traces = []

        # Collect traces for the path (if any)
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
                            size=5,  # Very small markers
                            color="red",  # Match marker color with line
                            opacity=1
                        ),
                        opacity=1,
                        name="Transitions",
                        legendgroup="Transitions",
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
                    mode = node["mode"]
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
                            line=dict(color=color, width=1.3),
                            opacity=0.2,
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=False
                        )
                    )

        #informed sampling 
        if config['informed_sampling'] != []:
            informed = data["informed_sampling"]
            for informed_sampling in informed:
                for robot_idx, robot in enumerate(env.robots):
                    if informed_sampling['L']:
                        if robot_idx in informed_sampling['L'].keys():
                            C = informed_sampling['C'][robot_idx].cpu().numpy()
                            L = informed_sampling['L'][robot_idx].cpu().numpy()
                            center = informed_sampling['center'][robot_idx].cpu().numpy()
                            focal_points = [informed_sampling['start'][robot_idx], informed_sampling['goal'][robot_idx]]

                            if C.shape[0] == 3:  # 3D case
                                theta = np.linspace(0, 2 * np.pi, 100)
                                phi = np.linspace(0, np.pi, 50)
                                theta, phi = np.meshgrid(theta, phi)

                                # Generate unit sphere points
                                x = np.sin(phi) * np.cos(theta)
                                y = np.sin(phi) * np.sin(theta)
                                z = np.cos(phi)
                                unit_sphere = np.array([x.flatten(), y.flatten(), z.flatten()])

                                # Transform the unit sphere into an ellipsoid
                                ellipsoid_transformed = C @ L @ unit_sphere

                                # Translate to center
                                x_ellipsoid = ellipsoid_transformed[0, :] + center[0]
                                y_ellipsoid = ellipsoid_transformed[1, :] + center[1]
                                z_ellipsoid = ellipsoid_transformed[2, :] + center[2] + 1

                                # Add 3D ellipsoid using Mesh3d
                                frame_traces.append(
                                    go.Mesh3d(
                                        x=x_ellipsoid,
                                        y=y_ellipsoid,
                                        z=z_ellipsoid,
                                        alphahull=0,  # Ensure it forms a hull
                                        color=color_informed,  # Grey color for the surface
                                        opacity=0.1,  # High transparency
                                        name='Ellipsoid (3D)',
                                        legendgroup = 'Informed Smapling - Ellipse/Ellipsoid',
                                        showlegend = False,
                                    )
                                )

                                # Add focal points (3D)
                                for i, f in enumerate(focal_points):
                                    frame_traces.append(go.Scatter3d(
                                        x=[f[0]],
                                        y=[f[1]],
                                        z = [1],
                                        mode='markers',
                                        name=f'Focal Point {i+1} (2D)',
                                        marker=dict(size=8, color=color_informed),
                                        legendgroup = 'Informed Smapling - Ellipse/Ellipsoid',
                                        showlegend = False,
                                    ))

                                # Add center (2D)
                                frame_traces.append(go.Scatter3d(
                                    x=[center[0]],
                                    y=[center[1]],
                                    z = [1],
                                    mode='markers',
                                    name='Center (2D)',
                                    marker=dict(size=3, color=color_informed),
                                    legendgroup = 'Informed Smapling - Ellipse/Ellipsoid',
                                    showlegend = False,
                                ))



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--show",
        choices=["development", "cost", "shortcutting_cost"],
        required=True,
        help="Select the mode of operation",
    )
    
    args = parser.parse_args()
    home_dir = os.path.expanduser("~")
    directory = os.path.join(home_dir, 'output')
    path = get_latest_folder(directory)
    # path = "/home/tirza/output/091224_083733"
    # path = '/home/tirza/output/091224_083733'
    
    env_name, config_params, _, _ = get_config(path)
    env = get_env_by_name(env_name)    
    pkl_folder = os.path.join(path, 'FramesData')
    env_path = os.path.join(home_dir, f'env/{env_name}')
    save_env_as_mesh(env, env_path)
    print(path)
    if args.show == "development":
        print("Development")
        with_tree = True
        if with_tree:
            output_html = os.path.join(path, 'tree_animation_3d.html')
            reducer = 100
        else:
            output_html = os.path.join(path, 'path_animation_3d.html')
            reducer = 400
        developement_animation(config_params, env, env_path, pkl_folder, output_html, with_tree, reducer)    
        webbrowser.open('file://' + os.path.realpath(output_html))
    if args.show == "cost":
        fix_axis = False
        output_filename_cost = os.path.join(path, 'Cost.png')
        get_cost_plot(env, config_params, pkl_folder, output_filename_cost, fix_axis)
    if args.show == "shortcutting_cost":
        fix_axis = False
        output_filename_cost = os.path.join(path, 'ShortcuttingCost.png')
        get_cost_shortcutting_plot(env, config_params, path, output_filename_cost)


