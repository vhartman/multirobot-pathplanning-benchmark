import argparse
import plotly.graph_objects as go
import os
import numpy as np
import time as time
import dill
import random
import traceback
import re
from analysis_util import(
    colors_plotly,
    colors_ry,
    mesh_traces_env,
    save_env_as_mesh
    )

from multi_robot_multi_goal_planning.problems import get_env_by_name
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.disp_path_2d import get_infos_of_obstacles_and_table_2d
import matplotlib.lines as mlines
def process_all_frame_traces_to_figure(env, all_frame_traces, static_traces):
    # Determine the maximum number of dynamic traces needed
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except Exception:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  
    
    fig = go.Figure()
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

def ellipse_with_samples(env, env_path, pkl_file):
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
        time.sleep(10)
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
    with open(pkl_file, 'rb') as file:
        frame_traces = []
        data = dill.load(file)
        try:
            q_rand_ellipse = data["ellipse"]
            mode = data["mode"]
            focal= data["focal_points"]
            center = data["center"]
            path = data["path"]
            if mode is None:
                mode_idx = 0
            else:
                try:
                    mode_idx = modes.index(mode)
                except Exception: 
                    mode_idx = None   
            for robot_idx, robot in enumerate(env.robots):
                legend_group = robot
                f = focal[robot]
                c = center[robot]
                r_indices = env.robot_idx[robot]
                frame_traces.append(go.Scatter3d(
                    x=[f[0][r_indices][0]],
                    y=[f[0][r_indices][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='grey'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                frame_traces.append(go.Scatter3d(
                    x=[f[1][r_indices][0]],
                    y=[f[1][r_indices][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='black'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                if c is not None:
                    frame_traces.append(go.Scatter3d(
                        x=[c[0]],
                        y=[c[1]],
                        z = [1],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        legendgroup = legend_group,
                        showlegend = False,
                    ))
                    frame_traces.append(go.Scatter3d(
                        x=[c[0]],
                        y=[c[1]],
                        z = [1],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        legendgroup = legend_group,
                        showlegend = False,
                    ))

                for q in q_rand_ellipse[robot]:
                    if mode_idx is None:
                        color = colors[robot_idx]
                    else:
                        color = colors[mode_idx]
                    
                    frame_traces.append(go.Scatter3d(
                    x=[q[0]],
                    y=[q[1]],
                    z = [1],
                    mode='markers',
                    opacity=0.6,
                    marker=dict(size=5, color=color),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                path_x = [state[r_indices][0] for state in path]
                path_y = [state[r_indices][1] for state in path]
                path_z = [1 for _ in path]
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
                        legendgroup=robot,
                        showlegend=False
                    )
                    )
        
        
        
        except Exception: 
            print("Error occured")
            pass
    all_frame_traces.append(frame_traces)

    process_all_frame_traces_to_figure(env, all_frame_traces, static_traces)

def ellipse_with_matrices(env, env_path, pkl_file):
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
    with open(pkl_file, 'rb') as file:
        frame_traces = []
        data = dill.load(file)
        try:
            mode = data["mode"]
            focal= data["focal_points"]
            center = data["center"]
            path = data["path"]
            C_matrices = data["C"]
            L_matrices = data["L"]


            if mode is None:
                mode_idx = 0
            else:
                try:
                    mode_idx = modes.index(mode)
                except Exception: 
                    mode_idx = None   

            for robot_idx, robot in enumerate(env.robots):
                if mode_idx is None:
                    color = colors[robot_idx]
                else:
                    color = colors[mode_idx]
                legend_group = robot
                f = focal[robot]
                c = center[robot]
                r_indices = env.robot_idx[robot]
                frame_traces.append(go.Scatter3d(
                    x=[f[0][r_indices][0]],
                    y=[f[0][r_indices][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='grey'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                frame_traces.append(go.Scatter3d(
                    x=[f[1][r_indices][0]],
                    y=[f[1][r_indices][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='black'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                if c is not None:
                    frame_traces.append(go.Scatter3d(
                        x=[c[0]],
                        y=[c[1]],
                        z = [1],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        legendgroup = legend_group,
                        showlegend = False,
                    ))
                    frame_traces.append(go.Scatter3d(
                        x=[c[0]],
                        y=[c[1]],
                        z = [1],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        legendgroup = legend_group,
                        showlegend = False,
                    ))
                try:
                    C = C_matrices[robot_idx]
                    L = L_matrices[robot_idx]
                    if L is not None:
                        if C.shape[0] == 3:  # 3D case
                            theta = np.linspace(0, 2 * np.pi, 100)
                            phi = np.linspace(0, np.pi, 50)
                            theta, phi = np.meshgrid(theta, phi)

                            # Generate unit sphere points
                            x = np.sin(phi) * np.cos(theta)
                            y = np.sin(phi) * np.sin(theta)
                            z = np.cos(phi)
                            

                            # Transform the unit sphere into an ellipsoid
                            unit_sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
                            ellipsoid_transformed = C @ L @ unit_sphere
                            

                            # Translate to center
                            x_ellipsoid = ellipsoid_transformed[0, :] + c[0]
                            y_ellipsoid = ellipsoid_transformed[1, :] + c[1]
                            z_ellipsoid = ellipsoid_transformed[2, :] + c[2] + 1


                            # Add 3D ellipsoid using Mesh3d
                            frame_traces.append(
                                go.Mesh3d(
                                    x=x_ellipsoid,
                                    y=y_ellipsoid,
                                    z=z_ellipsoid,
                                    alphahull=0,  # Ensure it forms a hull
                                    color=color,  # Grey color for the surface
                                    opacity=0.1,  # High transparency
                                    legendgroup = robot,
                                    showlegend = False,
                                )
                            )
                        if C.shape[0] == 2:  # 2D case
                            theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles for ellipse
                            
                            # Generate unit circle points
                            x = np.cos(theta)
                            y = np.sin(theta)
                            unit_circle = np.array([x, y])
                            
                            # Transform the unit circle into an ellipse
                            ellipse_transformed = C @ L @ unit_circle
                            
                            # Translate to center
                            x_ellipse = ellipse_transformed[0, :] + c[0]
                            y_ellipse = ellipse_transformed[1, :] + c[1]
                            z_ellipse = np.ones_like(x_ellipse)  # Set z height to 1
                            
                            # Add 2D ellipse using Scatter3d to position it at z = 1
                            frame_traces.append(
                                go.Scatter3d(
                                    x=x_ellipse,
                                    y=y_ellipse,
                                    z=z_ellipse,
                                    mode='lines',
                                    line=dict(color=color, width=2),
                                    legendgroup=robot,
                                    showlegend=False,
                                )
                            )
                except Exception as e:
                    tb = traceback.format_exc()  # Get the full traceback
                    print(
                        f"Error:  {e}\nTraceback:\n{tb}"
                    )

                path_x = [state[r_indices][0] for state in path]
                path_y = [state[r_indices][1] for state in path]
                path_z = [1 for _ in path]
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
                        legendgroup=robot,
                        showlegend=False
                    )
                    )
        
        
        
        except Exception as e:
                    tb = traceback.format_exc()  # Get the full traceback
                    print(
                        f"Error:  {e}\nTraceback:\n{tb}"
                    )
    all_frame_traces.append(frame_traces)

    process_all_frame_traces_to_figure(env, all_frame_traces, static_traces)

def samples(env, env_path, pkl_file):
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
        time.sleep(10)
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
    with open(pkl_file, 'rb') as file:
        frame_traces = []
        data = dill.load(file)
        try:
            q_samples = data["q_samples"]
            modes_samples = data["modes"]
            try:
                path = data["path"]
            except Exception:
                path = None
            for robot_idx, robot in enumerate(env.robots):
                legend_group = robot
                r_indices = env.robot_idx[robot]
                for idx, q in enumerate(q_samples):
                    mode = modes_samples[idx]
                    mode_idx = modes.index(mode)
                    if mode_idx is None:
                        color = colors[robot_idx]
                    else:
                        color = colors[mode_idx]
                    
                    frame_traces.append(go.Scatter3d(
                    x=[q[r_indices][0]],
                    y=[q[r_indices][1]],
                    z = [1],
                    mode='markers',
                    opacity=0.6,
                    marker=dict(size=5, color=color),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                if path is not None:
                    path_x = [state.q.state()[r_indices][0] for state in path]
                    path_y = [state.q.state()[r_indices][1] for state in path]
                    path_z = [1 for _ in path]
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
                            legendgroup=robot,
                            showlegend=False
                        )
                        )
        
        
        
        except Exception: 
            print("Error occured")
            pass
    all_frame_traces.append(frame_traces)

    process_all_frame_traces_to_figure(env, all_frame_traces, static_traces)

def visualize_tree_2d_with_color(env, path_to_folder,env_path, html:bool = False):
    dir = os.path.join(os.path.dirname(path_to_folder),'tree_vis')
    os.makedirs(dir, exist_ok=True)
    obstacles, table_size = get_infos_of_obstacles_and_table_2d(env)
    colors = colors_ry()
    reached_modes = []
    mode = env.start_mode
    time.sleep(10)
    color = {}
    counter = 0
   
    # goal = 
    #get all mods of this environment
    while True:     
        reached_modes.append(mode.task_ids)
        color[f'{mode.task_ids}'] = list(range(counter, counter + len(env.robots)))
        counter += len(env.robots)
        if env.is_terminal_mode(mode):
            break
        mode = env.get_next_mode(None, mode)

    q_start = env.get_start_pos().state()
    q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_modes[-1])

    for filename in os.listdir(path_to_folder):
        pkl_file = os.path.join(path_to_folder, filename)
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
            nodes = data['nodes']
            parents = data['parents']
            modes = data['modes']
            type = data['type']
            all_nodes = data['all_nodes']
            all_transition_nodes = data['all_transition_nodes']
            all = all_nodes + all_transition_nodes

        fig, ax = plt.subplots()
        ax.set_xlim(1*table_size[0], -1*table_size[0])
        ax.set_ylim(1*table_size[1], -1.0*table_size[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        
        
        
        # Draw obstacles
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]
            rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, color="black")
            ax.add_patch(rect)

        legend_handles = []

        for robot_idx, robot in enumerate(env.robots):
            for task_ids in reached_modes:
                color_idx = color[f'{task_ids}'][robot_idx]
                name = f"R{robot_idx}, Mode: {task_ids}"
                
                handle = mlines.Line2D(
                    [], [], 
                    marker='o', 
                    linestyle='None', 
                    markersize=5, 
                    markerfacecolor=colors[color_idx], 
                    markeredgecolor='none', 
                    label=name
                )
                legend_handles.append(handle)


        # Add legend with title and styling
        ax.legend(
            handles=legend_handles,
            title=f"{type} tree",
            title_fontsize=3,
            fontsize=2,
            markerscale=0.3,
            handlelength=1,
            loc='upper right',
            frameon=False,
            borderpad=0.1
        )


        #tree
        lines_by_color = {}
        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            for node, parent, mode in zip(nodes, parents, modes):                
                color_idx = color[f'{mode}'][robot_idx]
                x0 = node[indices][0]
                y0 = node[indices][1]

                if parent is not None:
                    x1 = parent[indices][0]
                    y1 = parent[indices][1]
                else:
                    x1 = x0
                    y1 = y0

                if color_idx not in lines_by_color:
                    lines_by_color[color_idx] = {'x': [], 'y': []}
                lines_by_color[color_idx]['x'].extend([x0, x1, None])
                lines_by_color[color_idx]['y'].extend([y0, y1, None])


        for color_idx, line_data in lines_by_color.items():
            ax.plot(
                line_data['x'],
                line_data['y'],
                color=colors[color_idx],
                linewidth=0.5,
                alpha=0.5
            )
            ax.scatter(
                line_data['x'],
                line_data['y'],
                color=colors[color_idx],
                alpha=0.5,
                s=4,  
                edgecolors='none'
            )


        all_dict = {'x': [], 'y':[]}
        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            for n in all:
                all_dict['x'].append(n[indices][0])
                all_dict['y'].append(n[indices][1])
            
        ax.scatter(
            all_dict['x'],
            all_dict['y'],
            color='grey',
            alpha=0.5,
            s=2,  
            edgecolors='none'
        )

        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            x_start = q_start[indices][0]
            y_start = q_start[indices][1]
            x_goal = q_goal[indices][0]
            y_goal = q_goal[indices][1]

            # Plot a circle marker at the start point
            ax.scatter(
                [x_goal],
                [y_goal],
                color='white',
                s=30,  # size of the circle
                edgecolors='black',
                zorder=5
            )

            # Plot a circle marker at the start point
            ax.scatter(
                [x_start],
                [y_start],
                color='black',
                s=13,  # size of the circle
                edgecolors='none',
                zorder=6
            )

    
        next_file_number = max(
                (int(file.split('.')[0]) for file in os.listdir(dir)
                if file.endswith('.png') and file.split('.')[0].isdigit()),
                default=-1
            ) + 1
        plt.savefig(os.path.join(dir,f"{next_file_number:04d}.png"), dpi=300, bbox_inches='tight')
        # plt.show()    

def visualize_tree_2d_paper(env, path_to_folder):
    #TODO need to add forward_tree
    colors =["magenta", "grey"]
    dir = os.path.join(os.path.dirname(path_to_folder),'tree_vis')
    os.makedirs(dir, exist_ok=True)
    obstacles, table_size = get_infos_of_obstacles_and_table_2d(env)
    reached_modes = []
    mode = env.start_mode
    time.sleep(10)
    counter = 0
    while True:     
        reached_modes.append(mode.task_ids)
        counter += len(env.robots)
        if env.is_terminal_mode(mode):
            break
        mode = env.get_next_mode(None, mode)
    
    q_start = env.get_start_pos().state()
    q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_modes[-1])

    for filename in os.listdir(path_to_folder):
        pkl_file = os.path.join(path_to_folder, filename)
        with open(pkl_file, 'rb') as file:
            all_data = dill.load(file)
            all_nodes = all_data['all_nodes']
            all_transition_nodes = all_data['all_transition_nodes']  
            all = all_nodes + all_transition_nodes
        
            

        fig, ax = plt.subplots()
        ax.set_xlim(1*table_size[0], -1*table_size[0])
        ax.set_ylim(1*table_size[1], -1.0*table_size[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        
        # Draw obstacles
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]
            rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, color="black")
            ax.add_patch(rect)
        for idx, type in enumerate(['forward', 'reverse']):
            if idx == 0:
                zorder = 1
            if idx == 1:
                zorder = 0
            data = all_data[type]
            nodes = data['nodes']
            parents = data['parents']
            modes = data['modes']
            color = colors[idx]
            #tree
            lines = {'x': [], 'y': []}
            for robot_idx, robot in enumerate(env.robots):
                indices = env.robot_idx[robot]
                for node, parent, mode in zip(nodes, parents, modes):                
                    x0 = node[indices][0]
                    y0 = node[indices][1]

                    if parent is not None:
                        x1 = parent[indices][0]
                        y1 = parent[indices][1]
                    else:
                        x1 = x0
                        y1 = y0

                    lines['x'].extend([x0, x1, None])
                    lines['y'].extend([y0, y1, None])

            ax.plot(
                lines['x'],
                lines['y'],
                color=color,
                linewidth=0.5,
                alpha=1,
                zorder=zorder
            )
            ax.scatter(
                lines['x'],
                lines['y'],
                color=color,
                alpha=1,
                s=2,  
                edgecolors='none',
                zorder=zorder
            )


        all_dict = {'x': [], 'y':[]}
        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            for n in all:
                all_dict['x'].append(n[indices][0])
                all_dict['y'].append(n[indices][1])
            
        ax.scatter(
            all_dict['x'],
            all_dict['y'],
            color='grey',
            alpha=1,
            s=2,  
            edgecolors='none',
            zorder=0
        )

        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            x_start = q_start[indices][0]
            y_start = q_start[indices][1]
            x_goal = q_goal[indices][0]
            y_goal = q_goal[indices][1]

            # Plot a circle marker at the start point
            ax.scatter(
                [x_goal],
                [y_goal],
                color='white',
                s=30,  # size of the circle
                edgecolors='black',
                zorder=5
            )

            # Plot a circle marker at the start point
            ax.scatter(
                [x_start],
                [y_start],
                color='black',
                s=13,  # size of the circle
                edgecolors='none',
                zorder=6
            )

        next_file_number = max(
                (int(file.split('.')[0]) for file in os.listdir(dir)
                if file.endswith('.png') and file.split('.')[0].isdigit()),
                default=-1
            ) + 1
        plt.savefig(os.path.join(dir,f"{next_file_number:04d}.png"), dpi=300, bbox_inches='tight') 

def visualize_tree_2d_with_color_html(env, env_path, pkl_folder, output_html):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    reached_modes = []
    mode = env.start_mode
    time.sleep(10)
    color = {}
    counter = 0
   
    # goal = 
    #get all mods of this environment
    while True:     
        reached_modes.append(mode.task_ids)
        color[f'{mode.task_ids}'] = list(range(counter, counter + len(env.robots)))
        counter += len(env.robots)
        if env.is_terminal_mode(mode):
            break
        mode = env.get_next_mode(None, mode)

    q_start = env.get_start_pos().state()
    q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_modes[-1])

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

    for robot_idx, robot in enumerate(env.robots):
        for type in ['forward', 'reverse', '']:
            if type == '':
                legend_group = robot
            else:
                legend_group = type + ' ' + robot
            static_traces.append(
            go.Mesh3d(
                x=[0],  # X-coordinates of the exterior points
                y=[0],  # Y-coordinates of the exterior points
                z=[0] ,  # Flat surface at z = 0
                color=colors[len(reached_modes)+robot_idx],  # Fill color from the agent's properties
                opacity=1,  # Transparency level
                name=legend_group,
                legendgroup=legend_group,
                showlegend=True
            )
        )
    # if with_tree:
    legends = []
    for idx in range(len(reached_modes)):
        name = f"Mode: {reached_modes[idx]}"
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
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as file:
            all_data = dill.load(file)
            all_nodes = all_data['all_nodes']
            all_transition_nodes = all_data['all_transition_nodes']
        frame_traces = []    
        for type in ['forward', 'reverse']:
            data = all_data[type]
            nodes = data['nodes']
            parents = data['parents']
            modes = data['modes']
            # if type == 'reverse':
            #     continue
            
            #forward and reverse tree
            for robot_idx, robot in enumerate(env.robots):
                lines_by_color = {}
                indices = env.robot_idx[robot]
                for node, parent, m in zip(nodes, parents, modes):    
                    mode_idx = reached_modes.index(m)            
                    color_idx = colors[mode_idx]
                    if type == 'forward':
                        color_idx = 'grey'
                    x0 = node[indices][0]
                    y0 = node[indices][1]

                    if parent is not None:
                        x1 = parent[indices][0]
                        y1 = parent[indices][1]
                    else:
                        x1 = x0
                        y1 = y0

                    if color_idx not in lines_by_color:
                        lines_by_color[color_idx] = {'x': [], 'y': [], 'legend_group': legends[mode_idx]}
                    lines_by_color[color_idx]['x'].extend([x0, x1, None])
                    lines_by_color[color_idx]['y'].extend([y0, y1, None])

                for color, line_data in lines_by_color.items():
                    legend_group = line_data['legend_group']
                    frame_traces.append(
                        go.Scatter3d(
                            x=line_data['x'],
                            y=line_data['y'],
                            z = [1] * len(line_data['x']),
                            mode='markers+lines',
                            marker=dict(size=5, color=color),
                            line=dict(color=color, width=6),
                            opacity=0.5,
                            name=type + ' ' + robot,
                            legendgroup=type + ' ' + robot,
                            showlegend=False
                        )
                    )

        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            all_dict = {'x': [], 'y':[]}
            for n in all_nodes:
                all_dict['x'].append(n[indices][0])
                all_dict['y'].append(n[indices][1])
            
            frame_traces.append(
                go.Scatter3d(
                    x=all_dict['x'],
                    y=all_dict['y'],
                    z = [1] * len(all_dict['x']),
                    mode='markers',
                    marker=dict(size=2, color='grey'),
                    opacity=1,
                    name=robot,
                    legendgroup=robot,
                    showlegend=False
                )
            )

            all_transition_dict = {'x': [], 'y':[]}
            for n in all_transition_nodes:
                all_transition_dict['x'].append(n[indices][0])
                all_transition_dict['y'].append(n[indices][1])
            
            frame_traces.append(
                go.Scatter3d(
                    x=all_transition_dict['x'],
                    y=all_transition_dict['y'],
                    z = [1] * len(all_transition_dict['x']),
                    mode='markers',
                    marker=dict(size=2, color="red"),
                    opacity=1,
                    name=robot,
                    legendgroup=robot,
                    showlegend=False
                )
            )
            
        for robot_idx, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            x_start = q_start[indices][0]
            y_start = q_start[indices][1]
            x_goal = q_goal[indices][0]
            y_goal = q_goal[indices][1]

            # Goal point: white fill, black edge (simulate with white marker and black line)
            frame_traces.append(
                go.Scatter3d(
                    x=[x_goal],
                    y=[y_goal],
                    z=[1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='white',
                        line=dict(color='black', width=2)
                    ),
                    opacity=1,
                    name=robot,
                    legendgroup=robot,
                    showlegend=False
                )
            )

            # Start point: black fill, no edge, smaller size
            frame_traces.append(
                go.Scatter3d(
                    x=[x_start],
                    y=[y_start],
                    z=[1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='black',
                        line=dict(color='black', width=0)
                    ),
                    opacity=1,
                    name=robot,
                    legendgroup=robot,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--path_to_pkl_file",
        required=False,
        help="Select the path of the folder to be processed",
    )
    parser.add_argument(
        "--path_to_folder",
        required=False,
        help="Select the path of the folder to be processed",
    )
    parser.add_argument("do", nargs="?", choices=["ellipse_with_samples", "ellipse_with_matrices", "samples", "tree", "tree_html", "tree_paper"], default="", help="action to do")
    parser.add_argument("env_name", nargs="?", default="", help="name of environment")
    parser.add_argument("seed", nargs="?", default="", help="seed")


    args = parser.parse_args()
    if args.path_to_pkl_file is not None:
        dir_path_pkl = args.path
    else:
        #get latest created folder
        home_dir = os.path.expanduser("~")
        dir = os.path.join(home_dir, 'multirobot-pathplanning-benchmark/out')
        if args.do != 'tree' and  args.do != 'tree_paper' and args.do != 'tree_html':
            dir_out = os.path.join(dir, 'Analysis')
            os.makedirs(dir_out, exist_ok=True)
            files = [f for f in os.listdir(dir_out) if os.path.isfile(os.path.join(dir_out, f))]
            pkl_file = sorted(files)[-1]
            dir_path_pkl = os.path.join(dir_out, pkl_file)
        else:
            dir = os.path.join(dir, 'Analysis/Tree')
            assert hasattr(args, 'path_to_folder'), (
            "Path to folder needs to be defined"
            )
            dir_out = os.path.join(args.path_to_folder, 'tree')
            os.makedirs(dir_out, exist_ok=True)

            # Get current max index in the target folder
            existing_files = [f for f in os.listdir(dir_out) if f.endswith('.pkl')]
            existing_indices = [
                int(re.findall(r'(\d+)\.pkl', f)[0]) for f in existing_files if re.match(r'\d+\.pkl', f)
            ]
            start_idx = max(existing_indices, default=-1) + 1  # Start after the last existing

            # Copy files one by one with updated names
            if os.path.exists(dir):
                new_files = sorted(f for f in os.listdir(dir) if f.endswith('.pkl'))
                for i, fname in enumerate(new_files):
                    src = os.path.join(dir, fname)
                    new_fname = f"{start_idx + i:04d}.pkl"
                    dst = os.path.join(dir_out, new_fname)
                    shutil.copy2(src, dst)

                # Clean up the source folder if needed
                shutil.rmtree(dir)
            
    seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    env = get_env_by_name(args.env_name)  
    home_dir = os.path.expanduser("~")
    env_path = os.path.join(home_dir, f'env/{args.env_name}') 
    if args.do != 'tree_paper' and args.do != 'tree': 
        save_env_as_mesh(env, env_path) 
    
    if args.do == "ellipse_with_samples":
        ellipse_with_samples(env, env_path, dir_path_pkl) 
    if args.do == "ellipse_with_matrices":
        ellipse_with_matrices(env, env_path, dir_path_pkl) 
       
    if args.do == "samples":
        samples(env, env_path, dir_path_pkl) 
    
    if args.do == "tree":
        visualize_tree_2d_with_color(env, dir_out, env_path, html= True) 
    
    if args.do == "tree_html":
        output_html = os.path.join(os.path.dirname(dir_out), 'Tree.html')
        visualize_tree_2d_with_color_html(env, env_path, dir_out, output_html)

    if args.do == "tree_paper":
        visualize_tree_2d_paper(env, dir_out)

