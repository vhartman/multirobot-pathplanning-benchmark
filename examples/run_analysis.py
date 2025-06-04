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
import subprocess
from matplotlib.collections import LineCollection


def generate_gif(dir, framerate = 2):
    palette_file = os.path.join(dir, "palette.png")
    output_gif = os.path.join(dir, "output.gif")

    # Command 1: Generate palette
    palette_command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", os.path.join(dir, "%04d.png"),
        "-vf", "scale=iw:-1:flags=lanczos,palettegen",
        palette_file
    ]

    # Command 2: Create gif using palette
    gif_command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", os.path.join(dir, "%04d.png"),
        "-i", palette_file,
        "-lavfi", "scale=iw:-1:flags=lanczos [scaled]; [scaled][1:v] paletteuse=dither=bayer:bayer_scale=5",
        output_gif
    ]

    print(f"Generating GIF for: {dir}")

    # Run commands
    subprocess.run(palette_command, check=True)
    subprocess.run(gif_command, check=True)

    print(f"GIF saved to: {output_gif}")

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

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, showticklabels=False, zeroline=False),
        )
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
            CL_matrices = data["CL"]


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
                    x=[f[0][0]],
                    y=[f[0][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='black'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                frame_traces.append(go.Scatter3d(
                    x=[f[1][0]],
                    y=[f[1][1]],
                    z = [1],
                    mode='markers',
                    marker=dict(size=10, color='black'),
                    legendgroup = legend_group,
                    showlegend = False,
                ))
                # if c is not None:
                #     frame_traces.append(go.Scatter3d(
                #         x=[c[0]],
                #         y=[c[1]],
                #         z = [1],
                #         mode='markers',
                #         marker=dict(size=10, color='red'),
                #         legendgroup = legend_group,
                #         showlegend = False,
                #     ))
                #     frame_traces.append(go.Scatter3d(
                #         x=[c[0]],
                #         y=[c[1]],
                #         z = [1],
                #         mode='markers',
                #         marker=dict(size=10, color='red'),
                #         legendgroup = legend_group,
                #         showlegend = False,
                #     ))
                try:
                    try:
                        C = C_matrices[robot_idx]
                        L = L_matrices[robot_idx]
                        CL = C @ L
                    except Exception:
                        CL = CL_matrices[robot_idx]
                    if CL is not None:
                        if CL.shape[0] == 3:  # 3D case
                            theta = np.linspace(0, 2 * np.pi, 100)
                            phi = np.linspace(0, np.pi, 50)
                            theta, phi = np.meshgrid(theta, phi)

                            # Generate unit sphere points
                            x = np.sin(phi) * np.cos(theta)
                            y = np.sin(phi) * np.sin(theta)
                            z = np.cos(phi)
                            

                            # Transform the unit sphere into an ellipsoid
                            unit_sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
                            ellipsoid_transformed = CL @ unit_sphere
                            

                            # Translate to center
                            x_ellipsoid = ellipsoid_transformed[0, :] + c[0]
                            y_ellipsoid = ellipsoid_transformed[1, :] + c[1]
                            z_ellipsoid = ellipsoid_transformed[2, :] + 0 + 1


                            # Add 3D ellipsoid using Mesh3d
                            frame_traces.append(
                                go.Mesh3d(
                                    x=x_ellipsoid,
                                    y=y_ellipsoid,
                                    z=z_ellipsoid,
                                    alphahull=0,  # Ensure it forms a hull
                                    color=color,  # Grey color for the surface
                                    opacity=0.15,  # High transparency
                                    legendgroup = robot,
                                    showlegend = False,
                                )
                            )
                        if CL.shape[0] == 2:  # 2D case
                            theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles for ellipse
                            
                            # Generate unit circle points
                            x = np.cos(theta)
                            y = np.sin(theta)
                            unit_circle = np.array([x, y])
                            
                            # Transform the unit circle into an ellipse
                            ellipse_transformed = CL @ unit_circle
                            
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
                        line=dict(color=colors[len(modes)+robot_idx], width=8),
                        marker=dict(
                            size=4,  # Very small markers
                            color=colors[len(modes)+robot_idx],  # Match marker color with line
                            opacity=1
                        ),
                        opacity=1,
                        legendgroup=robot,
                        showlegend=False
                    )
                    )
                q_start = env.get_start_pos().state()
                try:
                    q_goal = env.tasks[env.sequence[-1]].goal.sample([2,2])
                except:
                    q_goal = env.tasks[-1].goal.sample([2,2])

                start = q_start[r_indices]
                end = q_goal[r_indices]
                frame_traces.append(
                    go.Scatter3d(
                        x=[start[0]], 
                        y=[start[1]],
                        z=[1],
                        mode="markers",
                        marker=dict(
                            size=10,  # Very small markers
                            color=color,  # Match marker color with line
                            opacity=1
                        ),
                        opacity=1,
                        legendgroup=robot,
                        showlegend=False
                    )
                    )
                frame_traces.append(
                    go.Scatter3d(
                        x=[end[0]], 
                        y=[end[1]],
                        z=[1],
                        mode="markers",
                        marker=dict(
                            size=10,  # Very small markers
                            color=color,  # Match marker color with line
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
                i = 0
                for idx, q in enumerate(q_samples):
                    if i == 45 or i == 45:
                        s = 8
                    else:
                        s = 5
                    i +=1
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
                    marker=dict(size=s, color=color),
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
                                size=5,  # Very small markers
                                color=colors[len(modes)+robot_idx],  # Match marker color with line
                                opacity=0.8
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
    reached_task_ids = []
    mode = env.start_mode
    time.sleep(10)
    color = {}
    counter = 0
   
    # goal = 
    #get all mods of this environment
    while True:     
        reached_task_ids.append(mode.task_ids)
        # color[f'{mode.task_ids}'] = list(range(counter, counter + len(env.robots)))
        color[f'{mode.task_ids}'] = list(range(counter))
        # counter += len(env.robots)
        counter +=1
        if env.is_terminal_mode(mode):
            break
        mode = env.get_next_mode(None, mode)

    q_start = env.get_start_pos().state()
    q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_task_ids[-1])

    for filename in os.listdir(path_to_folder):
        pkl_file = os.path.join(path_to_folder, filename)
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
            nodes = data['nodes']
            parents = data['parents']
            modes = data['modes']
            tree_type = data['type']
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
            for task_ids in reached_task_ids:
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
            title=f"{tree_type} tree",
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
                zorder=7
            )

            # Plot a circle marker at the start point
            ax.scatter(
                [x_start],
                [y_start],
                color='black',
                s=13,  # size of the circle
                edgecolors='none',
                zorder=7
            )

        path = data["path"]


    
        next_file_number = max(
                (int(file.split('.')[0]) for file in os.listdir(dir)
                if file.endswith('.png') and file.split('.')[0].isdigit()),
                default=-1
            ) + 1
        plt.savefig(os.path.join(dir,f"{next_file_number:04d}.png"), dpi=300, bbox_inches='tight')
        # plt.show()    

def get_modes(env, last_file):
    colors = set()
    reached_task_ids, reached_modes = [], []
    modes = [env.start_mode]
    counter = 0
    color = {}

    try:
        while True:     
            for mode in modes:
                reached_task_ids.append(mode.task_ids)
                reached_modes.append(mode)
                color_ = []
                while True:
                    c = np.random.rand(3,)
                    if tuple(c) not in colors:
                        for robot in env.robots:
                            color_.append(c)
                            colors.add(tuple(c))
                    if len(color_) == len(env.robots):
                        break
                color[f'{mode.task_ids}'] = color_
                counter += len(env.robots)
            if env.is_terminal_mode(mode):  
                break
            modes = env.get_next_modes(None, mode)
    except Exception:
        reached_task_ids, reached_modes = [], []
        modes = [env.start_mode]
        counter = 0
        color = {}
        start_conf = env.get_start_pos()
        with open(last_file, 'rb') as file:
            all_data = dill.load(file)
            all_transition_nodes = all_data['all_transition_nodes'] 
            all_transition_nodes_modes = all_data['all_transition_nodes_mode']
        q = type(env.get_start_pos())(all_transition_nodes[0], start_conf.array_slice)
        mode = env.start_mode
        modes = env.get_next_modes(q, mode)
        task_ids = mode.task_ids
        reached_task_ids.append(task_ids)
        reached_modes.append(mode)
        color_robots = []
        while True:
            c = np.random.rand(3,)
            if tuple(c) not in colors:
                color_robots.append(c)
                colors.add(tuple(c))
            if len(color_robots) == len(env.robots):
                break
        color[f'{mode.task_ids}'] = color_robots
        cnt = set()
        cnt.add(0)
        all_modes = set()
        all_modes.add(modes[0])
        while True:
            next_modes = set()
            for mode in modes:
                task_ids = mode.task_ids
                reached_task_ids.append(task_ids)
                reached_modes.append(mode)
                if f'{task_ids}' not in color:
                    color_robots = []
                    while True:
                        c = np.random.rand(3,)
                        if tuple(c) not in colors:
                            color_robots.append(c)
                            colors.add(tuple(c))
                        if len(color_robots) == len(env.robots):
                            break
                    color[f'{mode.task_ids}'] = color_robots
                counter += len(env.robots)
                matches = [i for i, lst in enumerate(all_transition_nodes_modes) if lst == task_ids]
                for m in matches: 
                    cnt.add(m)
                    q = type(env.get_start_pos())(all_transition_nodes[m], start_conf.array_slice)
                    if not env.is_terminal_mode(mode):
                        next_modes.update(env.get_next_modes(q, mode))
                    all_modes.add(mode)
            modes = list(next_modes)
            if len(cnt) == len(all_transition_nodes_modes) or not modes:
                break
    return reached_modes, reached_task_ids, color

def visualize_tree_2d_paper(env, path_to_folder):
    #TODO need to add forward_tree
    dir = os.path.join(os.path.dirname(path_to_folder),'tree_vis')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
       

    obstacles, table_size = get_infos_of_obstacles_and_table_2d(env)
    last_file = os.path.join(path_to_folder, os.listdir(path_to_folder)[-1])
    reached_modes, reached_task_ids, color = get_modes(env, last_file)
    q_start = env.get_start_pos().state()
    try:
        q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_task_ids[-1])
    except Exception:
        q_goal = env.tasks[-1].goal.sample(reached_task_ids[-1])
    for main_robot_idx, robot in enumerate(env.robots):
        dir_out = os.path.join(dir, str(main_robot_idx))
        os.makedirs(dir_out, exist_ok=True)
        for filename in os.listdir(path_to_folder):
            pkl_file = os.path.join(path_to_folder, filename)
            with open(pkl_file, 'rb') as file:
                all_data = dill.load(file)
                all_nodes = all_data['all_nodes']
                all_transition_nodes = all_data['all_transition_nodes']  
                all_nodes_modes = all_data['all_nodes_mode']
                all_transition_nodes_modes = all_data['all_transition_nodes_mode']
                
            fig, ax = plt.subplots()
            ax.set_xlim(1*table_size[0], -1*table_size[0])
            ax.set_ylim(1*table_size[1], -1.0*table_size[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(10)  # Thickness in points (increase as needed)
                spine.set_edgecolor("black")

            
            # Draw obstacles
            for obs in obstacles:
                x, y = obs["pos"]
                w, h = obs["size"]
                rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, color="black", zorder = 0)
                ax.add_patch(rect)
            for idx, tree_type in enumerate(['forward', 'reverse']):
                if len(all_data['forward']['nodes']) > 1:
                    if tree_type == 'reverse':
                        continue
                # if tree_type == 'reverse':
                #     break
                if idx == 0:
                    zorder = 2
                if idx == 1:
                    zorder = 1
                data = all_data[tree_type]
                nodes = data['nodes']
                parents = data['parents']
                modes = data['modes']
                # color = colors[idx]
                colors_plot, segments = [], []
                #tree
                lines = {'x': [], 'y': []}
                for robot_idx, robot in enumerate(env.robots):
                    if robot_idx != main_robot_idx:
                        continue
                    indices = env.robot_idx[robot]
                    for node, parent, mode in zip(nodes, parents, modes):   
                        if idx == 1:  
                            c = color[f'{mode}'][robot_idx]

                        else:
                            c = color[f'{mode}'][robot_idx]
                        x0 = node[indices][0]
                        y0 = node[indices][1]

                        if parent is not None:
                            x1 = parent[indices][0]
                            y1 = parent[indices][1]
                        else:
                            x1 = x0
                            y1 = y0
                        segments.append([[x0, y0], [x1, y1]])
                        lines['x'].extend([x0, x1, None])
                        lines['y'].extend([y0, y1, None])
                        colors_plot.append(c)

                segments = np.array(segments)

                # Plot lines with individual colors
                lc = LineCollection(segments, colors=colors_plot, linewidths=1, alpha=1, zorder=zorder)
                ax.add_collection(lc)

                # Optional: Scatter plot the same points
                x_vals = segments[:, 0, 0]
                y_vals = segments[:, 0, 1]
                ax.scatter(
                    x_vals,
                    y_vals,
                    color=colors_plot,
                    alpha=1,
                    s=4,
                    edgecolors='none',
                    zorder=zorder
                )


            all_points = []
            all_colors = []

            for robot_idx, robot in enumerate(env.robots):
                if robot_idx != main_robot_idx:
                        continue
                indices = env.robot_idx[robot]
                for n, mode in zip(all_nodes, all_nodes_modes):  # <- all_modes must match length of 'all'
                    if any(np.array_equal(n, x) for x in all_transition_nodes):
                        continue
                    x = n[indices][0]
                    y = n[indices][1]
                    all_points.append((x, y))
                    c = color[f'{mode}'][robot_idx]
                    all_colors.append(c)

            all_points = np.array(all_points)
            ax.scatter(
                all_points[:, 0],
                all_points[:, 1],
                color=all_colors,
                alpha=0.5,
                s=4,
                edgecolors='none',
                zorder=0,
            )
            all_points = []
            all_colors = []
            try:
                for robot_idx, robot in enumerate(env.robots):
                    if robot_idx != main_robot_idx:
                        continue
                    indices = env.robot_idx[robot]
                    for n, mode in zip(all_transition_nodes, all_transition_nodes_modes):  # all_nodes and all_nodes_modes must match
                        x = n[indices][0]
                        y = n[indices][1]
                        all_points.append((x, y))
                        c = color[f'{mode}'][robot_idx]
                        all_colors.append(c)

                all_points = np.array(all_points)
                ax.scatter(
                    all_points[:, 0],
                    all_points[:, 1],
                    color=all_colors,
                    alpha=0.5,
                    s=4,  # slightly larger dots
                    edgecolors='black',
                    linewidths=0.35,  # thinner edge
                    zorder=5
                )
            except Exception:
                pass


            for robot_idx, robot in enumerate(env.robots):
                if robot_idx != main_robot_idx:
                        continue
                indices = env.robot_idx[robot]
                x_start = q_start[indices][0]
                y_start = q_start[indices][1]
                x_goal = q_goal[indices][0]
                y_goal = q_goal[indices][1]

                # Plot a circle marker at the start point
                # ax.scatter(
                #     [x_goal],
                #     [y_goal],
                #     color='white',
                #     s=30,  # size of the circle
                #     edgecolors=colors[color[f'{reached_modes[-1].task_ids}'][robot_idx]],
                #     zorder=5
                # )

                # Plot a circle marker at the start point
                ax.scatter(
                    [x_start],
                    [y_start],
                    color='black',
                    s=13,  # size of the circle
                    edgecolors='none',
                    zorder=6
                )
                # #plot transition nodes
            for m in reached_modes:
                possible_next_task_combinations = env.get_valid_next_task_combinations(m)
                if not possible_next_task_combinations:
                    for robot_idx, robot in enumerate(env.robots):
                        if robot_idx != main_robot_idx:
                            continue
                        c = color[f'{mode}'][robot_idx]
                        indices = env.robot_idx[robot]
                        ax.scatter(
                        [q_goal[indices][0]],
                        [q_goal[indices][1]],
                        color='white',
                        s=30,  # size of the circle
                        edgecolors=c,
                        zorder=5
                    )
                    continue
                next_ids = random.choice(possible_next_task_combinations)
                constrained_robot = env.get_active_task(m, next_ids).robots
                goal = env.get_active_task(m, next_ids).goal.sample(m)
                q = []
                end_idx = 0
                for robot_idx, robot in enumerate(env.robots):
                    if robot_idx != main_robot_idx:
                        continue
                    c = color[f'{mode}'][robot_idx]
                    if robot in constrained_robot:
                        dim = env.robot_dims[robot]
                        indices = list(range(end_idx, end_idx + dim))
                        q = goal[indices]
                        ax.scatter(
                            [q[0]],
                            [q[1]],
                            color=c,
                            s=30,  # size of the circle
                            edgecolors='black',
                            zorder=6
                        )
                        end_idx += dim 
                        continue
            try:
                nodes = all_data['pathnodes']
                parents = all_data['pathparents']
                # color = colors[idx]
                colors_plot, segments = [], []
                #tree
                lines = {'x': [], 'y': []}
                for robot_idx, robot in enumerate(env.robots):
                    if robot_idx != main_robot_idx:
                        continue
                    indices = env.robot_idx[robot]
                    for node, parent in zip(nodes, parents):   
                        c = 'black'
                        x0 = node[indices][0]
                        y0 = node[indices][1]

                        if parent is not None:
                            x1 = parent[indices][0]
                            y1 = parent[indices][1]
                        else:
                            x1 = x0
                            y1 = y0
                        segments.append([[x0, y0], [x1, y1]])
                        lines['x'].extend([x0, x1, None])
                        lines['y'].extend([y0, y1, None])
                        colors_plot.append(c)

                segments = np.array(segments)

                # Plot lines with individual colors
                lc = LineCollection(segments, colors=colors_plot, linewidths=2.5, alpha=1, zorder=6)
                ax.add_collection(lc)

                # Optional: Scatter plot the same points
                x_vals = segments[:, 0, 0]
                y_vals = segments[:, 0, 1]
                ax.scatter(
                    x_vals,
                    y_vals,
                    color=colors_plot,
                    alpha=1,
                    s=2.5,
                    edgecolors='none',
                    zorder=6
                )
            except Exception:
                pass

            next_file_number = max(
                    (int(file.split('.')[0]) for file in os.listdir(dir_out)
                    if file.endswith('.png') and file.split('.')[0].isdigit()),
                    default=-1
                ) + 1
            
            plt.savefig(os.path.join(dir_out,f"{next_file_number:04d}.png"), dpi=300, bbox_inches='tight') 
        generate_gif(dir_out)

def visualize_tree_2d_with_color_html(env, env_path, pkl_folder, output_html):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except:
        try:
            task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  
        except:
            task_sequence_text = "No task sequence available"
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )  

    reached_modes, reached_task_ids, color = get_modes(env, pkl_files[-1])

    q_start = env.get_start_pos().state()
    try:
        q_goal = env.tasks[env.sequence[-1]].goal.sample(reached_task_ids[-1])
    except:
        q_goal = env.tasks[-1].goal.sample(reached_task_ids[-1])

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
        for tree_type in ['forward', 'reverse', '']:
            if tree_type == '':
                legend_group = robot
            else:
                legend_group = tree_type + ' ' + robot
            static_traces.append(
            go.Mesh3d(
                x=[0],  # X-coordinates of the exterior points
                y=[0],  # Y-coordinates of the exterior points
                z=[0] ,  # Flat surface at z = 0
                color=colors[len(reached_task_ids)+robot_idx],  # Fill color from the agent's properties
                opacity=1,  # Transparency level
                name=legend_group,
                legendgroup=legend_group,
                showlegend=True
            )
        )
    # if with_tree:
    legends = []
    for idx in range(len(reached_task_ids)):
        name = f"Mode: {reached_task_ids[idx]}"
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
        for tree_type in ['forward', 'reverse']:
            data = all_data[tree_type]
            nodes = data['nodes']
            parents = data['parents']
            modes = data['modes']
            # if tree_type == 'reverse':
            #     continue
            
            #forward and reverse tree
            for robot_idx, robot in enumerate(env.robots):
                lines_by_color = {}
                indices = env.robot_idx[robot]
                for node, parent, m in zip(nodes, parents, modes):    
                    mode_idx = reached_task_ids.index(m)            
                    color_idx = colors[mode_idx]
                    if tree_type == 'forward':
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
                            name=tree_type + ' ' + robot,
                            legendgroup=tree_type + ' ' + robot,
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
            tree_type="buttons",
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
        dir_path_pkl = args.path_to_pkl_file
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
