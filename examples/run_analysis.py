import argparse
import plotly.graph_objects as go
import os
import numpy as np
import time as time
import dill
import random
import traceback
from analysis_util import(
    colors_plotly,
    mesh_traces_env,
    save_env_as_mesh
    )

from multi_robot_multi_goal_planning.problems import get_env_by_name

def ellipse_with_samples(env, env_path, pkl_file):

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
                    marker=dict(size=3, color=color),
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

def ellipse_with_matrices(env, env_path, pkl_file):

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
        "--path",
        required=False,
        help="Select the path of the folder to be processed",
    )
    parser.add_argument("do", nargs="?", choices=["ellipse_with_samples", "ellipse_with_matrices"], default="", help="action to do")
    parser.add_argument("env_name", nargs="?", default="", help="name of environment")
    parser.add_argument("seed", nargs="?", default="", help="seed")


    args = parser.parse_args()
    if args.path is not None:
        dir_path_pkl = args.path
    else:
        #get latest created folder
        home_dir = os.path.expanduser("~")
        dir = os.path.join(home_dir, 'multirobot-pathplanning-benchmark/out')
        dir_out = os.path.join(dir, 'Analysis')
        os.makedirs(dir_out, exist_ok=True)
        files = [f for f in os.listdir(dir_out) if os.path.isfile(os.path.join(dir_out, f))]
        pkl_file = sorted(files)[-1]
        dir_path_pkl = os.path.join(dir_out, pkl_file)

    seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    env = get_env_by_name(args.env_name)  
    home_dir = os.path.expanduser("~")
    env_path = os.path.join(home_dir, f'env/{args.env_name}')  
    save_env_as_mesh(env, env_path) 
    
    if args.do == "ellipse_with_samples":
        ellipse_with_samples(env, env_path, dir_path_pkl) 
    if args.do == "ellipse_with_matrices":
        ellipse_with_matrices(env, env_path, dir_path_pkl) 
       
    

