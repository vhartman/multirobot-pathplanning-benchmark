
import sys
import dill
import plotly.graph_objects as go
import os
from analysis.analysis_util import *

def interpolation_check(env):
    array = np.array
    colors = colors_plotly()

    interpolated_states = [array([0.50973078, 0.08875066, 0.39892376, 0.5       , 0.5       ,
       0.        ]), array([0.39538828, 0.15265072, 0.30783377, 0.5       , 0.38888889,
       0.        ]), array([0.28104578, 0.21655078, 0.21674377, 0.5       , 0.27777778,
       0.        ]), array([0.16670327, 0.28045083, 0.12565378, 0.5       , 0.16666667,
       0.        ]), array([0.05236077, 0.34435089, 0.03456378, 0.5       , 0.05555556,
       0.        ]), array([-0.06198173,  0.40825095, -0.05652621,  0.5       , -0.05555556,
        0.        ]), array([-0.17632423,  0.472151  , -0.1476162 ,  0.5       , -0.16666667,
        0.        ]), array([-0.29066674,  0.53605106, -0.2387062 ,  0.5       , -0.27777778,
        0.        ]), array([-0.40500924,  0.59995112, -0.32979619,  0.5       , -0.38888889,
        0.        ])]
    # interpolated_states = [[np.float64(-0.0872441810838756), np.float64(-0.5424920511404258), np.float64(-0.16872908700385802), np.float64(-0.08130792508826995), np.float64(0.5142211315673857), np.float64(-0.21907653166446994)], 
    #                        [np.float64(-0.1081872806631973), np.float64(-0.451645997510291), np.float64(-0.39542301053911505), np.float64(-0.09125921142967884), np.float64(0.3668265669203842), np.float64(-0.03970304664302826)]]
    fig = go.Figure()
    static_traces = []

    for r, robot in enumerate(env.robots):
        legend = True
        indices = env.robot_idx[robot]
        for i, state in enumerate(interpolated_states):
            state = np.array(state)
            states = state[indices]
            static_traces.append(go.Scatter3d(
                x=[states[0]],
                y=[states[1]],
                z = [0],
                mode='markers',
                name=f"Agent {r}",
                marker=dict(size=5, color=colors[r]),
                legendgroup = f"Agent {r}",
                showlegend = legend,
            ))
            if i != 0:
                u = np.cos(states[2]) * 0.2
                v = np.sin(states[2]) * 0.2
                w = 0 
                static_traces.append(
                    go.Scatter3d(
                        x=[states[0], states[0] + u],  # Line from base to arrowhead
                        y=[states[1], states[1] + v],
                        z=[0, 0],  # Constant z (no change in height)
                        mode='lines',
                        line=dict(color=colors[r], width=1),  # Arrow shaft style
                        showlegend=False,
                    )
                )

                # Add the arrowhead as a cone
                static_traces.append(
                    go.Cone(
                        x=[states[0] + u],  # End position of the arrow shaft
                        y=[states[1] + v],
                        z=[0],  # Arrowhead position
                        u=[u],  # x-direction
                        v=[v],  # y-direction
                        w=[w],  # z-direction
                        colorscale=[[0, colors[r]], [1, colors[r]]],  # Coloring the arrow
                        showscale=False,
                        sizemode="absolute",
                        sizeref=0.01,  # Adjust the arrowhead size
                    )
                )

            if i == 0 or i == len(interpolated_states)-1:
                if i == 0:
                    text = 'S'
                else: text = 'E'
                static_traces.append(
                    go.Scatter3d(
                        x=[states[0]],  # Single x coordinate for text
                        y=[states[1]],  # Single y coordinate for text
                        z=[0],
                        text=[text],  # Text label
                        mode="text",  # Only text mode
                        textposition="middle center",
                        textfont=dict(
                            color="white",
                            size=10,
                            family="Arial Black"
                        ),
                        showlegend=False
                    )
                )
            
            u = np.cos(interpolated_states[0][2]) * 0.1
            v = np.sin(interpolated_states[0][2]) * 0.1
            w = 0 
            static_traces.append(
                go.Scatter3d(
                    x=[states[0], states[0] + u],  # Line from base to arrowhead
                    y=[states[1], states[1] + v],
                    z=[0, 0],  # Constant z (no change in height)
                    mode='lines',
                    line=dict(color="grey", width=1),  # Arrow shaft style
                    showlegend=False,
                )
            )

            # Add the arrowhead as a cone
            static_traces.append(
                go.Cone(
                    x=[states[0] + u],  # End position of the arrow shaft
                    y=[states[1] + v],
                    z=[0],  # Arrowhead position
                    u=[u],  # x-direction
                    v=[v],  # y-direction
                    w=[w],  # z-direction
                    colorscale=[[0, "grey"], [1, "grey"]],  # Coloring the arrow
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.01,  # Adjust the arrowhead size
                )
            )

            legend = False

    fig.add_traces(static_traces)
    fig.update_layout(
        title="Interpolation",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure equal aspect ratio
        showlegend=True
    )

    fig.show()

def nearest_neighbor(config, env, env_path, pkl_folder, output_html, with_tree, divider = None):

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
        frame = 0
    print(f'Take every {frame}th frame')
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
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
   
    
    # dynamic_traces
    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
            tree = data["tree"]
            results = data["result"]
            inter_results = data["inter_result"]
            inter_paths = inter_results[0]['path']
            path = results["path"]
            transition = results["is_transition"]
            N_near = data["N_near"]
            N_parent = data["N_parent"]
            # print(N_near)
            rewire_radius = data["rewire_r"]
            n_new = data["n_new"]
            n_nearest = data["n_nearest"]
            n_rand= data["n_rand"]


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
                            marker=dict(size=5, color=color),
                            line=dict(color=color, width=8),
                            opacity=0.1,
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=False
                        )
                    )

        active_mode = data["active_mode"]
        mode_idx = modes.index(active_mode)
        legend_bool = True
        for r, robot in enumerate(env.robots):
            indices = env.robot_idx[robot]
            if n_new is not None and N_near is not None and N_near != []:
                theta = np.linspace(0, 2 * np.pi, 100)  # Angle values
                x = n_new[indices][0] + rewire_radius * np.cos(theta)  # X-coordinates
                y = n_new[indices][1] + rewire_radius * np.sin(theta)  # Y-coordinates
                z = [1] * len(x)
                frame_traces.append(
                go.Scatter3d(
                    x=x,  # X-coordinates
                    y=y,  # Y-coordinates
                    z=z,  # Z-coordinates
                    mode='lines',  # Use 'lines' or 'markers+lines' for connections
                    marker=dict(size=9, color=colors[len(modes)+r]), 
                    legendgroup = legends[mode_idx],
                    showlegend = False  # Marker style
                    )
                )
            if n_new is not None:
                frame_traces.append(
                go.Scatter3d(
                    x=[n_new[indices][0]],  # X-coordinates
                    y=[n_new[indices][1]],  # Y-coordinates
                    z=[1],  # Z-coordinates
                    mode='markers',  # Use 'lines' or 'markers+lines' for connections
                    marker=dict(size=9, color='black'), 
                    legendgroup = legends[mode_idx],
                    showlegend = legend_bool,  # Marker style
                    name = 'New Node'
                    )
                )
            if n_nearest is not None:
                frame_traces.append(
                go.Scatter3d(
                    x=[n_nearest[indices][0]],  # X-coordinates
                    y=[n_nearest[indices][1]],  # Y-coordinates
                    z=[1],  # Z-coordinates
                    mode='markers',  # Use 'lines' or 'markers+lines' for connections
                    marker=dict(size=9, color=colors[len(modes)+r]), 
                    legendgroup = legends[mode_idx],
                    showlegend = True , # Marker style
                    name = f'Nearest Node {robot}'
                    )
                )
            if n_rand is not None:
                frame_traces.append(
                go.Scatter3d(
                    x=[n_rand[indices][0]],  # X-coordinates
                    y=[n_rand[indices][1]],  # Y-coordinates
                    z=[1],  # Z-coordinates
                    mode='markers',  # Use 'lines' or 'markers+lines' for connections
                    marker=dict(size=6, color='red'), 
                    legendgroup = legends[mode_idx],
                    showlegend = legend_bool,  # Marker style,
                    name = 'Random Node'
                    )
                )
            if N_near is not None and N_near != []:
                for _, state in enumerate(N_near):
                    try:
                        states = np.array(state.cpu().numpy())[indices]
                        x_state = [states[0]]
                        y_state = [states[1]]
                    except:
                        x_state = [state[0]]
                        y_state = [state[1]] 


                    frame_traces.append(go.Scatter3d(
                        x=x_state,
                        y=y_state,
                        z = [1],
                        mode='markers',
                        name=robot,
                        marker=dict(size=5, color=colors[len(modes)+r]),
                        legendgroup = legends[mode_idx],
                        showlegend = False,
                    ))
            if N_parent is not None and N_parent != []:
                n_parent_bool = True
                for _, state in enumerate(N_parent):
                    states = np.array(state.cpu().numpy())[indices]
                    frame_traces.append(go.Scatter3d(
                        x=[states[0]],
                        y=[states[1]],
                        z = [1],
                        mode='markers',
                        name= f"Ancestors {robot}",
                        marker=dict(size=14, color=colors[len(modes)+r]),
                        legendgroup = legends[mode_idx],
                        showlegend = n_parent_bool,
                        opacity=0.2
                    ))
                    n_parent_bool = False
            legend_bool = False
        #graph
        try:

            graph = data["graph"]
            graph_transition = data["graph_transition"]

            graph_x = [g[0] for g in graph]
            graph_y = [g[1] for g in graph]
            graph_z = [1 for g in graph]

            frame_traces.append(
                go.Scatter3d(
                    x=graph_x, 
                    y=graph_y,
                    z=graph_z,
                    mode="markers",
                    marker=dict(
                        size=8,  # Very small markers
                        color='black',  # Match marker color with line
                        opacity=1
                    ),
                    opacity=0.5,
                    name='Graph Nodes',
                    showlegend=True
                )
            )

            graph_x = [g[0] for g in graph_transition]
            graph_y = [g[1] for g in graph_transition]
            graph_z = [1 for g in graph_transition]

            frame_traces.append(
                go.Scatter3d(
                    x=graph_x, 
                    y=graph_y,
                    z=graph_z,
                    mode="markers",
                    marker=dict(
                        size=8,  # Very small markers
                        color='black',  # Match marker color with line
                        opacity=1
                    ),
                    opacity=1,
                    name='Graph Transitions',
                    showlegend=True
                )
            )

        except:
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
    fig.show()
    fig.write_html(output_html)
    print(f"Animation saved to {output_html}")

def tree(config, env, env_path, pkl_folder, divider = None):

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
        frame = 0
    print(f'Take every {frame}th frame')
    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    print(f'Take {len(pkl_files)} .pkl files')

    # Initialize figure and static elements
    fig = go.Figure()
    frames = []
    all_frame_traces = []
    colors = colors_plotly()
    # static traces
    # static_traces = mesh_traces_env(env_path)
    static_traces = []
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
   
    
    # dynamic_traces
    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)
            tree = data["tree"]
            results = data["result"]
            path = results["path"]
            transition = results["is_transition"]
            N_near = data["N_near"]
            rewire_radius = data["rewire_r"]
            n_new = data["n_new"]


        frame_traces = []

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

        if N_near is not None and N_near != []:
            active_mode = data["active_mode"]
            mode_idx = modes.index(active_mode)
            for r, robot in enumerate(env.robots):
                indices = env.robot_idx[robot]
                n_new = np.array(n_new)
                theta = np.linspace(0, 2 * np.pi, 100)  # Angle values
                x = n_new[indices][0] + rewire_radius * np.cos(theta)  # X-coordinates
                y = n_new[indices][1] + rewire_radius * np.sin(theta)  # Y-coordinates
                z = [1] * len(x)
                frame_traces.append(
                go.Scatter3d(
                    x=x,  # X-coordinates
                    y=y,  # Y-coordinates
                    z=z,  # Z-coordinates
                    mode='lines',  # Use 'lines' or 'markers+lines' for connections
                    marker=dict(size=6, color=colors[len(modes)+r]), 
                    legendgroup = legends[mode_idx],
                    showlegend = False  # Marker style
                    )
                )
                for i, state in enumerate(N_near):
                    states = np.array(state.cpu().numpy())[indices]

                    frame_traces.append(go.Scatter3d(
                        x=[states[0]],
                        y=[states[1]],
                        z = [1],
                        mode='markers',
                        name=robot,
                        marker=dict(size=5, color=colors[len(modes)+r]),
                        legendgroup = legends[mode_idx],
                        showlegend = False,
                    ))
                    # if i != 0:
                    #     u = np.cos(states[2]) * 0.2
                    #     v = np.sin(states[2]) * 0.2
                    #     w = 0 
                    #     frame_traces.append(
                    #         go.Scatter3d(
                    #             x=[states[0], states[0] + u],  # Line from base to arrowhead
                    #             y=[states[1], states[1] + v],
                    #             z=[0, 0],  # Constant z (no change in height)
                    #             mode='lines',
                    #             line=dict(color=color_, width=3),  # Arrow shaft style
                    #             showlegend=False,
                    #         )
                    #     )

                    #     # Add the arrowhead as a cone
                    #     frame_traces.append(
                    #         go.Cone(
                    #             x=[states[0] + u],  # End position of the arrow shaft
                    #             y=[states[1] + v],
                    #             z=[0],  # Arrowhead position
                    #             u=[u],  # x-direction
                    #             v=[v],  # y-direction
                    #             w=[w],  # z-direction
                    #             colorscale=[[0, color_], [1, color_]],  # Coloring the arrow
                    #             showscale=False,
                    #             sizemode="absolute",
                    #             sizeref=0.05,  # Adjust the arrowhead size
                    #         )
                    #     )
                    

                    # u = np.cos(N_near[0][2]) * 0.2
                    # v = np.sin(N_near[0][2]) * 0.2
                    # w = 0 
                    # frame_traces.append(
                    #     go.Scatter3d(
                    #         x=[states[0], states[0] + u],  # Line from base to arrowhead
                    #         y=[states[1], states[1] + v],
                    #         z=[0, 0],  # Constant z (no change in height)
                    #         mode='lines',
                    #         line=dict(color="grey", width=3),  # Arrow shaft style
                    #         showlegend=False,
                    #     )
                    # )

                    # # Add the arrowhead as a cone
                    # frame_traces.append(
                    #     go.Cone(
                    #         x=[states[0] + u],  # End position of the arrow shaft
                    #         y=[states[1] + v],
                    #         z=[0],  # Arrowhead position
                    #         u=[u],  # x-direction
                    #         v=[v],  # y-direction
                    #         w=[w],  # z-direction
                    #         colorscale=[[0, "grey"], [1, "grey"]],  # Coloring the arrow
                    #         showscale=False,
                    #         sizemode="absolute",
                    #         sizeref=0.05,  # Adjust the arrowhead size
                    #     )
                    # )


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
    fig.show()
    # fig.write_html(output_html)
    # print(f"Animation saved to {output_html}")

def graph(env, env_path, pkl_folder, output_html, divider = None):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except:
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
   
    # dynamic_traces
    for idx, pkl_file in enumerate(pkl_files):
        with open(pkl_file, 'rb') as file:
            data = dill.load(file)

        frame_traces = []

        graph = data["graph"]
        graph_transition = data["graph_transition"]

        graph_x = [g[0] for g in graph]
        graph_y = [g[1] for g in graph]
        graph_z = [1 for g in graph]

        frame_traces.append(
            go.Scatter3d(
                x=graph_x, 
                y=graph_y,
                z=graph_z,
                mode="markers",
                marker=dict(
                    size=8,  # Very small markers
                    color='black',  # Match marker color with line
                    opacity=1
                ),
                opacity=1,
                name='Graph Nodes',
                showlegend=True
            )
        )

        graph_x = [g[0] for g in graph_transition]
        graph_y = [g[1] for g in graph_transition]
        graph_z = [1 for g in graph_transition]

        frame_traces.append(
            go.Scatter3d(
                x=graph_x, 
                y=graph_y,
                z=graph_z,
                mode="markers",
                marker=dict(
                    size=8,  # Very small markers
                    color='black',  # Match marker color with line
                    opacity=1
                ),
                opacity=1,
                name='Graph Transitions',
                showlegend=True
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
    # fig.show()
    fig.write_html(output_html)
    print(f"Animation saved to {output_html}")
