import argparse
import dill
import os
from rai_envs import *
from planning_env import *
from util import *
from rai_config import *
from planning_env import *
from util import *
from analysis.analysis_util import *
from rrtstar_planner import *
import webbrowser

def init_discretization(path, costs, modes, indices, transition, resolution=0.05):
    discretized_path, discretized_modes, discretized_costs, discretized_transition = [], [], [], []
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        if i == 0:
            discretized_costs.append(costs[i])
            
        # Calculate the vector difference and the length of the segment
        segment_vector = end - start
        segment_length = np.linalg.norm(segment_vector)
        
        # Determine the number of points needed for the current segment
        num_points = max(int(segment_length // resolution), 1)
        
        # Create the points along the segment
        mode_idx = i
        for j in range(num_points):
            interpolated_point = start + (segment_vector * (j / num_points))
            q_list = [interpolated_point[indices[i]] for i in range(len(indices))]
            mode = [modes[mode_idx]]
            if transition[i] and j == 0:
                discretized_transition.append(True)
                mode_idx +=1
                mode.append(modes[mode_idx])
            else:
                discretized_transition.append(False)

            discretized_path.append(State(ListConfiguration(q_list), mode))
            discretized_modes.append(mode)
            if i == 0 and j == 0:
                continue
            discretized_costs.append(discretized_costs[-1] + config_dist(discretized_path[-2].q, discretized_path[-1].q, "euclidean"))

    # Append the final point of the path
    q_list = [path[-1][indices[i]] for i in range(len(indices))]
    mode = [modes[i+1]]
    discretized_path.append(State(ListConfiguration(q_list), mode))
    discretized_modes.append(mode)
    discretized_costs.append(discretized_costs[-1]+ config_dist(discretized_path[-2].q, discretized_path[-1].q, "euclidean"))
    discretized_transition.append(True)
    return discretized_path, discretized_modes, discretized_costs, discretized_transition

def update(path , cost, idx2):
    
    if idx2 == len(cost)-1:
        return
    idx = idx2 +1
    while True:
        cost[idx] = cost[idx -1] + config_dist(path[idx-1].q, path[idx].q, "euclidean")
        # agent_cost[idx] = agent_cost[idx -1] + config_agent_dist(path[idx-1].q, path[idx].q, "euclidean")
        if idx == len(path)-1:
            break
        idx+=1

def interpolate(path, m1, cost, indices):
    q0 = path[0].q.state()
    q1 = path[-1].q.state()
    edge, edge_modes = [], []
    edge_cost = [cost]
    segment_vector = q1 - q0

    if len(m1) < 2 :
        mode = m1
    else:
        mode = [m1[1]]
    for i in range(len(path)):
        q = q0 +  (segment_vector * (i / len(path)))
        q_list = [q[indices[i]] for i in range(len(indices))]
        if i == 0:
            edge.append(State(ListConfiguration(q_list), m1))
            edge_modes.append(m1)
            continue
        edge.append(State(ListConfiguration(q_list), mode))
        edge_cost.append(cost + config_dist(edge[-2].q, edge[-1].q, "euclidean"))
        edge_modes.append(mode)

    return edge, edge_modes, edge_cost

# def interpolate(path, m1, cost, indices, resolution = 0.05):
#     q0 = path[0].q.state()
#     q1 = path[-1].q.state()modes_legend
#     edge, edge_modes = [], []
#     edge_cost = [cost]
#     segment_vector = q1 - q0
#     segment_length = np.linalg.norm(segment_vector)
        
#     # Determine the number of points needed for the current segment
#     num_points = max(int(segment_length // resolution), 1)
#     if len(m1) < 2 :
#         mode = m1
#     else:
#         mode = [m1[1]]
#     for i in range(num_points):
#         q = q0 +  (segment_vector * (i / num_points))
#         q_list = [q[indices[i]] for i in range(len(indices))]
#         if i == 0:
#             edge.append(State(ListConfiguration(q_list), m1))
#             edge_modes.append(m1)
#             continue
#         edge.append(State(ListConfiguration(q_list), mode))
#         edge_cost.append(cost + config_dist(edge[-2].q, edge[-1].q, "euclidean"))
#         edge_modes.append(mode)

#     return edge, edge_modes, edge_cost

def path_traces(colors, modes, path):
    trace = []
    for robot_idx, robot in enumerate(env.robots):
        path_x = [state.q[robot_idx][0] for state in path]
        path_y = [state.q[robot_idx][1] for state in path]
        path_z = [1 for _ in path]
        trace.append(go.Scatter3d(
                    x=path_x, 
                    y=path_y,
                    z=path_z,
                    mode="lines+markers",
                    line=dict(color=colors[len(modes)+robot_idx], width=6),
                    marker=dict(
                        size=4,  # Very small markers
                        color=colors[len(modes)+robot_idx],  # Match marker color with line
                        opacity=1
                    ),
                    opacity=1,
                    name=robot,
                    legendgroup=robot,
                    showlegend=False
                ))


    return trace

def path_visualization(all_frame_traces, env_path, modes_legend, original_path, original_transition, output_html):

    # Print the parsed task sequence
    try:
        task_sequence_text = "Task sequence: " + ", ".join(
        [env.tasks[idx].name for idx in env.sequence]   
    )
    except:
         task_sequence_text = f"Task sequence consists of {len(env.sequence)} tasks"  

    # Initialize figure and static elements
    fig = go.Figure()
    frames = []
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
    
    for idx in range(len(modes_legend)):
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
                name=f'Transitions {idx}: {modes_legend[idx]}',
                legendgroup= f'{idx}',
                showlegend=True
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
    tot_frmes = []
    for frame_traces in all_frame_traces:
        for robot_idx, robot in enumerate(env.robots):
            transition_x = [state.q[robot_idx][0] for idx, state in enumerate(original_path) if original_transition[idx]]
            transition_y = [state.q[robot_idx][1] for idx, state in enumerate(original_path) if original_transition[idx]]
            transition_z = [1] * len(transition_x)
            for idx in range(len(transition_x)):
                frame_traces.append(
                    go.Scatter3d(
                        x=[transition_x[idx]], 
                        y=[transition_y[idx]],
                        z=[transition_z[idx]],
                        mode="markers",
                        marker=dict(
                            size=5,  # Very small markers
                            color="red",  # Match marker color with line
                            opacity=1
                        ),
                        opacity=1,
                        name=f'Transitions {idx}: {modes_legend[idx]}',
                        legendgroup=f'{idx}',
                        showlegend=False
                    )
                )
        tot_frmes.append(frame_traces)
        
    all_frame_traces = tot_frmes
    # Determine the maximum number of dynamic traces needed
    max_dynamic_traces = max(len(frame_traces) for frame_traces in all_frame_traces)



    fig.add_traces(static_traces)

    for _ in range(max_dynamic_traces):
        fig.add_trace(go.Mesh3d(x=[], y=[], z =[]))

    frames = []
    for idx, frame_traces in enumerate(all_frame_traces):
        
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

def shortcutting(dir, env, env_path, pkl_folder, output_html):

    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    with open(pkl_files[-1], 'rb') as file:
        data = dill.load(file)
        results = data["result"]
        path_ = results["path"]
        total_cost = results["total"]
        agent_cost = results["agent_cost"]
        intermediate_tot = results["intermediate_tot"]
        intermediate_agent_cost = results["intermediate_agent_cost"]
        transition = results["is_transition"]
        modes = results["modes"]
        colors = colors_plotly()
        all_frame_traces = []
        mode_sequence = []
        m = env.start_mode
        indices  = [env.robot_idx[r] for r in env.robots]
        while True:
            mode_sequence.append(m)
            if m == env.terminal_mode:
                break
            m = env.get_next_mode(None, m)

        discretized_path, discretized_modes, discretized_costs, discretized_transition =init_discretization(path_, intermediate_tot, modes, indices, transition)  
        original_discretized_path = discretized_path.copy()     
        original_discretized_modes = discretized_modes.copy()      
        modes_legend = [mode for idx, mode in enumerate(original_discretized_modes) if discretized_transition[idx]]
        all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
        
        for i in range(1000):
            i1 = np.random.choice(len(discretized_path))
            i2 = np.random.choice(len(discretized_path))
            idx1 = min(i1, i2)
            idx2 = max(i1, i2)
            m1 = discretized_modes[idx1]    
            m2 = discretized_modes[idx2]    
            if len(m1) > len(m2):
                if m2[0] not in m1:
                    continue
            elif len(m2) > len(m1):
                if m1[0] not in m2:
                    continue
            elif m1 != m2:
                continue
            
            state1 = discretized_path[idx1]
            state2 = discretized_path[idx2]
            c_new = discretized_costs[idx1] + config_dist(state1.q, state2.q, "euclidean")
            if c_new < discretized_costs[idx2] and env.is_edge_collision_free(state1.q, state2.q, m1): # also need to check the terminal cost?
                discretized_costs[idx2] = c_new
                # intermediate_agent_cost[idx2] =  intermediate_agent_cost[idx1] + config_agent_dist(state1.q, state2.q, "euclidean")
                update(discretized_path, discretized_costs, idx2)
                edge, edge_modes, edge_cost =  interpolate(discretized_path[idx1:idx2+1], m1, discretized_costs[idx1], indices)
                discretized_path[idx1:idx2] = edge
                discretized_costs[idx2] = c_new
                discretized_costs[idx1:idx2] = edge_cost
                # intermediate_agent_cost[idx1:idx2+1] = [intermediate_agent_cost[idx1], intermediate_agent_cost[idx2]]
                discretized_modes[idx1:idx2] = edge_modes

                all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
        path_visualization(all_frame_traces, env_path,modes_legend,original_discretized_path, discretized_transition, output_html)
        path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(discretized_path)}
        json_dir =os.path.join(dir, 'general.log') 
        log_entry1 = f"Path shortcut: {json.dumps(path_dict, indent=4)}\n"
        log_entry2 = f"Before: {json.dumps(total_cost, indent=4)}\n"
        log_entry3 = f"Afzter: {json.dumps(discretized_costs[-1], indent=4)}\n"
        # Append to the log file
        with open(json_dir, 'a') as log_file:
            log_file.write(log_entry1)
            log_file.write(log_entry2)
            log_file.write(log_entry3)
        print("Before: ", total_cost, "After " ,   discretized_costs[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    parser.add_argument(
        "--do",
        choices=["shortcutting"],
        required=True,
        help="Select the mode of operation",
    )
    args = parser.parse_args()
    home_dir = os.path.expanduser("~")
    directory = os.path.join(home_dir, 'output')
    path = get_latest_folder(directory)
    env_name, config_params, _, _ = get_config(path)
    env = get_env_by_name(env_name)    
    pkl_folder = os.path.join(path, 'FramesData')
    env_path = os.path.join(home_dir, f'env/{env_name}')
    save_env_as_mesh(env, env_path)

    if args.do == "shortcutting":
        output_html = os.path.join(path, 'shortcutting.html')
        shortcutting(path, env, env_path, pkl_folder, output_html)
        webbrowser.open('file://' + os.path.realpath(output_html))



