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
from planner_rrtstar import *

def init_discretization(path, costs, modes, indices, transition, resolution=0.1):
    discretized_path, discretized_modes, discretized_costs, discretized_transition = [], [], [], []
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        if i == 0:
            discretized_costs.append(costs[i].item())
            
        # Calculate the vector difference and the length of the segment
        segment_vector = end - start
        segment_length = np.linalg.norm(segment_vector)
        
        # Determine the number of points needed for the current segment
        if resolution is None: 
            num_points = 1
        else:
            num_points = max(int(segment_length // resolution), 1)
        # num_points = 1
        
        # Create the points along the segment
        mode_idx = i
        
        for j in range(num_points):
            interpolated_point = start + (segment_vector * (j / num_points))
            q_list = [interpolated_point[indices[i]] for i in range(len(indices))]
            mode = [modes[mode_idx]]
            if transition[i] and j == 0:
                discretized_transition.append(True)
                mode_idx +=1
                if mode[0] != modes[mode_idx]:
                    mode.append(modes[mode_idx])

            else:
                discretized_transition.append(False)
            if j == 0:
                original_mode = mode[0]
            else:
                original_mode = mode[-1]
            discretized_path.append(State(ListConfiguration(q_list), original_mode))
            discretized_modes.append(mode)
            if i == 0 and j == 0:
                continue
            discretized_costs.append(discretized_costs[-1] + config_cost(discretized_path[-2].q, discretized_path[-1].q, "euclidean"))

    # Append the final point of the path
    q_list = [path[-1][indices[i]] for i in range(len(indices))]
    mode = [modes[i+1]]
    discretized_path.append(State(ListConfiguration(q_list), original_mode))
    discretized_modes.append(mode)
    discretized_costs.append(discretized_costs[-1]+ config_cost(discretized_path[-2].q, discretized_path[-1].q, "euclidean"))
    discretized_transition.append(True)
    return discretized_path, discretized_modes, discretized_costs, discretized_transition

def update(path , cost, idx):
    while True:
        cost[idx] = cost[idx -1] + config_cost(path[idx-1].q, path[idx].q, "euclidean")
        # agent_cost[idx] = agent_cost[idx -1] + config_agent_dist(path[idx-1].q, path[idx].q, "euclidean")
        if idx == len(path)-1:
            break
        idx+=1

def interpolate(path, cost, modes, indices, dim, version, r = None, robots =None):
    q0 = path[0].q.state()
    q1 = path[-1].q.state()
    edge  = []
    edge_cost = [cost]
    segment_vector = q1 - q0
    # dim_indices = [indices[i][dim] for i in range(len(indices))]

    for i in range(len(path)):
        mode = modes[i]
        if len(mode) > 1:
            if i == 0:
                mode = mode[1]
            else:
                mode = mode[0]
            
        if version == 0 :
            q = q0 +  (segment_vector * (i / len(path)))

        elif version == 3: #shortcutting agent
            q = path[i].q.state()
            for robot in range(len(robots)):
                if r is not None and r == robot:
                    q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i / len(path)))
                    break
                if r is None:
                    q[indices[robot]] = q0[indices[robot]] +  (segment_vector[indices[robot]] * (i / len(path)))
                
        elif version == 1:
            q = path[i].q.state()
            q[dim] = q0[dim] + ((q1[dim] - q0[dim])* (i / len(path)))

        elif version == 4: #partial shortcutting agent single dim 
            q = path[i].q.state()
            for robot in range(len(robots)):
                if r is not None and r == robot:
                    q[dim] = q0[dim] +  (segment_vector[dim] * (i / len(path)))
                    break
                if r is None:
                    q[dim[robot]] = q0[dim[robot]] +  (segment_vector[dim[robot]] * (i / len(path)))

        elif version == 2:
            q = path[i].q.state()
            for idx in dim:
                q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / len(path)))
        
        elif version == 5: #partial shortcutting agent random set of dim 
            q = path[i].q.state()
            for robot in range(len(robots)):
                if r is not None and r == robot:
                    for idx in dim:
                        q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / len(path)))
                    break
                if r is None:
                    for idx in dim[robot]:
                        q[idx] = q0[idx] + ((q1[idx] - q0[idx])* (i / len(path)))

        q_list = [q[indices[i]] for i in range(len(indices))]
        if i == 0:
            edge.append(State(ListConfiguration(q_list),mode))
            continue
        edge.append(State(ListConfiguration(q_list), mode))
        edge_cost.append(edge_cost[-1] + config_cost(edge[-2].q, edge[-1].q, "euclidean"))

    return edge, edge_cost

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
                        size=5,  # Very small markers
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

def shortcutting_mode(dir, env, env_path, pkl_folder, config,  output_html, version = 0):
    """Shortcutting per mode meaning cannot connect two nodes that aren't assigned to the same mode"""

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
        transition[0] = True
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
        overall_costs = [total_cost]
        time_list = [0]
        start = time.time()
        discretized_path, discretized_modes, discretized_costs, discretized_transition =init_discretization(path_, intermediate_tot, modes, indices, transition)  
        original_discretized_path = discretized_path.copy()     
        original_discretized_modes = discretized_modes.copy()      
        modes_legend = [mode for idx, mode in enumerate(original_discretized_modes) if discretized_transition[idx]]
        all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
        dim = None
        all_indices = [i for i in range(len(discretized_path[0].q.state()))]
        for _ in range(100000):
            i1 = np.random.choice(len(discretized_path))
            i2 = np.random.choice(len(discretized_path))
            if version == 1:
                dim = np.random.choice(range(len(discretized_path[0].q.state())))
                # dim = np.random.choice(range(env.robot_dims[env.robots[0]]))# TODO only feasible for same dimension across all robots
            if version == 2:
                num_indices = np.random.choice(range(len(discretized_path[0].q.state())))
                random.shuffle(all_indices)
                dim = all_indices[:num_indices]
            if np.abs(i1-i2) < 2:
                continue
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
            edge, edge_cost =  interpolate(discretized_path[idx1:idx2],  
                                                           discretized_costs[idx1], discretized_modes, indices, dim, version)
            c_new = edge_cost[-1] + config_cost(edge[-1].q, state2.q, "euclidean") - edge_cost[0]
            if c_new < (discretized_costs[idx2]- discretized_costs[idx1]) and env.is_edge_collision_free(state1.q, state2.q, m1):
                discretized_path[idx1:idx2] = edge
                discretized_costs[idx1:idx2] = edge_cost
                update(discretized_path, discretized_costs, idx2)
                all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
                overall_costs.append(discretized_costs[-1])
                diff = time.time()- start
                time_list.append(diff)
        path_visualization(all_frame_traces, env_path,modes_legend,original_discretized_path, discretized_transition, output_html)
        # path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(discretized_path)}
        # json_dir =os.path.join(dir, 'general.log') 
        # with open(json_dir, 'a') as log_file:
        #     log_file.write(f"Path shortcut: {json.dumps(path_dict, indent=4)}\n")
        #     log_file.write(f"Before: {json.dumps(total_cost.item(), indent=4)}\n")
        #     log_file.write(f"After: {json.dumps(discretized_costs[-1].item(), indent=4)}\n")
        print("Before: ", total_cost.item(), "After " ,   discretized_costs[-1].item())
        frames_directory = os.path.join(config["output_dir"], f'Post_{version}')
        os.makedirs(frames_directory, exist_ok=True)
        data = {
            "total_cost": overall_costs, 
            "time": time_list,
            "path": discretized_path }
        # Find next available file number
        existing_files = (int(file.split('.')[0]) for file in os.listdir(frames_directory) if file.endswith('.pkl') and file.split('.')[0].isdigit())
        next_file_number = max(existing_files, default=-1) + 1

        # Save data as pickle file
        filename = os.path.join(frames_directory, f"{next_file_number:04d}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def shortcutting_agent(dir, env, env_path, pkl_folder, config,  output_html, version = 0, choice = 0, deterministic = False):

    pkl_files = sorted(
        [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    with open(pkl_files[-1], 'rb') as file:
        data = dill.load(file)
        results = data["result"]
        path_ = results["path"]
        total_cost = results["total"]
        intermediate_tot = results["intermediate_tot"]
        transition = results["is_transition"]
        transition[0] = True
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
        overall_costs = [total_cost]
        time_list = [0]
        start = time.time()
        discretized_path, discretized_modes, discretized_costs, discretized_transition =init_discretization(path_, intermediate_tot, modes, indices, transition)  
        termination_cost = discretized_costs[-1]
        termination_iter = 0
        original_discretized_path = discretized_path.copy()     
        original_discretized_modes = discretized_modes.copy()      
        modes_legend = [mode for idx, mode in enumerate(original_discretized_modes) if discretized_transition[idx]]
        all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
        dim = None

        if not deterministic:
            range1 = 1
            range2 = 100000
        else:
            range1 = len(discretized_path)
            range2 = range1

        for i in range(range1):
            for j in range(range2):
                if not deterministic:
                    i1 = np.random.choice(len(discretized_path))
                    i2 = np.random.choice(len(discretized_path))
                else:
                    i1 = i
                    i2 = j
                    # dim = np.random.choice(range(env.robot_dims[env.robots[0]]))# TODO only feasible for same dimension across all robots
                if np.abs(i1-i2) < 2:
                    continue
                idx1 = min(i1, i2)
                idx2 = max(i1, i2)
                m1 = discretized_modes[idx1]    
                m2 = discretized_modes[idx2]    
                if m1 == m2 and choice == 0: #take all possible robots
                    robot = None
                    if version == 4:
                        dim = [np.random.choice(indices[r_idx]) for r_idx in range(len(env.robots))]
                    
                    if version == 5:
                        dim = []
                        for r_idx in range(len(env.robots)):
                            all_indices = [i for i in indices[r_idx]]
                            num_indices = np.random.choice(range(len(indices[r_idx])))
                            random.shuffle(all_indices)
                            dim.append(all_indices[:num_indices])
                else:
                    robot = np.random.choice(len(env.robots))
                    if len(m1) > 1:
                        task_agent = m1[1][robot]
                    else:
                        task_agent = m1[0][robot]

                    if m2[0][robot] != task_agent:
                        continue

                    if version == 4:
                        dim = [np.random.choice(indices[robot])]
                    if version == 5:
                        all_indices = [i for i in indices[robot]]
                        num_indices = np.random.choice(range(len(indices[robot])))

                        random.shuffle(all_indices)
                        dim = all_indices[:num_indices]

                state1 = discretized_path[idx1]
                state2 = discretized_path[idx2]
                edge, edge_cost =  interpolate(discretized_path[idx1:idx2], 
                                                            discretized_costs[idx1], discretized_modes, indices, dim, version, robot, env.robots)
                c_new = edge_cost[-1] + config_cost(edge[-1].q, state2.q, "euclidean") - edge_cost[0]
                if c_new < (discretized_costs[idx2]- discretized_costs[idx1]) and env.is_edge_collision_free(state1.q, state2.q, m1): #what when two different modes??? (possible for one task) 
                    discretized_path[idx1:idx2] = edge
                    discretized_costs[idx1:idx2] = edge_cost
                    # discretized_modes[idx1:idx2] = edge_modes
                    update(discretized_path, discretized_costs, idx2)
                    all_frame_traces.append(path_traces(colors, mode_sequence, discretized_path))
                    overall_costs.append(discretized_costs[-1])
                    time_list.append(time.time()- start)
                    if not deterministic:
                        if np.abs(discretized_costs[-1] - termination_cost) > 0.001:
                            termination_cost = discretized_costs[-1]
                            termination_iter = j
                        elif np.abs(termination_iter -j) > 25000:
                            break
                    
        path_visualization(all_frame_traces, env_path,modes_legend,original_discretized_path, discretized_transition, output_html)
        # path_dict = {f"{i}": state.q.state().tolist() for i, state in enumerate(discretized_path)}
        # json_dir =os.path.join(dir, 'general.log') 
        # with open(json_dir, 'a') as log_file:
        #     log_file.write(f"Path shortcut: {json.dumps(path_dict, indent=4)}\n")
        #     log_file.write(f"Before: {json.dumps(total_cost.item(), indent=4)}\n")
        #     log_file.write(f"After: {json.dumps(discretized_costs[-1].item(), indent=4)}\n")
        print("Before: ", total_cost.item(), "After " ,   discretized_costs[-1].item())
        if not deterministic:
            frames_directory = os.path.join(config["output_dir"], f'Post_{version+3*choice}')
        else:
            frames_directory = os.path.join(config["output_dir"], f'Post_{version+6+(choice*3)}')
        os.makedirs(frames_directory, exist_ok=True)
        data = {
            "total_cost": overall_costs, 
            "time": time_list,
            "path": discretized_path }
        # Find next available file number
        existing_files = (int(file.split('.')[0]) for file in os.listdir(frames_directory) if file.endswith('.pkl') and file.split('.')[0].isdigit())
        next_file_number = max(existing_files, default=-1) + 1

        # Save data as pickle file
        filename = os.path.join(frames_directory, f"{next_file_number:04d}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Env shower")
    group = parser.add_mutually_exclusive_group(required=True)

    # Nodes need to be in same mode (Post_{idx} with idx: 0,1,2)
    group.add_argument(
        "--mode",
        choices=["shortcutting", "partial_singledim", "partial_random"],
        help="Select the mode of operation for the environment",
    )
    # Nodes need to be at least for one agent in the same task execution 
    # (Post_{idx} with idx: 3-11, 3-5: choice=0, 6-8: choice=1, 9-11: choice=0/deterministic=True, 12-14: choice=1/deterministic=True)
    group.add_argument(
        "--agent",
        choices=["shortcutting", "partial_singledim", "partial_random"],
        help="Select the agent's behavior mode",
    )

    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    directory = os.path.join(home_dir, 'output')
    path = get_latest_folder(directory)
    # path = "/home/tirza/output/091224_083733"
    # path = '/home/tirza/output/091224_083733'
    # path = '/home/tirza/output/151224_105746'
    # path = '/home/tirza/output/151224_101608'
    env_name, config_params, _, _ = get_config(path)
    env = get_env_by_name(env_name)    
    pkl_folder = os.path.join(path, 'FramesData')
    env_path = os.path.join(home_dir, f'env/{env_name}')
    save_env_as_mesh(env, env_path)
    choice = 1 # 0: all possible agents, 1: only one agent at a time
    iteration = 1
    deterministic = False
    if choice == 0:
        choice_name = 'all'
    else:
        choice_name = 'one'
    if not deterministic:
        name = 'prob'
    else:
        name = 'deterministic'
    



    #Each creates a folder with name Post_{idx} with idx as the corresponding number
    if args.mode == "shortcutting":
        for _ in range(iteration):
            output_html = os.path.join(path, 'shortcutting_mode.html')
            shortcutting_mode(path, env, env_path, pkl_folder, config_params, output_html, 0)
    if args.mode == "partial_singledim": 
        for _ in range(iteration):
            output_html = os.path.join(path, 'partial_shortcutting_singledim_mode.html')
            shortcutting_mode(path, env, env_path, pkl_folder, config_params, output_html, 1)
    if args.mode == "partial_random": 
        for _ in range(iteration):
            output_html = os.path.join(path, 'partial_shortcutting_random_mode.html')
            shortcutting_mode(path, env, env_path, pkl_folder, config_params, output_html, 2)
    if args.agent == "shortcutting":
        for _ in range(iteration):
            output_html = os.path.join(path, f'shortcutting_agent_{choice_name}_{name}.html')
            shortcutting_agent(path, env, env_path, pkl_folder, config_params, output_html, 3, choice, deterministic)
    if args.agent == "partial_singledim": 
        for _ in range(iteration):
            output_html = os.path.join(path, f'partial_shortcutting_singledim_agent_{choice_name}_{name}.html')
            shortcutting_agent(path, env, env_path, pkl_folder, config_params, output_html, 4, choice, deterministic )
    if args.agent == "partial_random": 
        for _ in range(iteration):
            output_html = os.path.join(path, f'partial_shortcutting_random_agent_{choice_name}_{name}.html')
            shortcutting_agent(path, env, env_path, pkl_folder, config_params, output_html, 5, choice, deterministic)

    # webbrowser.open('file://' + os.path.realpath(output_html))



