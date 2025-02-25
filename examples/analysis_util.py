import os
import json
from plyfile import PlyData, PlyElement
import numpy as np
import plotly.graph_objects as go

def colors_plotly():
    return [
    'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 
    'brown', 'lime', 'olive', 'navy', 'teal', 'coral', 'turquoise', 'gold', 'violet',
    'indigo', 'salmon', 'maroon', 'plum', 'orchid', 'tan', 'crimson', 'lavender',
    'chartreuse', 'azure', 'beige', 'khaki', 'slateblue', 'forestgreen', 'darkorange',
    'steelblue', 'deepskyblue', 'tomato', 'lightcoral', 'wheat', 'lightseagreen',
    'mediumslateblue', 'peru', 'darkviolet', 'firebrick', 'dodgerblue', 'mediumorchid',
    'mistyrose', 'sienna', 'darkkhaki', 'peachpuff', 'lightgoldenrodyellow', 'skyblue', 'palegreen',
    'rosybrown', 'midnightblue', 'lightpink', 'lavenderblush', 'darkgoldenrod', 'sandybrown',
    'slategray', 'mediumturquoise', 'orchid', 'seashell', 'mediumseagreen', 'royalblue',
    'darkseagreen', 'cornflowerblue', 'burlywood', 'mediumaquamarine', 'hotpink', 'palevioletred',
    'darkslategray', 'darksalmon', 'lightsteelblue', 'cadetblue', 'mediumvioletred', 'mediumspringgreen',
    'darkturquoise', 'lightsalmon', 'darkred', 'goldenrod', 'darkgreen', 'lightcyan', 'springgreen',
    'powderblue', 'lightblue', 'slategrey', 'darkorchid', 'silver', 'fuchsia', 'lightgreen', 
    'papayawhip', 'blanchedalmond', 'antiquewhite', 'bisque', 'honeydew', 'gainsboro', 'thistle',
    'peach', 'mintcream', 'aliceblue', 'ghostwhite'
]

def colors_ry():
    return [
        [0, 0, 1],  # blue
        [0, 0.5, 0],  # green
        [0.5, 0, 0.5],  # purple
        [1, 0.65, 0],  # orange
        [0, 1, 1],  # cyan
        [1, 0, 1],  # magenta
        [1, 0.75, 0.8],  # pink
        [0.6, 0.4, 0.2],  # brown
        [0, 1, 0],  # lime
        [0.5, 0.5, 0],  # olive
        [0, 0, 0.5],  # navy
        [0, 0.5, 0.5],  # teal
        [1, 0.5, 0.31],  # coral
        [0.25, 0.88, 0.82],  # turquoise
        [1, 0.84, 0],  # gold
        [0.93, 0.51, 0.93],  # violet
        [0.29, 0, 0.51],  # indigo
        [0.98, 0.5, 0.45],  # salmon
        [0.5, 0, 0],  # maroon
        [0.87, 0.63, 0.87],  # plum
        [0.85, 0.44, 0.84],  # orchid
        [0.82, 0.71, 0.55],  # tan
        [0.86, 0.08, 0.24],  # crimson
        [0.9, 0.9, 0.98],  # lavender
        [0.5, 1, 0],  # chartreuse
        [0, 0.5, 1],  # azure
        [0.96, 0.96, 0.86],  # beige
        [0.94, 0.9, 0.55],  # khaki
        [0.42, 0.35, 0.8],  # slateblue
        [0.13, 0.55, 0.13],  # forestgreen
        [1, 0.55, 0],  # darkorange
        [0.27, 0.51, 0.71],  # steelblue
        [0, 0.75, 1],  # deepskyblue
        [1, 0.39, 0.28],  # tomato
        [0.94, 0.5, 0.5],  # lightcoral
        [0.96, 0.87, 0.7],  # wheat
        [0.13, 0.7, 0.67],  # lightseagreen
        [0.48, 0.41, 0.93],  # mediumslateblue
        [0.8, 0.52, 0.25],  # peru
        [0.58, 0, 0.83],  # darkviolet
        [0.7, 0.13, 0.13],  # firebrick
        [0.12, 0.56, 1],  # dodgerblue
        [0.73, 0.33, 0.83],  # mediumorchid
        [1, 0.89, 0.88],  # mistyrose
        [0.63, 0.32, 0.18],  # sienna
        [0.74, 0.72, 0.42],  # darkkhaki
        [1, 0.85, 0.73],  # peachpuff
        [0.98, 0.98, 0.82],  # lightgoldenrodyellow
        [0.53, 0.81, 0.98],  # skyblue
        [0.6, 0.98, 0.6],  # palegreen
        [0.74, 0.56, 0.56],  # rosybrown
        [0.1, 0.1, 0.44],  # midnightblue
        [1, 0.71, 0.76],  # lightpink
        [1, 0.94, 0.96],  # lavenderblush
        [0.72, 0.53, 0.04],  # darkgoldenrod
        [0.96, 0.64, 0.38],  # sandybrown
        [0.44, 0.5, 0.56],  # slategray
        [0.28, 0.82, 0.8],  # mediumturquoise
        [0.85, 0.44, 0.84],  # orchid (duplicate)
        [1, 0.96, 0.93],  # seashell
        [0.24, 0.7, 0.44],  # mediumseagreen
        [0.25, 0.41, 0.88],  # royalblue
        [0.56, 0.74, 0.56],  # darkseagreen
        [0.39, 0.58, 0.93],  # cornflowerblue
        [0.87, 0.72, 0.53],  # burlywood
        [0.4, 0.8, 0.67],  # mediumaquamarine
        [1, 0.41, 0.71],  # hotpink
        [0.86, 0.44, 0.58],  # palevioletred
        [0.18, 0.31, 0.31],  # darkslategray
        [0.91, 0.59, 0.48],  # darksalmon
        [0.69, 0.77, 0.87],  # lightsteelblue
        [0.37, 0.62, 0.63],  # cadetblue
        [0.78, 0.08, 0.52],  # mediumvioletred
        [0, 0.98, 0.6],  # mediumspringgreen
        [0, 0.81, 0.82],  # darkturquoise
        [1, 0.63, 0.48],  # lightsalmon
        [0.55, 0, 0],  # darkred
        [0.85, 0.65, 0.13],  # goldenrod
        [0, 0.39, 0],  # darkgreen
        [0.88, 1, 1],  # lightcyan
        [0, 1, 0.5],  # springgreen
        [0.69, 0.88, 0.9],  # powderblue
        [0.68, 0.85, 0.9],  # lightblue
        [0.44, 0.5, 0.56],  # slategrey (duplicate)
        [0.6, 0.2, 0.8],  # darkorchid
        [0.75, 0.75, 0.75],  # silver
        [1, 0, 1],  # fuchsia
        [0.56, 0.93, 0.56],  # lightgreen
        [1, 0.94, 0.84],  # papayawhip
        [1, 0.92, 0.8],  # blanchedalmond
        [0.98, 0.92, 0.84],  # antiquewhite
        [1, 0.89, 0.77],  # bisque
        [0.94, 1, 0.94],  # honeydew
        [0.86, 0.86, 0.86],  # gainsboro
        [0.85, 0.75, 0.85],  # thistle
        [1, 0.85, 0.73],  # peach
        [0.96, 1, 0.98],  # mintcream
        [0.94, 0.97, 1],  # aliceblue
        [0.97, 0.97, 1],  # ghostwhite
    ]

def get_latest_folder(directory):
    entries = [os.path.join(directory, d) for d in os.listdir(directory)]
    folders = [d for d in entries if os.path.isdir(d)]
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

def get_config(path):
    file_path = os.path.join(path, 'general.log')
    env = None
    path_data = []
    short_path_data = []
    current_list = []
    config_params = {}

    with open(file_path, "r") as log_file:
        lines = log_file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Extract environment
            if line.startswith("Environment:"):
                env = line.split(":")[1].strip().strip('"')
                i+=1
                continue

            if line.startswith("Configuration Parameters:"):
                config_lines = []
                # Collect all lines for the configuration block
                while i < len(lines) and not lines[i].strip().endswith("}"):
                    config_lines.append(lines[i].strip())
                    i += 1
                config_lines.append(lines[i].strip())  # Append the closing brace

                # Parse the configuration parameters
                config_string = " ".join(config_lines).split(":", 1)[1].strip()
                try:
                    config_params = json.loads(config_string)
                except json.JSONDecodeError as e:
                    print("Error parsing Configuration Parameters:", e)
                    print("Configuration content:", config_string)

            # Detect the start of the Path section
            if line.startswith("Path:"):
                i+=1
                while i < len(lines) and not lines[i].strip().endswith("}"):
                    line = lines[i].strip()
                    # Skip keys like `"0": [`
                    if line.startswith('"') and line.endswith(": ["):
                        i+=1
                        continue

                    # Detect the end of a list
                    if line == "],":
                        path_data.append(current_list)
                        current_list = []
                        i+=1
                        continue

                    # Detect the end of the Path section
                    if line == "}":
                        if current_list:  # Append the last list if not empty
                            path_data.append(current_list)
                        break

                    # Otherwise, collect valid numerical values
                    try:
                        current_list.append(float(line.strip(",").strip("]")))
                    except ValueError:
                        i+=1
                        continue  # Skip non-numeric lines or empty lines
                    i+=1
            current_list = []
            if line.startswith("Path shortcut:"):
                short_path_data = []
                i+=1
                while i < len(lines) and not lines[i].strip().endswith("}"):
                    line = lines[i].strip()
                    # Skip keys like `"0": [`
                    if line.startswith('"') and line.endswith(": ["):
                        i+=1
                        continue

                    # Detect the end of a list
                    if line == "],":
                        short_path_data.append(current_list)
                        current_list = []
                        i+=1
                        continue

                    # Detect the end of the Path section
                    if line == "}":
                        if current_list:  # Append the last list if not empty
                            short_path_data.append(current_list)
                        break

                    # Otherwise, collect valid numerical values
                    try:
                        current_list.append(float(line.strip(",").strip("]")))
                    except ValueError:
                        i+=1
                        continue  # Skip non-numeric lines or empty lines
                    i+=1                        
            i+=1
    return env, config_params, path_data, short_path_data

def get_absolute_transforms(env):
    """
    Get the absolute transformations (positions) of all frames in an environment.

    :param env: The ry.Config object containing the environment
    :return: A dictionary with frame names as keys and absolute positions as values
    """
    frame_state = env.C.getFrameState()  # Nx7 matrix: [x, y, z, qx, qy, qz, qw]
    frame_names = env.C.getFrameNames()
    
    absolute_positions = {}
    for name, state in zip(frame_names, frame_state):
        position = state[:3]  # Extract [x, y, z]
        absolute_positions[name] = position
    
    return absolute_positions

def save_env_as_mesh(env, directory):
    if os.path.exists(directory):
        return
    os.makedirs(directory, exist_ok=True)

    # Check the environment before exporting
    frame_names = env.C.getFrameNames()
    absolute_positions = get_absolute_transforms(env)

    for frame_name in frame_names:
        full_dir = os.path.join(directory, f'{frame_name}.ply')

        try:
            # Access the frame and get its mesh data
            frame = env.C.frame(frame_name)
            mesh = frame.getMesh()
            vertices, faces, color = mesh

            # Adjust vertices to absolute position
            absolute_position = absolute_positions[frame_name]
            adjusted_vertices = vertices + absolute_position  # Apply translation

            # Prepare vertex data with color
            vertex_data = [
                (v[0], v[1], v[2], color[0], color[1], color[2])
                for v in adjusted_vertices
            ]
            vertex_array = np.array(
                vertex_data,
                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            )

            # Prepare face data
            face_data = [(face,) for face in faces]
            face_array = np.array(face_data, dtype=[('vertex_indices', 'i4', (3,))])

            # Create PlyElements
            vertex_element = PlyElement.describe(vertex_array, 'vertex')
            face_element = PlyElement.describe(face_array, 'face')

            # Write to .ply file
            PlyData([vertex_element, face_element], text=True).write(full_dir)
            print(f"Saved {frame_name} to {full_dir}.")

        except Exception as e:
            print(f"Error processing frame '{frame_name}': {e}")
            continue

def load_ply_file(filepath):
    """
    Load vertices, faces, and colors from a .ply file.

    :param filepath: Path to the .ply file
    :return: vertices (np.ndarray), faces (np.ndarray), colors (np.ndarray)
    """
    ply_data = PlyData.read(filepath)

    # Extract vertex positions
    vertices = np.array([(v['x'], v['y'], v['z']) for v in ply_data['vertex']])

    # Extract vertex colors
    colors = np.array([(v['red'], v['green'], v['blue']) for v in ply_data['vertex']])

    # Extract faces
    faces = np.array([f['vertex_indices'] for f in ply_data['face']])

    return vertices, faces, colors

def mesh_traces_env(folder_path):
    """
    Load and plot all .ply files in a folder with distinct colors and no transparency.

    :param folder_path: Path to the folder containing .ply files
    """
    meshes = []
    color_map = ['grey']  # Distinct colors for each mesh
    color_idx = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.ply'):
            if os.path.splitext(os.path.basename(filename))[0][0] == "a":
                continue
            filepath = os.path.join(folder_path, filename)
            vertices, faces, colors = load_ply_file(filepath)

            # Extract x, y, z and faces
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            # Assign distinct color
            color = color_map[color_idx % len(color_map)]
            color_idx += 1

            # Append the mesh
            meshes.append(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.4,  # Fully opaque
                color=color,
                name=filename
            ))

    # Create and show the plot
    return meshes

def count_files_in_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Found {len(files)} files")
    return len(files)
