import json
import subprocess
import tempfile
import datetime
import os

representative_envs = [
    ("rai.simple", 2),
    ("rai.one_agent_many_goals", 2),
    ("rai.two_agents_many_goals_dep", 2),
    ("rai.three_agent_many_goals", 2),
    ("rai.single_agent_mover", 2),
    ("rai.piano_dep", 2),
    ("rai.2d_handover", 2),
    ("rai.random_2d", 2),
    ("rai.other_hallway", 2),
    ("rai.three_agents", 2),
    ("rai.triple_waypoints", 2),
    ("rai.welding", 2),
    ("rai.handover", 2),
    ("rai.eggs", 2),
    ("rai.bottles", 2),
    ("rai.box_rearrangement", 2),
    ("rai.box_reorientation", 2),
    ("rai.box_reorientation_dep", 2),
    ("rai.pyramid", 2),
    ("rai.box_stacking", 2),
    ("rai.box_stacking_dep", 2),
    ("rai.mobile_wall_four", 2),
    ("rai.mobile_wall_four_dep", 2),
    ("rai.mobile_strut", 7),
    ("rai.three_robot_truss", 7),
    ("rai.spiral_tower", 0),
    ("rai.spiral_tower_two", 0),
    ("rai.cube_four", 2),
    ("rai.unordered_box_reorientation", 0),
    ("rai.unassigned_two_dim", 0),
    ("rai.unassigned_cleanup", 0),
    ("rai.unassigned_stacking", 0),
    ("rai.unordered_bottles", 0),
]

num_parallel = 2
num_runs = 1
max_runtime = 500
optimize = False

planners = [
    {"name": "prio", "type": "prioritized", "options": {}},
    {"name": "birrt", "type": "birrtstar", "options": {}},
]

for env_name, seed in representative_envs:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running {env_name} with seed {seed}")

    # create temporary JSON config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
        config = {
            "experiment_name": "convergence",
            "environment": env_name,
            "per_agent_cost": "euclidean",
            "cost_reduction": "max",
            "max_planning_time": max_runtime,
            "num_runs": num_runs,
            "seed": seed,
            "optimize": optimize,
            "planners": planners,
        }
        json.dump(config, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        # spawn a fresh Python process for strict isolation
        subprocess.run(
            [
                "python",
                "./examples/run_experiment.py",
                tmpfile_path,
                "--num_processes",
                str(num_parallel),
                "--parallel_execution",
            ],
            check=True,
        )
    finally:
        # clean up temporary file
        os.remove(tmpfile_path)
