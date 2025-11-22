import json
import subprocess
import tempfile
import datetime
import os

representative_envs = [
    ("rai.simple", 2, 100),
    ("rai.one_agent_many_goals", 2, 100),
    ("rai.two_agents_many_goals_dep", 2, 100),
    ("rai.three_agent_many_goals", 2, 500),
    ("rai.single_agent_mover", 2, 100),
    ("rai.piano_dep", 2, 500),
    ("rai.2d_handover", 2, 500),
    ("rai.random_2d", 2, 500),
    ("rai.other_hallway", 2, 500),
    ("rai.three_agents", 2, 500),
    ("rai.triple_waypoints", 2, 1000),
    ("rai.welding", 2, 1000),
    ("rai.handover", 2, 100),
    ("rai.eggs", 2, 500),
    ("rai.bottles", 2, 500),
    ("rai.box_rearrangement", 2, 1000),
    ("rai.box_reorientation", 2, 500),
    ("rai.box_reorientation_dep", 2, 1000),
    ("rai.pyramid", 2, 500),
    ("rai.box_stacking", 2, 1000),
    ("rai.box_stacking_dep", 2, 1000),
    ("rai.mobile_wall_four", 2, 1000),
    ("rai.mobile_wall_four_dep", 2, 1000),
    ("rai.mobile_strut", 7, 1000),
    ("rai.three_robot_truss", 7, 1000),
    ("rai.spiral_tower", 0, 1000),
    ("rai.spiral_tower_two", 0, 500),
    ("rai.cube_four", 2, 1000),
    ("rai.unordered_box_reorientation", 0, 1000),
    ("rai.unassigned_two_dim", 0, 500),
    ("rai.unassigned_cleanup", 0, 1000),
    ("rai.unassigned_stacking", 0, 1000),
    ("rai.unordered_bottles", 0, 500),
]

num_parallel = 10
num_runs = 25
optimize = True

planners = [
    {"name": "prio", "type": "prioritized", "options": {}},
    {"name": "birrt", "type": "birrtstar", "options": {}},
    {"name": "ait", "type": "aitstar", "options": {}},
]

for env_name, seed, runtime in representative_envs:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running {env_name} with seed {seed}")

    # create temporary JSON config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmpfile:
        config = {
            "experiment_name": "convergence",
            "environment": env_name,
            "per_agent_cost": "euclidean",
            "cost_reduction": "max",
            "max_planning_time": runtime,
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
