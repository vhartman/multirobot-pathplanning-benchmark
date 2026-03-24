# TEST WITH SPECIFIC PLANNERS (demos)
### Run prioritized planner on 2D hallway
python3 examples/run_planner.py rai.2d_hallway --planner=prioritized --max_time=30 --optimize

### Run composite PRM on box stacking
python3 examples/run_planner.py rai.box_stacking --planner=composite_prm --max_time=30 --optimize

### Run RRT* on mobile wall navigation
python3 examples/run_planner.py rai.mobile_wall --planner=rrt_star --max_time=30 --optimize


# RUN AN EXPERIMENT
python3 examples/run_experiment.py configs/demo/mobile_wall.json


# PYTESTS
pytest tests/test_planners.py -v
pytest tests/test_envs.py -v


# VISUALIZE AVAILABLE ENVIRONMENTS
python3 examples/show_problems.py --mode list_all
python3 examples/show_problems.py rai.box_stacking --mode modes


# VIDEO
python3 examples/run_planner.py rai.single_agent_screw --planner=prioritized --max_time=30 --optimize --save

python3 examples/display_single_path.py ./out/20260220_224545_rai.single_agent_screw/prioritized/0/path_0.json rai.single_agent_screw --export

cd z.vid

ffmpeg -framerate 30 -i %04d.png -c:v libx264 -pix_fmt yuv420p out.mp4


# PLOTS
Create an experiment config file in configs/experiments-william/example.json

Run the experiment (useful flags are: --parallel_execution --num_proesses 4)
python3 ./examples/run_experiment.py configs/experiments-william/example.json  

Make plots (useful flags are: --png --use_paper_style --save)
python3 examples/make_plots.py out/timestamp_comparison_rai.environment/

# PROFILE
Run the speedscope:
py-spy record -F -r 500 -o profile.svg --nonblocking -f speedscope -- python3  examples/run_planner.py rai.skill_box_stacking_two_robots --distance_metric=max_euclidean --max_time=100 --seed=0 --planner composite_prm --cost_reduction=max --save