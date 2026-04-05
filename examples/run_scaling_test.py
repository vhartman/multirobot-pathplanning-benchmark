def run_stacking_scaling_env():
    for _ in num_robots:
        for _ in num_tasks:
            env = load_env()
            run_exp(env, num_runs)

def main():
    # run_mobile_scaling_env()
    run_stacking_scaling_env()