import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path to import skills
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from multi_robot_multi_goal_planning.problems.skills import DummyStochasticSkill

# NOTE: script generated with Gemini (AI)

def main():
    # Scenario setup
    q_start = np.array([0.0, 0.0])
    q_goal = np.array([10.0, 10.0])
    
    # Params
    dt = 0.01
    duration = 2.0
    noise_bound = 1.0 # Fixed noise bound as requested
    
    # Initialize the stochastic skill
    joints = ["joint_x", "joint_y"]
    skill = DummyStochasticSkill(
        joints=joints, 
        goal_state=q_goal, 
        dt=dt, 
        noise_bound=noise_bound, 
        is_deterministic=False, 
        duration=duration
    )
    
    num_rollouts = 500
    all_trajectories = []
    
    n_steps = max(1, round(duration / dt))
    
    print(f"Rolling out {num_rollouts} trajectories (duration={duration}s, dt={dt}s)...")
    
    # Generate rollouts
    for _ in range(num_rollouts):
        q = q_start.copy()
        traj = [q.copy()]
        
        for i in range(n_steps):
            t_norm = (i + 1) / n_steps
            q = skill.step(t_norm, q, env=None) # Dummy skill doesn't use env
            traj.append(q.copy())
            
        all_trajectories.append(np.array(traj))
        
    all_trajectories = np.array(all_trajectories)
    
    # Calculate statistics
    mean_traj = np.mean(all_trajectories, axis=0)
    std_traj = np.std(all_trajectories, axis=0)
    
    # Compute the 95% confidence interval (approx 2 standard deviations)
    # The magnitude of standard deviation radially
    radial_std = np.linalg.norm(std_traj, axis=1)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.title(f"Stochastic Skill Rollouts (N={num_rollouts}, noise_bound={noise_bound})")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Plot all individual trajectories
    for traj in all_trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1, linewidth=1)
        
    # Plot the mean trajectory
    plt.plot(mean_traj[:, 0], mean_traj[:, 1], color='blue', linewidth=2, label="Mean Path")
    
    # Plot start and goal
    plt.scatter(q_start[0], q_start[1], color='green', s=100, label="Start", zorder=5)
    plt.scatter(q_goal[0], q_goal[1], color='red', s=100, label="Goal", zorder=5)
    
    # Draw circles to represent the 2-sigma variance envelope at regular intervals
    for i in range(0, len(mean_traj), len(mean_traj)//10):
        circle = plt.Circle((mean_traj[i, 0], mean_traj[i, 1]), 2 * radial_std[i], 
                            color='blue', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        
    # Add a dummy handle for the legend for the variance circles
    circle_handle = plt.Line2D([0], [0], color='blue', linestyle='--', label='2-Sigma Envelope')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(circle_handle)
    plt.legend(handles=handles)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    
    save_path = os.path.join(os.path.dirname(__file__), "stochastic_rollouts_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
