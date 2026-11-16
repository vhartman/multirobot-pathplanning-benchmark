"""Goal preparation for frozen RoboBallet instances; not used during planning."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def solve_ik(env, robot, target, seed, max_steps=100, return_best=False):
    """Damped Jacobian IK; collision screening is separate from this local solve."""
    indices = env.robot_idx[env.robots[robot]]
    x = env.start_pos.state().copy()
    lo, hi = env.limits[:, indices]
    q = np.clip(np.asarray(seed), lo + 1e-5, hi - 1e-5)
    desired_p, desired_R = np.asarray(target["position"]), np.asarray(
        target["rotation"]
    )
    site = env.tcp[robot]
    jp, jr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    best, best_error = q.copy(), np.inf
    for _ in range(max_steps):
        x[indices] = q
        env.set_configuration(x, collision=False)
        error_p = desired_p - env.data.site_xpos[site]
        error_r = Rotation.from_matrix(
            desired_R @ env.data.site_xmat[site].reshape(3, 3).T
        ).as_rotvec()
        error = np.linalg.norm(error_p) ** 2 + 0.09 * np.linalg.norm(error_r) ** 2
        if error < best_error:
            best, best_error = q.copy(), error
        if np.linalg.norm(error_p) < 3e-4 and np.linalg.norm(error_r) < 2e-3:
            return q.copy()
        mujoco.mj_jacSite(env.model, env.data, jp, jr, site)
        J = np.vstack([jp[:, env.dofadr[indices]], 0.3 * jr[:, env.dofadr[indices]]])
        e = np.r_[error_p, 0.3 * error_r]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.002**2 * np.eye(6), e)
        dq *= min(1.0, 0.25 / (np.max(np.abs(dq)) + 1e-12))
        q = np.clip(q + dq, lo + 1e-5, hi - 1e-5)
    return best if return_best else None


def refine_goal_region(env, robot, target, q):
    """Find a nearby collision-free IK inside RoboBallet's Cartesian tolerance.

    This is scene preparation only. The planner later receives the resulting
    fixed joint goal, and its collision checker is never relaxed.
    """
    from scipy.optimize import minimize

    indices = env.robot_idx[env.robots[robot]]
    p, R = np.asarray(target["position"]), np.asarray(target["rotation"])

    def constraints(candidate):
        full = env.start_pos.state().copy()
        full[indices] = candidate
        env.set_configuration(full)
        dp = np.linalg.norm(env.data.site_xpos[env.tcp[robot]] - p)
        dr = Rotation.from_matrix(
            R @ env.data.site_xmat[env.tcp[robot]].reshape(3, 3).T
        ).magnitude()
        minimum = min((c[2] for c in env.colliding_pairs()), default=0.001)
        return np.array([0.025 - 1e-5 - dp, np.deg2rad(15) - 1e-4 - dr, minimum - 1e-6])

    result = minimize(
        lambda x: np.sum((x - q) ** 2),
        q,
        method="SLSQP",
        bounds=list(zip(*env.limits[:, indices])),
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": 150, "ftol": 1e-9},
    )
    if np.all(constraints(result.x) >= 0):
        return result.x.copy()
    return None


def solve_bounded_ik(env, robot, target, seed, rotation_weight=0.1):
    """Bounded least-squares seed for difficult goal-region refinement."""
    from scipy.optimize import least_squares

    idx = env.robot_idx[env.robots[robot]]
    full = env.start_pos.state().copy()
    R, p = np.asarray(target["rotation"]), np.asarray(target["position"])

    def error(q):
        full[idx] = q
        env.set_configuration(full, collision=False)
        return np.r_[
            env.data.site_xpos[env.tcp[robot]] - p,
            rotation_weight
            * Rotation.from_matrix(
                R @ env.data.site_xmat[env.tcp[robot]].reshape(3, 3).T
            ).as_rotvec(),
        ]

    lo, hi = env.limits[:, idx]
    result = least_squares(
        error, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=150
    )
    return result.x
