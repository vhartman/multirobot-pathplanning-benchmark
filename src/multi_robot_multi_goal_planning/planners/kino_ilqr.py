"""
iLQR-based shortcutting for kinodynamic problems WITHOUT an exact steering function
(e.g. the second-order car).

Shortcutting fundamentally needs a two-point boundary-value solver: connect two path states
exactly under the dynamics so the reconnection splices back in seamlessly. Systems with a
closed-form steering function get this for free; the car does not. iLQR supplies it
numerically -- it optimises a control sequence driving x0 -> x1 under the real dynamics,
minimising control effort with a heavy terminal cost on reaching x1.

Everything is env-agnostic: dynamics come from env.forward_dynamics, Jacobians are taken by
finite differences (no analytic derivatives), and the only env-specific knowledge is
env.angular_dims (state indices that are angles), used to wrap angle errors.

Two passes, mirroring the exact-steering shortcuts:
  * kinodynamic_ilqr_shortcut       -- composite, same-mode, reconnect over a SHORTER horizon
                                       (removes stop-and-go / compresses time).
  * kinodynamic_ilqr_robot_shortcut -- per-robot, FIXED horizon, may cross modes while that
                                       robot's task is unchanged (straightens single-robot
                                       detours the composite pass cannot touch).
"""

import time
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from multi_robot_multi_goal_planning.problems.planning_env import State, BaseProblem


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def ilqr_connect(
    step: Callable[[NDArray, NDArray], NDArray],
    u_lo: NDArray,
    u_hi: NDArray,
    x0: NDArray,
    x1: NDArray,
    N: int,
    angular_dims: Optional[List[int]] = None,
    R_weight: float = 1e-2,
    Qf_weight: float = 1e2,
    max_iter: int = 40,
) -> Tuple[NDArray, NDArray, float]:
    """Optimise an N-step control sequence connecting x0 -> x1 under the discrete dynamics
    `step(x, u)` (controls clamped to [u_lo, u_hi]). Returns (states[N+1,n], controls[N,m],
    reach_error). Generic: pass the composite dynamics or a single-robot sub-dynamics."""
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    n = x0.shape[0]
    m = u_lo.shape[0]
    ang = list(angular_dims or [])
    R = R_weight * np.eye(m)
    Qf = Qf_weight * np.eye(n)

    def F(x, u):
        return step(x, np.clip(u, u_lo, u_hi))

    def sdiff(x, xg):
        d = x - xg
        for i in ang:
            d[i] = _wrap(d[i])
        return d

    def rollout(u):
        xs = np.zeros((N + 1, n))
        xs[0] = x0
        for k in range(N):
            xs[k + 1] = F(xs[k], u[k])
        return xs

    def cost(xs, u):
        c = sum(0.5 * u[k] @ R @ u[k] for k in range(N))
        d = sdiff(xs[-1], x1)
        return c + 0.5 * d @ Qf @ d

    def jac(x, u, eps=1e-6):
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        f0 = F(x, u)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            A[:, i] = (F(x + dx, u) - f0) / eps
        for i in range(m):
            du = np.zeros(m)
            du[i] = eps
            B[:, i] = (F(x, u + du) - f0) / eps
        return A, B

    u = np.zeros((N, m))
    xs = rollout(u)
    J = cost(xs, u)
    mu = 1e-6

    for _ in range(max_iter):
        Vx = Qf @ sdiff(xs[-1], x1)
        Vxx = Qf.copy()
        kff = np.zeros((N, m))
        Kfb = np.zeros((N, m, n))
        ok = True
        for k in range(N - 1, -1, -1):
            A, B = jac(xs[k], u[k])
            Qx = A.T @ Vx
            Qu = R @ u[k] + B.T @ Vx
            Qxx = A.T @ Vxx @ A
            Quu = R + B.T @ Vxx @ B + mu * np.eye(m)
            Qux = B.T @ Vxx @ A
            try:
                Quu_inv = np.linalg.inv(Quu)
            except np.linalg.LinAlgError:
                ok = False
                break
            kff[k] = -Quu_inv @ Qu
            Kfb[k] = -Quu_inv @ Qux
            Vx = Qx + Kfb[k].T @ Quu @ kff[k] + Kfb[k].T @ Qu + Qux.T @ kff[k]
            Vxx = Qxx + Kfb[k].T @ Quu @ Kfb[k] + Kfb[k].T @ Qux + Qux.T @ Kfb[k]
            Vxx = 0.5 * (Vxx + Vxx.T)
        if not ok:
            mu *= 10.0
            if mu > 1e8:
                break
            continue

        improved = False
        Jn = J
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
            xn = np.zeros((N + 1, n))
            xn[0] = x0
            un = np.zeros((N, m))
            for k in range(N):
                un[k] = np.clip(u[k] + alpha * kff[k] + Kfb[k] @ (xn[k] - xs[k]), u_lo, u_hi)
                xn[k + 1] = F(xn[k], un[k])
            Jn = cost(xn, un)
            if Jn < J - 1e-9:
                improved = True
                break
        if improved:
            converged = abs(J - Jn) < 1e-6
            xs, u, J = xn, un, Jn
            mu = max(mu * 0.7, 1e-8)
            if converged:
                break
        else:
            mu *= 10.0
            if mu > 1e8:
                break

    return xs, u, float(np.linalg.norm(sdiff(xs[-1], x1)))


def _composite_step(env):
    dt = getattr(env, "integration_dt", 0.05)
    return lambda x, u: x + dt * env.forward_dynamics(x, u)


def _robot_subdynamics(env, r: int):
    """Discrete sub-dynamics, control bounds and angular dims for robot r alone. Valid for the
    per-robot-decoupled kino envs (car, double integrator): the other robots' states do not
    enter robot r's derivative, so a dummy base state is fine."""
    dt = getattr(env, "integration_dt", 0.05)
    s, e = env.start_pos._array_slice[r]
    n_rob = env.start_pos.num_agents()
    cb = env.control_bounds()
    dc = cb.shape[1] // n_rob
    cs = r * dc
    base = env.start_pos.state()

    def step(xr, ur):
        fx = base.copy()
        fx[s:e] = xr
        fu = np.zeros(cb.shape[1])
        fu[cs : cs + dc] = ur
        return xr + dt * env.forward_dynamics(fx, fu)[s:e]

    ang = [a - s for a in getattr(env, "angular_dims", []) if s <= a < e]
    return step, cb[0, cs : cs + dc], cb[1, cs : cs + dc], ang, (s, e)


def _seg_valid(env, states: List[State], mode, lo, hi) -> bool:
    for st in states:
        x = st.q.state()
        if not (np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)):
            return False
        if not env.is_collision_free(st.q, mode):
            return False
    return True


def kinodynamic_ilqr_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 400,
    compression: float = 0.85,
    reach_tol: float = 0.12,
    time_budget: Optional[float] = None,
    verbose: bool = True,
):
    """Composite same-mode iLQR shortcut: reconnect two path states over a SHORTER horizon
    (N sized by how compressible the segment is) and accept if iLQR reaches the endpoint,
    the segment is collision-free, and it is shorter. Reduces time (fewer steps)."""
    dt = getattr(env, "integration_dt", 0.05)
    ang = list(getattr(env, "angular_dims", []) or [])
    lo, hi = env.state_bounds()[0], env.state_bounds()[1]
    step = _composite_step(env)
    cb = env.control_bounds()

    new_path = list(path)
    costs = [(len(new_path) - 1) * dt]
    times = [0.0]
    start_time = time.time()
    cnt = 0
    n_rob = env.start_pos.num_agents()

    for _ in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        nlen = len(new_path)
        if nlen < 4:
            break
        i = np.random.randint(0, nlen)
        j = np.random.randint(0, nlen)
        if i > j:
            i, j = j, i
        span = j - i
        if span < 3:
            continue
        mode = new_path[i].mode
        if new_path[j].mode != mode:
            continue

        # size the horizon by compressibility: N ~ span * (direct dist / travelled length)
        def rpos(state):
            return [np.asarray(env.robot_position(state.q, rr)) for rr in range(n_rob)]

        pi, pj = rpos(new_path[i]), rpos(new_path[j])
        direct = max(float(np.linalg.norm(pj[rr] - pi[rr])) for rr in range(n_rob))
        length = [0.0] * n_rob
        prev = pi
        for k in range(i + 1, j + 1):
            cur = rpos(new_path[k])
            for rr in range(n_rob):
                length[rr] += float(np.linalg.norm(cur[rr] - prev[rr]))
            prev = cur
        travelled = max(length)
        if travelled < 1e-6:
            continue
        frac = (direct / travelled) * random.uniform(1.2, 1.6)
        N = max(2, min(span - 1, int(np.ceil(span * min(frac, compression)))))
        if N >= span:
            continue

        x0 = np.asarray(new_path[i].q.state(), dtype=float)
        x1 = np.asarray(new_path[j].q.state(), dtype=float)
        xs, _u, err = ilqr_connect(step, cb[0], cb[1], x0, x1, N, angular_dims=ang)
        if err > reach_tol:
            continue
        xs[-1] = x1
        seg = [new_path[i]] + [State(env.start_pos.from_flat(xs[k]), mode) for k in range(1, N + 1)]
        if not _seg_valid(env, seg[1:-1], mode, lo, hi):
            continue
        new_path[i : j + 1] = seg
        cnt += 1
        costs.append((len(new_path) - 1) * dt)
        times.append(time.time() - start_time)

    if verbose:
        print(f"[kino-ilqr-shortcut] accepted {cnt} shortcuts, cost {costs[0]:.3f} -> {costs[-1]:.3f}")
    return new_path, [costs, times]


def _robot_seg_len(env, states: List[State], r: int) -> float:
    total = 0.0
    for a, b in zip(states[:-1], states[1:]):
        total += float(np.linalg.norm(np.asarray(env.robot_position(b.q, r)) - np.asarray(env.robot_position(a.q, r))))
    return total


def kinodynamic_ilqr_robot_shortcut(
    env: BaseProblem,
    path: List[State],
    max_iter: int = 400,
    reach_tol: float = 0.12,
    time_budget: Optional[float] = None,
    verbose: bool = True,
):
    """Per-robot iLQR shortcut over a FIXED horizon (keeps the multi-robot timeline aligned).
    Re-optimises one robot's sub-trajectory to straighten a detour, keeping the others fixed;
    may span mode boundaries as long as that robot's task is unchanged across the segment.
    Doubled mode-switch states (zero elapsed time) are handled via a path-index -> real-step
    map so both halves get the same re-optimised slice."""
    dt = getattr(env, "integration_dt", 0.05)
    lo, hi = env.state_bounds()[0], env.state_bounds()[1]
    n_rob = env.start_pos.num_agents()

    subs = [_robot_subdynamics(env, r) for r in range(n_rob)]
    new_path = list(path)
    costs = [sum(_robot_seg_len(env, new_path, r) for r in range(n_rob))]
    times = [0.0]
    start_time = time.time()
    cnt = 0
    rr_cycle = 0

    for _ in range(max_iter):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        nlen = len(new_path)
        if nlen < 4:
            break
        i = np.random.randint(0, nlen)
        j = np.random.randint(0, nlen)
        if i > j:
            i, j = j, i
        if j - i < 3:
            continue
        r = rr_cycle % n_rob
        rr_cycle += 1

        task_r = new_path[i].mode.task_ids[r]
        if any(new_path[i + k].mode.task_ids[r] != task_r for k in range(j - i + 1)):
            continue

        # path index -> real integration step (doubled mode-switch states take 0 time)
        t_index = [0]
        for k in range(1, j - i + 1):
            step_k = 0 if new_path[i + k - 1].mode != new_path[i + k].mode else 1
            t_index.append(t_index[-1] + step_k)
        N = t_index[-1]
        if N < 3:
            continue

        step_fn, u_lo, u_hi, ang_sub, (s, e) = subs[r]
        x0r = np.asarray(new_path[i].q.robot_state(r), dtype=float)
        x1r = np.asarray(new_path[j].q.robot_state(r), dtype=float)
        xs_r, _u, err = ilqr_connect(step_fn, u_lo, u_hi, x0r, x1r, N, angular_dims=ang_sub)
        if err > reach_tol:
            continue
        xs_r[-1] = x1r

        seg = []
        for k in range(j - i + 1):
            full = new_path[i + k].q.state().copy()
            full[s:e] = xs_r[t_index[k]]
            seg.append(State(env.start_pos.from_flat(full), new_path[i + k].mode))

        if _robot_seg_len(env, seg, r) >= _robot_seg_len(env, new_path[i : j + 1], r) - 1e-9:
            continue
        # only the changed interior states need collision checking
        ok = True
        for k in range(1, j - i):
            x = seg[k].q.state()
            if not (np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)):
                ok = False
                break
            if not env.is_collision_free(seg[k].q, seg[k].mode):
                ok = False
                break
        if not ok:
            continue
        new_path[i : j + 1] = seg
        cnt += 1
        costs.append(sum(_robot_seg_len(env, new_path, r) for r in range(n_rob)))
        times.append(time.time() - start_time)

    if verbose:
        print(f"[kino-ilqr-robot-shortcut] accepted {cnt} shortcuts, length {costs[0]:.3f} -> {costs[-1]:.3f}")
    return new_path, [costs, times]
