from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from time import perf_counter
import os, time

# IMPORT existing + new projectors
from .constraints_projection import (
    project_affine_only,   # Analytic affine
    project_to_manifold,   # Gauss–Newton
    project_nlp_sqp        # SQP/NLP projection (SLSQP)
)

# Toy constraints
#   Equality-manifolds: expose F/J with F(x)=0 desired
#   Regions (inequalities): expose G/dG with G(x)<=0 feasible

# ----- Equality examples (2D/3D) -----
class CircleConstraint:
    # x^2 + y^2 - 1 = 0
    def F(self, x): return np.array([x[0] ** 2 + x[1] ** 2 - 1.0])
    def J(self, x): return np.array([[2.0 * x[0], 2.0 * x[1]]])

class TorusConstraint:
    """3D torus to 2D section: (sqrt(x^2+y^2)-R)^2 + z^2 = r^2"""
    def __init__(self, R=1.0, r=0.3):
        self.R, self.r = R, r
    def F(self, x):
        r_xy = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return np.array([(r_xy - self.R) ** 2 + x[2] ** 2 - self.r ** 2])
    def J(self, x):
        r_xy = np.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-12
        return np.array([[(x[0]/r_xy)*(2*(r_xy - self.R)),
                          (x[1]/r_xy)*(2*(r_xy - self.R)),
                          2.0 * x[2]]])
    
class SphereConstraint:
    """2D sphere surface in 3D: x^2 + y^2 + z^2 = 1"""
    def F(self, x):
        return np.array([x[0]**2 + x[1]**2 + x[2]**2 - 1.0])
    def J(self, x):
        return np.array([[2*x[0], 2*x[1], 2*x[2]]])


class AffineConstraint:
    # 1*x0 + 2*x1 = 1  (a 1D line manifold inside R^2)
    def __init__(self):
        self.mat = np.array([[1.0, 2.0]])
        self.constraint_pose = np.array([1.0])
    def F(self, x):
        return self.mat @ x - self.constraint_pose
    def J(self, x):
        return self.mat

# ----- Region examples (2D) -----
class BoxRegion:
    """
    Axis-aligned box: |x_i| <= 1, i.e. max(|x| - 1, 0) <= 0.
    """
    def G(self, x):
        # For feasibility G(x) <= 0; we define a single scalar that becomes >0 outside.
        # Use max over coordinates: g(x) = max(|x_i| - 1)
        g = max(abs(x[0]) - 1.0, abs(x[1]) - 1.0)
        return np.array([g])
    def dG(self, x):
        # Subgradient of max: pick active coord; tie-break arbitrarily
        a0 = abs(x[0]) - 1.0
        a1 = abs(x[1]) - 1.0
        if a0 >= a1:
            s0 = 0.0 if abs(x[0]) < 1e-12 else np.sign(x[0])
            return np.array([[s0, 0.0]])
        else:
            s1 = 0.0 if abs(x[1]) < 1e-12 else np.sign(x[1])
            return np.array([[0.0, s1]])

class EllipseRegion:
    """
    Elliptic disk: (x/a)^2 + (y/b)^2 - 1 <= 0  (convex smooth region)
    """
    def __init__(self, a=1.5, b=0.9):
        self.a, self.b = a, b
    def G(self, x):
        gx = (x[0] / self.a) ** 2 + (x[1] / self.b) ** 2 - 1.0
        return np.array([gx])
    def dG(self, x):
        return np.array([[2.0 * x[0] / (self.a ** 2), 2.0 * x[1] / (self.b ** 2)]])

# class L1DiamondRegion:
#     """
#     L1-ball (diamond): |x|/a + |y|/b - 1 <= 0 (convex, non-smooth corners)
#     """
#     def __init__(self, a=1.2, b=0.8, eps=1e-9):
#         self.a, self.b, self.eps = a, b, eps
#     def G(self, x):
#         gx = np.abs(x[0]) / self.a + np.abs(x[1]) / self.b - 1.0
#         return np.array([gx])
#     def dG(self, x):
#         sx = 0.0 if abs(x[0]) < self.eps else np.sign(x[0])
#         sy = 0.0 if abs(x[1]) < self.eps else np.sign(x[1])
#         return np.array([[sx / self.a, sy / self.b]])


def random_samples(n, dim, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, dim))

class DummyConfig:
    """Minimal stand-in for your Configuration class."""
    def __init__(self, arr): self._x = np.asarray(arr)
    def state(self): return self._x
    def from_flat(self, arr): return DummyConfig(arr)

def project_samples(X, constraint, method="gauss-newton", ctype="eq"):
    """
    Apply the chosen projection method to all points.

    This version tries ALL methods on ALL constraint types (even nonsensical ones).
    If projection fails, it falls back to returning the original point
    to preserve coverage density.
    """
    projected = []
    n_eval = 0

    for x in X:
        q = DummyConfig(x)
        q_proj = None

        try:
            if method == "affine-analytic":
                if hasattr(constraint, "mat"):
                    q_proj = project_affine_only(q, [constraint])
                else:
                    # fake a projection (return original)
                    q_proj = q

            elif method == "gauss-newton":
                q_proj = project_to_manifold(q, [constraint], eps=1e-6, max_iters=50)
                if q_proj is None:
                    q_proj = q  # fallback to original

            elif method == "sqp":
                # Attempt both equality and inequality forms
                eqs = [constraint] if hasattr(constraint, "F") and hasattr(constraint, "J") else None
                ineqs = [constraint] if hasattr(constraint, "G") and hasattr(constraint, "dG") else None
                q_proj = project_nlp_sqp(q, eq_constraints=eqs, ineq_constraints=ineqs)
                if q_proj is None:
                    q_proj = q

            else:
                q_proj = q  # unknown method fallback

        except Exception:
            # any numerical/shape error fallback
            q_proj = q

        projected.append(q_proj.state())
        n_eval += 1

    return np.vstack(projected), n_eval

# ============================================================
# Metrics
# ============================================================

def emd_to_uniform(X, ref_lim=1.0, n_ref=5000):
    """Approximate Earth Mover distance to uniform in box [-ref_lim,ref_lim]^2."""
    if X.shape[1] != 2 or X.size == 0:
        return np.nan
    rng = np.random.default_rng(0)
    ref = rng.uniform(-ref_lim, ref_lim, size=(n_ref, 2))
    d = wasserstein_distance(X[:, 0], ref[:, 0]) + wasserstein_distance(X[:, 1], ref[:, 1])
    return d / 2.0

# ============================================================
# Visualization
# ============================================================

def plot_by_geometry(results, limits=(-2, 2)):
    """
    Arrange plots by geometry (rows) and projection method (columns).

    Row order:
        AffineLine, Circle, TorusSection, BoxRegion, EllipseRegion, L1Diamond
    Column order:
        affine-analytic, gauss-newton, newton, sqp
    """
    geom_order = [
        "AffineLine",
        "Circle",
        "TorusSection",
        "Sphere",
        "BoxRegion",
        "EllipseRegion",
#        "L1Diamond",
    ]
    method_order = ["affine-analytic", "gauss-newton", "sqp"]

    nrows = len(geom_order)
    ncols = len(method_order)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.3 * ncols, 3.3 * nrows),
        squeeze=False
    )

    for i, geom in enumerate(geom_order):
        for j, method in enumerate(method_order):
            ax = axes[i, j]

            # Find matching result
            res = next(
                (r for r in results if r["name"].startswith(geom + "-") and r["name"].endswith(method)),
                None,
            )

            if res is None or res["samples"].size == 0:
                ax.axis("off")
                continue

            X = res["samples"]
            ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.6)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(method, fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(geom, fontsize=10, rotation=90, labelpad=6)

            # Add performance stats as small text
            subtitle = f"E/S={res['E_per_S']:.2f} | EMD={res['EMD']:.2f} | t={res['time_ms']:.1f}ms"
            ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                    fontsize=7, ha="center", va="top")

    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    return fig, axes

# ============================================================
# Main experiment
# ============================================================

def main():
    print("[*] benchmarking_proj starting ...")

    # (name) : (constraint, dim, ctype)
    manifolds = {
        "Circle":        (CircleConstraint(), 2, "eq"),
        "TorusSection":  (TorusConstraint(), 3, "eq"),
        "Sphere":       (SphereConstraint(), 3, "eq"),
        "AffineLine":    (AffineConstraint(), 2, "eq"),
        "BoxRegion":     (BoxRegion(),        2, "reg"),
        "EllipseRegion": (EllipseRegion(1.5, 0.9), 2, "reg"),
#        "L1Diamond":     (L1DiamondRegion(1.2, 0.8), 2, "reg"),
    }

    methods = [
        "affine-analytic",   # analytic (only if affine .mat present)
        "gauss-newton",      # existing GN
        "sqp",               # SLSQP Euclidean projection
    ]

    results = []
    for name, (constraint, dim, ctype) in manifolds.items():
        X0 = random_samples(1000, dim, scale=2.0, seed=42)
        for m in methods:
            if m == "affine-analytic" and not hasattr(constraint, "mat"):
                print(f"[!] Skipping affine projection for {name} (no .mat attribute)")
                continue
            t0 = perf_counter()
            Xp, n_eval = project_samples(X0, constraint, m, ctype)
            t1 = perf_counter()

            if Xp.size == 0:
                print(f"[!] {name}-{m}: no successful projections")
                continue

            Xv = Xp[:, :2]          # visualize first 2 dims if 3D
            E_per_S = n_eval / len(X0)
            emd = emd_to_uniform(Xv)
            results.append({
                "name": f"{name}-{m}",
                "samples": Xv,
                "E_per_S": E_per_S,
                "EMD": emd,
                "time": (t1 - t0),
                "time_ms": (t1 - t0) * 1e3,
            })

    # Output folder
    os.makedirs("multi_robot_multi_goal_planning/figures", exist_ok=True)

    # Plot + save
    fig, axes = plot_by_geometry(results, limits=(-2.0, 2.0))
    timestamp = int(time.time())
    png_path = f"multi_robot_multi_goal_planning/figures/coverage_grid_{timestamp}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    print(f"[+] Saved PNG: {os.path.abspath(png_path)}")

if __name__ == "__main__":
    main()
