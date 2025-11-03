from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from time import perf_counter
import os, time

# ============================================================
# IMPORT your projectors + constraints
# ============================================================

from .constraints_projection import (
    project_affine_only,
    project_affine_cspace,
    project_affine_cspace_interior,
    project_affine_cspace_explore,
    project_nlp_sqp,
    project_affine
)

from .constraints import (
    AffineConfigurationSpaceEqualityConstraint,
    AffineConfigurationSpaceInequalityConstraint,
)

# ============================================================
# AFFINE GEOMETRIC TEST CASES (EXTENDED)
# ============================================================

# ============================================================
# I. PURE EQUALITY CONSTRAINTS  (Flat manifolds)
# ============================================================

def make_affine_line_diag():
    """
    1D line in R²:  x - y = 0  (main diagonal)
    """
    A = np.array([[1.0, -1.0]])
    b = np.array([0.0])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 2, "eq"


def make_affine_plane_xy():
    """
    3D horizontal plane:  z = 0
    """
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([0.0])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 3, "eq"


def make_affine_plane_tilted():
    """
    3D tilted plane:  2x + y - z = 0.5
    """
    A = np.array([[2.0, 1.0, -1.0]])
    b = np.array([0.5])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 3, "eq"


def make_affine_point_origin():
    """
    2D point at origin:  x = 0, y = 0
    (intersection of two lines)
    """
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 2, "eq"


def make_affine_line3D():
    """
    Line in 3D: intersection of two planes
        x + y + z = 0
        y - z = 0
    """
    A = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
    ])
    b = np.array([0.0, 0.0])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 3, "eq"


def make_affine_plane4D():
    """
    3D hyperplane in R⁴:  x₀ + 2x₁ - x₂ + 0.5x₃ = 0
    """
    A = np.array([[1.0, 2.0, -1.0, 0.5]])
    b = np.array([0.0])
    eq = AffineConfigurationSpaceEqualityConstraint(A, b)
    return [eq], [], 4, "eq"


# ============================================================
# II. PURE INEQUALITY CONSTRAINTS  (Convex regions)
# ============================================================

def make_affine_halfspace():
    """
    Single half-space in R²:  x + y <= 0.5
    """
    A = np.array([[1.0, 1.0]])
    b = np.array([0.5])
    ineq = AffineConfigurationSpaceInequalityConstraint(A, b)
    return [], [ineq], 2, "ineq"


def make_affine_triangle():
    """
    2D triangle:  x >= 0, y >= 0, x + y <= 1
    """
    A = np.array([
        [-1.0,  0.0],   # x >= 0  →  -x <= 0
        [ 0.0, -1.0],   # y >= 0  →  -y <= 0
        [ 1.0,  1.0],   # x + y <= 1
    ])
    b = np.array([0.0, 0.0, 1.0])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(3)]
    return [], ineqs, 2, "ineq"


def make_affine_box():
    """
    Square region in R²: |x| <= 1, |y| <= 1
    """
    A = np.array([
        [ 1.0,  0.0],
        [-1.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0, -1.0],
    ])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(4)]
    return [], ineqs, 2, "ineq"


def make_affine_strip():
    """
    2D strip: -0.8 <= x + y <= 0.8
    """
    A = np.array([[ 1.0,  1.0],
                  [-1.0, -1.0]])
    b = np.array([0.8, 0.8])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(2)]
    return [], ineqs, 2, "ineq"


def make_affine_box3D():
    """
    Cube in R³: |x|, |y|, |z| <= 1
    """
    A = np.array([
        [ 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 0.0, -1.0],
    ])
    b = np.ones(6)
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(6)]
    return [], ineqs, 3, "ineq"


def make_affine_prism4D():
    """
    4D “box”: |x₀| <= 1, |x₁| <= 1, |x₂| <= 1, |x₃| <= 1
    (Hyper-rectangle in R⁴)
    """
    A = np.vstack([
        np.eye(4),
        -np.eye(4)
    ])
    b = np.ones(8)
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(8)]
    return [], ineqs, 4, "ineq"


# ============================================================
# III. MIXED EQUALITY + INEQUALITY CONSTRAINTS  (Bounded manifolds)
# ============================================================

def make_affine_segment():
    """
    1D segment on x + y = 0  with  |x| <= 1.
    """
    Aeq = np.array([[1.0, 1.0]])
    beq = np.array([0.0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]

    Aineq = np.array([
        [ 1.0, 0.0],
        [-1.0, 0.0],
    ])
    bineq = np.array([1.0, 1.0])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], bineq[i:i+1]) for i in range(2)]

    return eqs, ineqs, 2, "mix"


def make_affine_plane_bounded():
    """
    3D bounded plane patch:
        z = 0 (equality)
        -1 <= x <= 1, -1 <= y <= 1  (inequalities)
    """
    Aeq = np.array([[0.0, 0.0, 1.0]])
    beq = np.array([0.0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]

    Aineq = np.array([
        [ 1.0,  0.0, 0.0],
        [-1.0,  0.0, 0.0],
        [ 0.0,  1.0, 0.0],
        [ 0.0, -1.0, 0.0],
    ])
    bineq = np.array([1.0, 1.0, 1.0, 1.0])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], bineq[i:i+1]) for i in range(4)]

    return eqs, ineqs, 3, "mix"


def make_affine_line_bounded3D():
    """
    Line segment in 3D:
        Equalities:
            x + y = 0
            z = 0
        Inequalities:
            |x| <= 1
    Result: 1D line segment in 3D.
    """
    Aeq = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    beq = np.array([0.0, 0.0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]

    Aineq = np.array([
        [ 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    bineq = np.array([1.0, 1.0])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], bineq[i:i+1]) for i in range(2)]

    return eqs, ineqs, 3, "mix"


def make_affine_cube_face4D():
    """
    3D plane bounded inside 4D cube:
        Equality: x₃ = 0
        Inequalities: |x₀|,|x₁|,|x₂| <= 1
    """
    Aeq = np.array([[0.0, 0.0, 0.0, 1.0]])
    beq = np.array([0.0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]

    Aineq = np.vstack([
        np.hstack([np.eye(3), np.zeros((3,1))]),
        np.hstack([-np.eye(3), np.zeros((3,1))]),
    ])
    bineq = np.ones(6)
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], bineq[i:i+1]) for i in range(6)]

    return eqs, ineqs, 4, "mix"

# ============================================================
# Helpers
# ============================================================

def random_samples(n, dim, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, dim))

class DummyConfig:
    """Minimal stand-in for your Configuration class."""
    def __init__(self, arr): self._x = np.asarray(arr)
    def state(self): return self._x
    def from_flat(self, arr): return DummyConfig(arr)


def project_samples(X, eq_constraints, ineq_constraints, method="affine-cspace"):
    """Project each sample with the chosen method."""
    projected = []
    n_eval = 0

    for x in X:
        q = DummyConfig(x)
        q_proj = None
        try:
            if method == "affine-analytic":
                if eq_constraints:
                    q_proj = project_affine_only(q, eq_constraints)
                else:
                    q_proj = q

            elif method == "affine-cspace":
                q_proj = project_affine_cspace(
                    q,
                    eq_constraints=eq_constraints if eq_constraints else None,
                    ineq_constraints=ineq_constraints if ineq_constraints else None,
                )
                if q_proj is None: q_proj = q

            elif method == "affine-cspace-interior":
                q_proj = project_affine_cspace_interior(
                    q,
                    eq_constraints=eq_constraints if eq_constraints else None,
                    ineq_constraints=ineq_constraints if ineq_constraints else None,
                )
                if q_proj is None: q_proj = q

            elif method == "affine-cspace-explore":
                q_proj = project_affine_cspace_explore(
                    q,
                    eq_constraints=eq_constraints if eq_constraints else None,
                    ineq_constraints=ineq_constraints if ineq_constraints else None,
                )
                if q_proj is None: q_proj = q

            elif method == "sqp":
                q_proj = project_nlp_sqp(q, eq_constraints=eq_constraints, ineq_constraints=ineq_constraints)
                if q_proj is None: q_proj = q

            elif method == "affine":
                q_proj = project_affine(q, eq_constraints + ineq_constraints)
                if q_proj is None: q_proj = q

            else:
                q_proj = q

        except Exception:
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
    # Make sure geom_order matches exactly the keys you used in `manifolds`
    geom_order = [
        "LineDiag",
        "PlaneXY",
        "PlaneTilted",
        "PointOrigin",
        "Line3D",
        "Plane4D",
        "Halfspace",
        "Triangle",
        "Box",
        "Strip",
        "Box3D",
        "Prism4D",
        "Segment",
        "PlaneBounded",
        "LineBounded3D",
        "CubeFace4D",
    ]
    method_order = ["affine-cspace", "affine-cspace-interior", "affine-cspace-explore", "sqp", "affine"]

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
            res = next((r for r in results if r["name"].startswith(geom + "-") and r["name"].endswith(method)), None)
            if res is None or res["samples"].size == 0:
                ax.axis("off")
                continue

            X = res["samples"]
            if X.shape[1] > 2:
                X = X[:, :2]
            ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.6)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(method, fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(geom, fontsize=10, rotation=90, labelpad=6)

            subtitle = f"E/S={res['E_per_S']:.2f} | EMD={res['EMD']:.2f} | t={res['time_ms']:.1f}ms"
            ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                    fontsize=7, ha="center", va="top")

    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    return fig, axes


# ============================================================
# Main experiment
# ============================================================

def main():
    print("[*] benchmarking_proj_affine starting ...")

    manifolds = {
        "LineDiag": make_affine_line_diag(),
        "PlaneXY": make_affine_plane_xy(),
        "PlaneTilted": make_affine_plane_tilted(),
        "PointOrigin": make_affine_point_origin(),
        "Line3D": make_affine_line3D(),
        "Plane4D": make_affine_plane4D(),
        "Halfspace": make_affine_halfspace(),
        "Triangle": make_affine_triangle(),
        "Box": make_affine_box(),
        "Strip": make_affine_strip(),
        "Box3D": make_affine_box3D(),
        "Prism4D": make_affine_prism4D(),
        "Segment": make_affine_segment(),
        "PlaneBounded": make_affine_plane_bounded(),
        "LineBounded3D": make_affine_line_bounded3D(),
        "CubeFace4D": make_affine_cube_face4D(),
    }

    # methods = ["affine-analytic", "affine-cspace", "gauss-newton", "sqp"]
    methods = ["affine-cspace", "affine-cspace-interior", "affine-cspace-explore", "sqp", "affine"]


    results = []
    for name, (eqs, ineqs, dim, ctype) in manifolds.items():
        X0 = random_samples(1000, dim, scale=2.0, seed=42)
        for m in methods:
            # skip analytic for pure inequalities
            if m == "affine-analytic" and not eqs:
                print(f"[!] Skipping affine-analytic for {name} (no equalities)")
                continue

            t0 = perf_counter()
            Xp, n_eval = project_samples(X0, eqs, ineqs, m)
            t1 = perf_counter()

            if Xp.size == 0:
                print(f"[!] {name}-{m}: no successful projections")
                continue

            Xv = Xp[:, :2]
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

    os.makedirs("multi_robot_multi_goal_planning/figures", exist_ok=True)
    fig, axes = plot_by_geometry(results, limits=(-2.0, 2.0))
    timestamp = int(time.time())
    png_path = f"multi_robot_multi_goal_planning/figures/coverage_affine_{timestamp}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"[+] Saved PNG: {os.path.abspath(png_path)}")

if __name__ == "__main__":
    main()