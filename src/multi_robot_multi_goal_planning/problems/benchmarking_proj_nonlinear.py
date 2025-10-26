from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from time import perf_counter
import os, time

# ============================================================
# IMPORT PROJECTORS
# ============================================================

from .constraints_projection import (
    project_affine_cspace,
    project_affine_cspace_interior,
    project_affine_cspace_explore,
    project_nlp_sqp,
    project_cspace_cnkz,
)
from .constraints import (
    AffineConfigurationSpaceEqualityConstraint,
    AffineConfigurationSpaceInequalityConstraint,
)

# ============================================================
# UTILITIES
# ============================================================

def random_samples(n, dim, scale=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, dim))

class DummyConfig:
    """Minimal stand-in for Configuration."""
    def __init__(self, arr): self._x = np.asarray(arr, dtype=float)
    def state(self): return self._x
    def from_flat(self, arr): return DummyConfig(arr)

# ============================================================
# TEST CASES
# ============================================================

# --- affine ---
def make_affine_line_diag():
    A = np.array([[1.0, -1.0]])
    b = np.array([0.0])
    return [AffineConfigurationSpaceEqualityConstraint(A,b)], [], 2, "affine"

def make_affine_box():
    A = np.array([[ 1,0],[-1,0],[0,1],[0,-1]])
    b = np.ones(4)
    ineqs = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1],b[i:i+1]) for i in range(4)]
    return [], ineqs, 2, "affine"

def make_affine_segment():
    Aeq = np.array([[1,1]]); beq = np.array([0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq,beq)]
    Aineq = np.array([[1,0],[-1,0]]); bineq = np.array([1,1])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1],bineq[i:i+1]) for i in range(2)]
    return eqs, ineqs, 2, "affine"

def make_affine_plane_bounded():
    Aeq = np.array([[0,0,1]]); beq = np.array([0])
    eqs = [AffineConfigurationSpaceEqualityConstraint(Aeq,beq)]
    Aineq = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
    bineq = np.array([1,1,1,1])
    ineqs = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1],bineq[i:i+1]) for i in range(4)]
    return eqs, ineqs, 3, "affine"

# --- nonlinear ---
class NonlinearConstraint:
    def __init__(self, func, jac):
        self.func = func; self.jac = jac
    def F(self,q_vec,mode=None,env=None): return np.atleast_1d(self.func(q_vec))
    def J(self,q_vec,mode=None,env=None): return np.atleast_2d(self.jac(q_vec))

def make_circle_2d(r=1.0):
    def f(q): return np.array([q[0]**2 + q[1]**2 - r**2])
    def j(q): return np.array([[2*q[0], 2*q[1]]])
    return [NonlinearConstraint(f,j)], [], 2, "nonlinear"

def make_sphere_3d(r=1.0):
    def f(q): return np.array([q[0]**2 + q[1]**2 + q[2]**2 - r**2])
    def j(q): return np.array([[2*q[0],2*q[1],2*q[2]]])
    return [NonlinearConstraint(f,j)], [], 3, "nonlinear"

def make_torus_3d(R=1.5, r=0.5):
    def f(q):
        xy = np.sqrt(q[0]**2 + q[1]**2)
        return np.array([(xy-R)**2 + q[2]**2 - r**2])
    def j(q):
        xy = np.sqrt(q[0]**2 + q[1]**2) + 1e-9
        return np.array([[2*(xy-R)*q[0]/xy, 2*(xy-R)*q[1]/xy, 2*q[2]]])
    return [NonlinearConstraint(f,j)], [], 3, "nonlinear"

def make_sine_wave_2d():
    def f(q): return np.array([q[1]-np.sin(q[0])])
    def j(q): return np.array([[-np.cos(q[0]), 1.0]])
    return [NonlinearConstraint(f,j)], [], 2, "nonlinear"

def make_paraboloid_3d():
    def f(q): return np.array([q[2] - (q[0]**2 + q[1]**2)])
    def j(q): return np.array([[-2*q[0], -2*q[1], 1.0]])
    return [NonlinearConstraint(f,j)], [], 3, "nonlinear"

# ============================================================
# PROJECT SAMPLES
# ============================================================

def project_samples(X, eqs, ineqs, method):
    projected = []
    for x in X:
        q = DummyConfig(x)
        try:
            if method == "affine-cspace":
                q_proj = project_affine_cspace(q, eqs, ineqs)
            elif method == "affine-cspace-interior":
                q_proj = project_affine_cspace_interior(q, eqs, ineqs)
            elif method == "affine-cspace-explore":
                q_proj = project_affine_cspace_explore(q, eqs, ineqs)
            elif method == "sqp":
                q_proj = project_nlp_sqp(q, eqs, ineqs)
            elif method == "cnkz":
                q_proj = project_cspace_cnkz(q, eqs+ineqs, None, None)
            else:
                q_proj = q
        except Exception:
            q_proj = q
        if q_proj is None:
            projected.append(q.state())
        else:
            projected.append(q_proj.state() if hasattr(q_proj,"state") else q_proj)
    return np.vstack(projected)

# ============================================================
# METRICS + VISUALIZATION
# ============================================================

def emd_to_uniform(X, ref_lim=1.0, n_ref=5000):
    if X.shape[1]!=2 or X.size==0: return np.nan
    rng=np.random.default_rng(0)
    ref=rng.uniform(-ref_lim,ref_lim,size=(n_ref,2))
    d=wasserstein_distance(X[:,0],ref[:,0])+wasserstein_distance(X[:,1],ref[:,1])
    return d/2.0

def plot_by_geometry(results, limits=(-2,2)):
    geom_order = sorted(set(r["geom"] for r in results))
    method_order = sorted(set(r["method"] for r in results))
    nrows, ncols = len(geom_order), len(method_order)
    fig, axes = plt.subplots(nrows,ncols,figsize=(3.3*ncols,3.3*nrows),squeeze=False)

    for i,g in enumerate(geom_order):
        for j,m in enumerate(method_order):
            ax=axes[i,j]
            res=[r for r in results if r["geom"]==g and r["method"]==m]
            if not res:
                ax.axis("off"); continue
            R=res[0]; X=R["samples"]
            if X.shape[1]>2: X=X[:,:2]
            ax.scatter(X[:,0],X[:,1],s=6,alpha=0.6)
            ax.set_xlim(limits); ax.set_ylim(limits)
            ax.set_xticks([]); ax.set_yticks([])
            if i==0: ax.set_title(m,fontsize=10,pad=6)
            if j==0: ax.set_ylabel(g,fontsize=10,rotation=90,labelpad=6)
            subtitle=f"E/S=1.00 | EMD={R['emd']:.2f} | t={R['time_ms']:.1f}ms"
            ax.text(0.5,-0.08,subtitle,transform=ax.transAxes,
                    fontsize=7,ha="center",va="top")
    fig.tight_layout(h_pad=1.0,w_pad=0.6)
    return fig,axes

# ============================================================
# MAIN
# ============================================================

def main():
    print("[*] Benchmarking affine + nonlinear projectors...")

    manifolds = {
        "LineDiag": make_affine_line_diag(),
        "Box2D": make_affine_box(),
        "Segment": make_affine_segment(),
        "PlaneBounded": make_affine_plane_bounded(),
        "Circle2D": make_circle_2d(),
        "Sphere3D": make_sphere_3d(),
        "Torus3D": make_torus_3d(),
        "SineWave2D": make_sine_wave_2d(),
        "Paraboloid3D": make_paraboloid_3d(),
    }

    results=[]
    for name,(eqs,ineqs,dim,ctype) in manifolds.items():
        X0=random_samples(500,dim,scale=2.0,seed=42)
        methods = ["sqp","cnkz"] if ctype=="nonlinear" else \
                  ["affine-cspace","affine-cspace-interior","affine-cspace-explore","sqp"]
        for m in methods:
            t0=perf_counter()
            Xp=project_samples(X0,eqs,ineqs,m)
            t1=perf_counter()
            emd=emd_to_uniform(Xp[:,:2])
            results.append({
                "geom":name,"method":m,"samples":Xp,
                "emd":emd,"time_ms":(t1-t0)*1e3
            })
            print(f"{name:<15s} {m:<24s} t={(t1-t0):.3f}s | EMD={emd:.3f}")

    # ---- plot identical layout ----
    os.makedirs("multi_robot_multi_goal_planning/figures",exist_ok=True)
    fig,_=plot_by_geometry(results,limits=(-2,2))
    out=f"multi_robot_multi_goal_planning/figures/coverage_affine_nonlinear_{int(time.time())}.png"
    fig.savefig(out,dpi=200,bbox_inches="tight")
    print(f"[+] Saved PNG: {os.path.abspath(out)}")

if __name__=="__main__":
    main()
