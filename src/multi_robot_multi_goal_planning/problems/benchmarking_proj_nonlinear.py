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
    project_gauss_newton,
    project_gn_adv,
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
    def __init__(self, arr): self._x = np.asarray(arr, dtype=float)
    def state(self): return self._x
    def from_flat(self, arr): return DummyConfig(arr)

# ============================================================
# BASIC CONSTRAINT CLASSES
# ============================================================

class NonlinearConstraint:
    def __init__(self, func, jac):
        self.func = func; self.jac = jac
    def F(self,q_vec,mode=None,env=None): return np.atleast_1d(self.func(q_vec))
    def J(self,q_vec,mode=None,env=None): return np.atleast_2d(self.jac(q_vec))

class NonlinearInequality:
    def __init__(self, func, jac):
        self.func = func; self.jac = jac
    def G(self,q_vec,mode=None,env=None): return np.atleast_1d(self.func(q_vec))
    def dG(self,q_vec,mode=None,env=None): return np.atleast_2d(self.jac(q_vec))

# ============================================================
# TEST GENERATORS
# ============================================================

# --- Affine Equality ---
def make_aff_eq_line():
    Aeq = np.array([[1, -1]]); beq = np.array([0])
    return [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)], [], 2, "aff_eq"

def make_aff_eq_plane():
    Aeq = np.array([[1, 0, -1]]); beq = np.array([0])
    return [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)], [], 3, "aff_eq"

# --- Affine Inequality ---
def make_aff_ineq_box():
    A = np.array([[1,0],[-1,0],[0,1],[0,-1]]); b = np.ones(4)
    ineq = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(4)]
    return [], ineq, 2, "aff_ineq"

def make_aff_ineq_prism():
    A = np.vstack([np.eye(3), -np.eye(3)]); b = np.ones(6)
    ineq = [AffineConfigurationSpaceInequalityConstraint(A[i:i+1], b[i:i+1]) for i in range(6)]
    return [], ineq, 3, "aff_ineq"

# --- Affine Mixed ---
def make_aff_mixed_slab():
    Aeq = np.array([[1,1,1]]); beq = np.array([0])
    eq = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]
    Aineq = np.vstack([np.eye(3), -np.eye(3)]); b = np.ones(6)
    ineq = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], b[i:i+1]) for i in range(6)]
    return eq, ineq, 3, "aff_mixed"

def make_aff_mixed_hyperplane():
    Aeq = np.array([[1,0,-1,0]]); beq = np.array([0])
    eq = [AffineConfigurationSpaceEqualityConstraint(Aeq, beq)]
    Aineq = np.vstack([np.eye(4), -np.eye(4)]); b = np.ones(8)
    ineq = [AffineConfigurationSpaceInequalityConstraint(Aineq[i:i+1], b[i:i+1]) for i in range(8)]
    return eq, ineq, 4, "aff_mixed"

# --- Nonlinear Equality ---
def make_circle_2d():
    f = lambda q: np.array([q[0]**2 + q[1]**2 - 1])
    j = lambda q: np.array([[2*q[0],2*q[1]]])
    return [NonlinearConstraint(f,j)], [], 2, "nl_eq"

def make_sine_2d():
    f = lambda q: np.array([q[1]-np.sin(q[0])])
    j = lambda q: np.array([[-np.cos(q[0]),1.0]])
    return [NonlinearConstraint(f,j)], [], 2, "nl_eq"

def make_paraboloid_3d():
    f = lambda q: np.array([q[2]-(q[0]**2+q[1]**2)])
    j = lambda q: np.array([[-2*q[0],-2*q[1],1.0]])
    return [NonlinearConstraint(f,j)], [], 3, "nl_eq"

def make_sphere_3d():
    f = lambda q: np.array([q[0]**2+q[1]**2+q[2]**2-1])
    j = lambda q: np.array([[2*q[0],2*q[1],2*q[2]]])
    return [NonlinearConstraint(f,j)], [], 3, "nl_eq"

def make_torus_3d():
    f = lambda q: np.array([(np.sqrt(q[0]**2+q[1]**2)-1.5)**2+q[2]**2-0.25])
    j = lambda q: np.array([[2*(np.sqrt(q[0]**2+q[1]**2)-1.5)*q[0]/(np.sqrt(q[0]**2+q[1]**2)+1e-9),
                             2*(np.sqrt(q[0]**2+q[1]**2)-1.5)*q[1]/(np.sqrt(q[0]**2+q[1]**2)+1e-9),
                             2*q[2]]])
    return [NonlinearConstraint(f,j)], [], 3, "nl_eq"

# --- Nonlinear Inequality ---
def make_quadratic_bowl():
    g = lambda q: np.array([q[0]**2+q[1]**2-1])
    j = lambda q: np.array([[2*q[0],2*q[1]]])
    return [], [NonlinearInequality(g,j)], 2, "nl_ineq"

def make_cone_3d():
    g = lambda q: np.array([q[2]-np.sqrt(q[0]**2+q[1]**2)])
    j = lambda q: np.array([[-q[0]/(np.sqrt(q[0]**2+q[1]**2)+1e-9),
                             -q[1]/(np.sqrt(q[0]**2+q[1]**2)+1e-9),1.0]])
    return [], [NonlinearInequality(g,j)], 3, "nl_ineq"

def make_parabolic_cap():
    g = lambda q: np.array([q[2]-(q[0]**2+q[1]**2)])
    j = lambda q: np.array([[-2*q[0],-2*q[1],1.0]])
    return [], [NonlinearInequality(g,j)], 3, "nl_ineq"

def make_half_sine():
    g = lambda q: np.array([q[1]-np.sin(q[0])])
    j = lambda q: np.array([[-np.cos(q[0]),1.0]])
    return [], [NonlinearInequality(g,j)], 2, "nl_ineq"

def make_ring_ineq():
    g = lambda q: np.array([(q[0]**2+q[1]**2-1)*(1-q[0]**2-q[1]**2)])
    j = lambda q: np.array([[2*q[0]*(1-2*(q[0]**2+q[1]**2)),2*q[1]*(1-2*(q[0]**2+q[1]**2))]])
    return [], [NonlinearInequality(g,j)], 2, "nl_ineq"

# --- Nonlinear Mixed ---
def make_sine_parabola():
    f = lambda q: np.array([q[1]-np.sin(q[0])])
    j = lambda q: np.array([[-np.cos(q[0]),1.0]])
    g = lambda q: np.array([q[1]-(q[0]**2)])
    jg = lambda q: np.array([[-2*q[0],1.0]])
    return [NonlinearConstraint(f,j)], [NonlinearInequality(g,jg)], 2, "nl_mixed"

def make_sphere_cap():
    f = lambda q: np.array([q[0]**2+q[1]**2+q[2]**2-1])
    j = lambda q: np.array([[2*q[0],2*q[1],2*q[2]]])
    g = lambda q: np.array([q[2]-0.5])
    jg = lambda q: np.array([[0,0,1]])
    return [NonlinearConstraint(f,j)], [NonlinearInequality(g,jg)], 3, "nl_mixed"

def make_helix_band():
    f = lambda q: np.array([q[2]-np.sin(q[0])-np.cos(q[1])])
    j = lambda q: np.array([[-np.cos(q[0]),np.sin(q[1]),1.0]])
    g = lambda q: np.array([q[0]**2+q[1]**2-4])
    jg = lambda q: np.array([[2*q[0],2*q[1],0]])
    return [NonlinearConstraint(f,j)], [NonlinearInequality(g,jg)], 3, "nl_mixed"

def make_banana():
    f = lambda q: np.array([q[1]-0.1*(q[0]**2-1)])
    j = lambda q: np.array([[-0.2*q[0],1.0]])
    g = lambda q: np.array([q[0]**2+q[1]**2-2])
    jg = lambda q: np.array([[2*q[0],2*q[1]]])
    return [NonlinearConstraint(f,j)], [NonlinearInequality(g,jg)], 2, "nl_mixed"

def make_saddle_band():
    f = lambda q: np.array([q[2]-(q[0]**2-q[1]**2)])
    j = lambda q: np.array([[-2*q[0],2*q[1],1.0]])
    g = lambda q: np.array([q[0]**2+q[1]**2-2])
    jg = lambda q: np.array([[2*q[0],2*q[1],0]])
    return [NonlinearConstraint(f,j)], [NonlinearInequality(g,jg)], 3, "nl_mixed"

# ============================================================
# PROJECT + PLOT
# ============================================================

def project_samples(X, eqs, ineqs, method):
    projected = []
    for x in X:
        q = DummyConfig(x)
        try:
            if method == "affine-cspace":
                q_proj = project_affine_cspace(q, eqs + ineqs)
            elif method == "affine-cspace-interior":
                q_proj = project_affine_cspace_interior(q, eqs, ineqs)
            elif method == "affine-cspace-explore":
                q_proj = project_affine_cspace_explore(q, eqs, ineqs)
            elif method == "sqp":
                q_proj = project_nlp_sqp(q, eqs + ineqs)
            elif method == "cnkz":
                q_proj = project_cspace_cnkz(q, eqs + ineqs, None, None)
            elif method == "gn":
                q_proj = project_gauss_newton(q, eqs + ineqs)
            elif method == "gn_adv":
                q_proj = project_gn_adv(q, eqs + ineqs)
            else:
                q_proj = q
        except Exception:
            q_proj = q
        projected.append(q_proj.state() if hasattr(q_proj,"state") else q_proj)
    return np.vstack(projected)

def emd_to_uniform(X, ref_lim=1.0, n_ref=5000):
    if X.shape[1]!=2 or X.size==0: return np.nan
    rng=np.random.default_rng(0)
    ref=rng.uniform(-ref_lim,ref_lim,size=(n_ref,2))
    d=wasserstein_distance(X[:,0],ref[:,0])+wasserstein_distance(X[:,1],ref[:,1])
    return d/2.0

def plot_by_geometry(results, geom_order, limits=(-2,2)):
    method_order = sorted(set(r["method"] for r in results))
    nrows, ncols = len(geom_order), len(method_order)
    fig, axes = plt.subplots(nrows,ncols,figsize=(3.8*ncols+1.5,3.4*nrows),squeeze=False)

    for i,g in enumerate(geom_order):
        for j,m in enumerate(method_order):
            ax = axes[i,j]
            res = [r for r in results if r["geom"]==g and r["method"]==m]
            if not res:
                ax.axis("off"); continue
            R=res[0]; X=R["samples"]
            if X.shape[1]>2: X=X[:,:2]
            ax.scatter(X[:,0],X[:,1],s=10,alpha=0.6)
            ax.set_xlim(limits); ax.set_ylim(limits)
            ax.set_xticks([]); ax.set_yticks([])
            if i==0: ax.set_title(m,fontsize=10,pad=6)
            subtitle=f"EMD={R['emd']:.2f} | t={R['time_ms']:.1f}ms"
            ax.text(0.5,-0.08,subtitle,transform=ax.transAxes,fontsize=7,ha="center",va="top")

        fig.text(0.94, 1 - (i + 0.5) / nrows, g, fontsize=9, va="center", ha="left")

    fig.tight_layout(h_pad=1.1,w_pad=0.6)
    return fig,axes

# ============================================================
# MAIN
# ============================================================

def main():
    print("[*] Benchmarking all affine/nonlinear combinations...")

    manifolds_ordered = [
        # affine eq
        ("AffEqLine2D", make_aff_eq_line()),
        ("AffEqPlane3D", make_aff_eq_plane()),
        # affine ineq
        ("AffIneqBox2D", make_aff_ineq_box()),
        ("AffIneqPrism3D", make_aff_ineq_prism()),
        # affine mixed
        ("AffMixedHyper4D", make_aff_mixed_hyperplane()),
        # nl eq
        ("Circle2D", make_circle_2d()),
        ("Sine2D", make_sine_2d()),
        ("Paraboloid3D", make_paraboloid_3d()),
        ("Sphere3D", make_sphere_3d()),
        ("Torus3D", make_torus_3d()),
        # nl ineq
        ("QuadBowl2D", make_quadratic_bowl()),
        ("HalfSine2D", make_half_sine()),
        # nl mixed
        # ("SineParabola", make_sine_parabola()),
        # ("SphereCap", make_sphere_cap()),
        ("HelixBand", make_helix_band()),
        ("Banana", make_banana()),
        # ("SaddleBand", make_saddle_band()),

    ]

    results=[]
    for name,(eqs,ineqs,dim,ctype) in manifolds_ordered:
        X0=random_samples(500,dim,scale=2.0,seed=42)
        if "aff" in ctype:
            methods=["affine-cspace","affine-cspace-interior","affine-cspace-explore","gn","sqp","cnkz","gn_adv"]
        else:
            methods=["gn","sqp","cnkz","gn_adv"]

        for m in methods:
            t0=perf_counter()
            Xp=project_samples(X0,eqs,ineqs,m)
            t1=perf_counter()
            emd=emd_to_uniform(Xp[:,:2])
            results.append({
                "geom":name,"method":m,"samples":Xp,
                "emd":emd,"time_ms":(t1-t0)*1e3
            })
            print(f"{name:<18s} {m:<24s} t={(t1-t0):.3f}s | EMD={emd:.3f}")

    geom_order=[name for name,_ in manifolds_ordered]
    fig,_=plot_by_geometry(results,geom_order,limits=(-2,2))

    os.makedirs("multi_robot_multi_goal_planning/figures",exist_ok=True)
    out=f"multi_robot_multi_goal_planning/figures/coverage_all_mix_{int(time.time())}.png"
    fig.savefig(out,dpi=200,bbox_inches="tight")
    print(f"[+] Saved PNG: {os.path.abspath(out)}")

if __name__=="__main__":
    main()
