# test_acq_optimizers.py  — trials with resampled objectives, mean ± SEM
from __future__ import annotations
import time
from typing import Tuple, List, Dict
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit here)
# =========================
DD: int = 14          # discrete dimension
DC: int = 10          # continuous dimension
MC: int = 16          # number of RFF features
STARTS: int = 1       # random starts per method (per trial)
N_TRIALS: int = 10    # number of independent trials (objective resampled each time)

# =========================
# Your project imports
# =========================
from MIVABO import (
    MixedSpace, DiscreteSpec, ContinuousSpec,
    LinearConstraint,
    DiscreteFeatures, RFFContinuousFeatures, MixedFeatures,
    AlternatingOptimizer, BranchAndBoundOptimizer
)

from acquisition_optimizers import (
    FlipAndPolishOptimizer, SAOptimizer, TabuSearchOptimizer, GurobiAcquisitionOptimizer, HeuristicBnBOptimizer
)

# Methods to compare (editable)
TIME_BUDGET = 15.0  # seconds (change this to 5, 30, etc.)

# Empirically from your runs:
#   Tabu: ~1.333 s with default max_iters=300  → ~0.004443 s/iter
TABU_BASE_TIME   = 1.333
TABU_BASE_ITERS  = 300
TABU_PER_ITER    = TABU_BASE_TIME / TABU_BASE_ITERS
TABU_ITERS       = max(300, int(round(TIME_BUDGET / TABU_PER_ITER)))
TABU_NO_IMPROVE  = TABU_ITERS  # avoid early stop before the budget

# RFF-BnB: don’t let node cap end early; rely on time_limit instead
RFF_MAX_NODES = 1_000_000

METHODS = [

    ("Gurobi", lambda sp, pd, pc, pm:
        GurobiAcquisitionOptimizer(
            sp, pd, pc, pm,
            time_limit=TIME_BUDGET,
            mip_gap=0.03,        # modest gap so it returns a good incumbent within the budget
            verbose=False
        )
    ),

    ("RFFSpecialized", lambda sp, pd, pc, pm:
        HeuristicBnBOptimizer(
            sp, pd, pc, pm,
            time_limit=TIME_BUDGET,
            max_nodes=RFF_MAX_NODES,
            verbose=False
        )
    ),
]

# =========================
# Problem generator
# =========================
def make_random_problem(Dd: int, Dc: int, Mc: int):
    """Resamples a fresh objective: constraint, features, and weights."""
    device, dtype = torch.device("cpu"), torch.double

    # Mixed linear constraint: Ad xd + Ac xc <= b
    Ad = torch.randn(1, Dd, dtype=dtype, device=device)
    Ac = torch.randn(1, Dc, dtype=dtype, device=device)
    # pick a non-trivial b so origin is feasible and region is not degenerate
    b  = torch.tensor([float(max(0.5, 0.25 * torch.abs(Ad).sum().item()))], dtype=dtype, device=device)

    space = MixedSpace(
        disc=DiscreteSpec(Dd=Dd),
        cont=ContinuousSpec(Dc=Dc),
        lin_cons=LinearConstraint(A=torch.cat([Ad, Ac], dim=1), b=b),
        quad_cons=None, dtype=dtype, device=device
    )

    # Features (RFFs are randomized each trial)
    phi_d = DiscreteFeatures(Dd, include_bias=True)
    phi_c = RFFContinuousFeatures(Dc, num_features=Mc, kernel_lengthscale=0.25, dtype=dtype, device=device)
    phi_m = MixedFeatures(phi_d.dim, phi_c.dim)

    # Random weights per trial
    wd = torch.randn(phi_d.dim, dtype=dtype, device=device)
    wc = torch.randn(phi_c.dim, dtype=dtype, device=device)
    wm = torch.randn(phi_m.dim, dtype=dtype, device=device)

    return space, phi_d, phi_c, phi_m, wd, wc, wm

# =========================
# Feasibility helpers
# =========================
def ensure_feasible(space: MixedSpace, xd: Tensor, xc: Tensor, who: str):
    ok = True
    try:
        ok = space.mixed_feasible(xd, xc)
    except Exception:
        ok = False
    if not ok:
        # attempt to compute magnitude of violation for linear constraints if available
        violation = None
        try:
            if getattr(space, 'lin_cons', None) is not None and hasattr(space.lin_cons, 'A'):
                import torch as _t
                A = space.lin_cons.A
                b = space.lin_cons.b
                # build combined vector [xd; xc]
                vec = _t.cat([xd.to(dtype=_t.double), xc.to(dtype=_t.double)])
                A_t = A if isinstance(A, _t.Tensor) else _t.tensor(A, dtype=_t.double)
                b_t = b if isinstance(b, _t.Tensor) else _t.tensor(b, dtype=_t.double)
                if A_t.shape[1] == vec.numel():
                    diff = A_t @ vec - b_t
                    violation = float(_t.clamp(diff, min=0.0).max().item())
                else:
                    # try discrete-only check (first Dd columns)
                    Dd = space.disc.Dd
                    if A_t.shape[1] >= Dd:
                        left = A_t[:, :Dd] @ xd.to(dtype=_t.double)
                        diff = left - b_t
                        violation = float(_t.clamp(diff, min=0.0).max().item())
        except Exception:
            violation = None
        msg = f"[{who}] returned an infeasible solution under the mixed linear constraint."
        if violation is not None:
            msg += f" Max violation = {violation:.6e}"
        print(msg)
        raise RuntimeError(msg)

def random_start(space: MixedSpace) -> Tuple[Tensor, Tensor]:
    xd = torch.randint(0, 2, (space.disc.Dd,), dtype=space.dtype, device=space.device)
    xc = torch.rand(space.cont.Dc, dtype=space.dtype, device=space.device)
    # simple repair: zero-out xc then bits until feasible (rarely needed but cheap)
    try:
        if not space.mixed_feasible(xd, xc):
            xc = torch.zeros_like(xc)
            for i in range(space.disc.Dd):
                if not space.mixed_feasible(xd, xc):
                    xd[i] = 0.0
    except Exception:
        pass
    return xd, xc

# =========================
# One trial (one objective)
# =========================
def run_one_trial(Dd: int, Dc: int, Mc: int, n_methods_starts: int, methods=METHODS):
    """Creates a fresh random objective and evaluates all methods on it."""
    space, phi_d, phi_c, phi_m, wd, wc, wm = make_random_problem(Dd, Dc, Mc)

    trial_results: List[Tuple[str, float, float]] = []  # (label, best_val, elapsed)
    for label, ctor in methods:
        print(f"  Running method: {label}")
        opt = ctor(space, phi_d, phi_c, phi_m)
        best_val = float("inf")
        t0 = time.perf_counter()
        for s in range(n_methods_starts):
            # show light progress for multi-starts
            if n_methods_starts <= 8 or s in (0, n_methods_starts - 1):
                print(f"    start {s+1}/{n_methods_starts}...")
            xd0, xc0 = random_start(space)
            xd, xc, val = opt.solve(wd, wc, wm, xd0, xc0)
            ensure_feasible(space, xd, xc, who=label)
            if val < best_val:
                best_val = val
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"  Method {label} done: best={best_val:.6f} time={elapsed:.3f}s")
        trial_results.append((label, best_val, elapsed))
    return trial_results

# =========================
# Trials aggregation (mean ± SEM)
# =========================
def run_trials(n_trials: int, Dd: int, Dc: int, Mc: int, n_methods_starts: int, methods=METHODS):
    labels = [m[0] for m in methods]
    vals: Dict[str, List[float]] = {lab: [] for lab in labels}
    tms:  Dict[str, List[float]] = {lab: [] for lab in labels}

    for ti in range(n_trials):
        print(f"[Trial {ti+1}/{n_trials}] Creating problem and running methods...")
        trial = run_one_trial(Dd, Dc, Mc, n_methods_starts, methods)
        print(f"[Trial {ti+1}/{n_trials}] Completed. Summary:")
        for lab, v, tt in trial:
            vals[lab].append(v)
            tms[lab].append(tt)
            print(f"  - {lab}: best={v:.6f} time={tt:.3f}s")

    stats = []
    for lab in labels:
        v = np.asarray(vals[lab], dtype=float)
        t = np.asarray(tms[lab], dtype=float)
        mean_v = v.mean()
        sem_v  = (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        mean_t = t.mean()
        sem_t  = (t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else 0.0
        stats.append((lab, mean_v, sem_v, mean_t, sem_t))
    return stats

# =========================
# Plotting: mean ± SEM
# =========================
def plot_results(stats, title: str):
    labels  = [s[0] for s in stats]
    mean_v  = [s[1] for s in stats]
    sem_v   = [s[2] for s in stats]
    mean_t  = [s[3] for s in stats]
    sem_t   = [s[4] for s in stats]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].bar(x, mean_v, yerr=sem_v, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title("Best acquisition value (mean ± SEM) ↓")
    axes[0].set_ylabel("Acquisition")

    axes[1].bar(x, mean_t, yerr=sem_t, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_title("Runtime per method (mean ± SEM)")
    axes[1].set_ylabel("Seconds")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    stats = run_trials(N_TRIALS, DD, DC, MC, STARTS, methods=METHODS)
    for lab, mv, sv, mt, st in stats:
        print(f"{lab:>14}: value = {mv:.6f} ± {sv:.6f}  |  time = {mt:.3f} ± {st:.3f}s")
    plot_results(stats, title=f"Optimizers (Dd={DD}, Dc={DC}, Mc={MC}, starts={STARTS}, trials={N_TRIALS})")
