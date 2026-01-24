"""
Run two optimization trials on a mixed (10 discrete, 10 continuous) problem:
 - one using AlternatingOptimizer
 - one using BranchAndBoundOptimizer

The script prints per-iteration: acquisition value (min w^T phi), observed y, active bits, and xc.
"""
import math
import os
import sys
import torch
import time
import csv
import argparse
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Parameters: \{.*use_label_encoder.*\} are not used\..*",
    category=UserWarning,
)

# Ensure repo root (one level up) is on sys.path so `import MIVABO` and local modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grover_objective import *
from xgb_breast_cancer_objective import *

# Optional: print which grover_objective module is being used, to avoid stale/duplicate imports
if os.environ.get('DEBUG_IMPORTS') == '1':
    import grover_objective as _go
    print(f"[DEBUG] grover_objective loaded from: {_go.__file__}")

# (sys.path was updated above)

from tests.MIVABO import (
    MixedSpace,
    DiscreteSpec,
    ContinuousSpec,
    DiscreteFeatures,
    RFFContinuousFeatures,
    MixedFeatures,
    MIVABO,
)
from tests.acquisition_optimizers import (
    HeuristicBnBOptimizer,
    # GurobiAcquisitionOptimizer,
    FlipAndPolishOptimizer,
)


torch.set_default_dtype(torch.double)

# Indicates whether RBFOpt was actually run (True) or the code fell back to random (False)
# RBFOpt baseline removed in this comparison script

# --- MINLP objective wrapper (from user-provided GAMS-like model) ---
# Integer variables i1..i10 are encoded as 4 binary bits each (little-endian); values in 1..10
# Continuous variables x11..x20 are represented by xc in [0,1]^10 and rescaled to [1,10]

# exponents alpha for integers and beta for continuous as in the provided model
alpha = [-0.32, -0.19, -0.405, -0.265, -0.175, -0.44, -0.275, -0.47, -0.31, -0.295]
beta  = [-0.105, -0.15, -0.235, -0.115, -0.42, -0.095, -0.115, -0.085, -0.115, -0.22]

BITS_PER_INT = 4
NUM_INTS = 10
NUM_CONT = 10


def decode_bits_to_ints(xd_bits: torch.Tensor) -> torch.Tensor:
    """Decode flat binary tensor of length NUM_INTS*BITS_PER_INT into integers 1..10."""
    ints = []
    for k in range(NUM_INTS):
        seg = xd_bits[k * BITS_PER_INT : (k + 1) * BITS_PER_INT]
        idx = 0
        for b_i, b in enumerate(seg):
            idx |= (int(b.item()) & 1) << b_i
        val = 1 + idx  # map 0->1, 1->2, ... up to 16
        # clamp to [1,10]
        if val < 1:
            val = 1
        if val > 10:
            val = 10
        ints.append(float(val))
    return torch.tensor(ints, dtype=torch.double)


def decode_continuous(xc_unit: torch.Tensor) -> torch.Tensor:
    """Map xc in [0,1]^10 to continuous variables in [1,10]."""
    return 1.0 + 9.0 * xc_unit


def minlp_objective_wrapper(space: MixedSpace):
    """Return a callable f(xd, xc) that evaluates the MINLP objective from the user's model.

    xd: binary tensor length NUM_INTS*BITS_PER_INT
    xc: unit tensor length NUM_CONT in [0,1]
    """
    def f(xd_bits, xc_unit):
        # sanity-check that the provided space matches the expected MINLP encoding
        expected_Dd = NUM_INTS * BITS_PER_INT
        expected_Dc = NUM_CONT
        if getattr(space, 'disc', None) is None or getattr(space, 'cont', None) is None:
            raise ValueError('space must be a MixedSpace with .disc and .cont specs')
        if space.disc.Dd != expected_Dd or space.cont.Dc != expected_Dc:
            raise ValueError(
                f"minlp objective expects discrete Dd={expected_Dd} and cont Dc={expected_Dc}, "
                f"but provided space has Dd={space.disc.Dd}, Dc={space.cont.Dc}. "
                "If you want to optimize a different small test problem, use make_small_problem() "
                "or provide a compatible space."
            )

        # decode
        ints = decode_bits_to_ints(xd_bits)
        cont = decode_continuous(xc_unit)

        # compute product term safely in double
        prod = 1.0
        for i in range(NUM_INTS):
            prod *= floatsafe_pow(ints[i], alpha[i])
        for j in range(NUM_CONT):
            prod *= floatsafe_pow(cont[j], beta[j])

        prod_term = 20000.0 * prod
        sum_term = float(ints.sum().item()) + float(cont.sum().item())
        obj = prod_term + sum_term
        return float(obj)

    return f


def floatsafe_pow(base, exponent):
    # base > 0 here; use Python pow for stability
    return base ** exponent


def run_optimizer(name: str, OptimizerClass, space: MixedSpace, n_init=30, n_iters=8, n_starts=8, objective=None, verbosity=1, collect_acquisition=False, trial_idx: int | None = None, log_callback=None):
    """Run one optimizer trial.

    verbosity: 0 = quiet (only essential prints), 1 = default (detailed per-iteration),
    collect_acquisition: if True return (incumbents, acquisitions) where acquisitions is
    a list of acquisition values (best_val) per iteration.
    """
    if verbosity > 0:
        print(f"\n=== Running optimizer: {name} ===")
    md = lambda Dd: DiscreteFeatures(Dd, include_bias=True)
    mc = lambda Dc: RFFContinuousFeatures(Dc, num_features=16, kernel_lengthscale=0.2, seed=None, dtype=space.dtype, device=space.device)
    # Construct optimizer; if unavailable (e.g., Gurobi not installed), raise with a clear message
    try:
        # Construct optimizer; enable verbose specifically for HeuristicBnB
        if OptimizerClass is HeuristicBnBOptimizer:
            # Match optimizer verbosity to the caller's verbosity flag so
            # optimizer internals don't spam output when the script is quiet.
            opt = OptimizerClass(
                space,
                md(space.disc.Dd),
                mc(space.cont.Dc),
                MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim),
                time_limit=40.0,
                verbose=(verbosity > 0),
            )
        # elif OptimizerClass is GurobiAcquisitionOptimizer:
        #     opt = OptimizerClass(
        #         space,
        #         md(space.disc.Dd),
        #         mc(space.cont.Dc),
        #         MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim),
        #         time_limit=15.0,
        #     )
        else:
            opt = OptimizerClass(
                space,
                md(space.disc.Dd),
                mc(space.cont.Dc),
                MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim),
            ) if OptimizerClass is not None else None
    except Exception as e:
        raise RuntimeError(f"Failed to construct optimizer '{name}': {type(e).__name__}: {e}")
    bo = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=opt)

    # Use provided objective if given, otherwise default to the MINLP wrapper
    if objective is not None:
        f_orig = objective
    else:
        f_orig = minlp_objective_wrapper(space)

    # Wrap objective to log every evaluation (including initialization inside bo.initialize)
    state = {"phase": "idle", "iter": 0, "init_idx": 0}

    def f_logged(xd, xc):
        t0 = time.time()
        y_val = f_orig(xd, xc)
        dt = time.time() - t0
        if callable(log_callback):
            # Determine index based on phase
            if state["phase"] == "init":
                state["init_idx"] += 1
                idx = state["init_idx"]
            else:
                idx = state.get("iter", 0)
            try:
                log_callback(trial_idx, state["phase"], idx, xd, xc, float(y_val), float(dt))
            except Exception:
                pass
        return y_val

    # Initialization evaluations
    state["phase"] = "init"
    state["init_idx"] = 0
    bo.initialize(f_logged, n_init=n_init)

    incumbents = []
    best_y_so_far = float("inf")
    acquisitions = []
    for t in range(n_iters):
        iter_start = time.time()
        if verbosity > 0:
            print(f"\n[iter {t+1:02d}] starting")
        # sample acquisition weight and run multi-start optimization to find argmin
        if verbosity > 0:
            print("  sampling posterior weight...")
        w = bo.ts.sample_weight()
        Md = bo.phi_d.dim
        Mc = bo.phi_c.dim
        wd = w[:Md]
        wc = w[Md : Md + Mc]
        wm = w[Md + Mc :]
        best_val = float("inf")
        best_x = None
        for s in range(n_starts):
            if verbosity > 0:
                print(f"  start {s+1}/{n_starts}: random init -> sampling start point...")
            xd0 = torch.randint(0, 2, (space.disc.Dd,), dtype=space.dtype, device=space.device)
            xc0 = torch.rand(space.cont.Dc, dtype=space.dtype, device=space.device)
            start_time = time.time()
            xd, xc, val = bo.optimizer.solve(wd, wc, wm, xd0, xc0)
            dur = time.time() - start_time
            if verbosity > 0:
                print(f"    solved start {s+1}: val={val:.6f} time={dur:.3f}s bits={int(xd.sum().item())}")
            if val < best_val:
                best_val = val
                best_x = (xd.clone(), xc.clone())

        xd, xc = best_x
        # Check feasibility before calling the expensive/externally provided objective.
        def _is_feasible(xd_t, xc_t):
            try:
                if hasattr(space, 'mixed_feasible'):
                    return bool(space.mixed_feasible(xd_t, xc_t))
                else:
                    return bool(space.disc_feasible(xd_t))
            except Exception:
                # If feasibility check fails, be conservative and treat as infeasible
                return False

        if not _is_feasible(xd, xc):
            # Per user request: raise an error when the acquisition optimizer returns infeasible solution
            raise RuntimeError(f"Acquisition optimizer returned infeasible solution at iter {t+1}: xd={xd.tolist()}, xc={xc.tolist()}")

        # Now call objective. Any infeasible evaluation is a hard error by contract.
        # Mark phase and iteration for logging
        state["phase"] = "iter"
        state["iter"] = t + 1
        y = f_logged(xd, xc)
        bo.observe(xd, xc, y)
        acquisitions.append(best_val)
        # update incumbent best-y
        if y < best_y_so_far:
            best_y_so_far = y
        incumbents.append(best_y_so_far)
        iter_dur = time.time() - iter_start
        if verbosity > 0:
            print(f"  chosen best_val={best_val:.6f} | y={y:.6f} | active_bits={int(xd.sum().item())} | iter_time={iter_dur:.3f}s")
            print(f"  xc={ [round(float(x),6) for x in xc.tolist()] }")
        elif (t + 1) % 10 == 0:
            print(f"  [iter {t+1:02d}] y={y:.6f} | best={best_y_so_far:.6f}")

    if collect_acquisition:
        return incumbents, acquisitions
    return incumbents


def make_minlp_problem():
    """Factory that returns (space, objective_callable).

    space: MixedSpace where discrete dims are binary bits encoding integers
    objective: callable(xd_bits, xc_unit) -> float
    """
    Dd = NUM_INTS * BITS_PER_INT
    Dc = NUM_CONT
    # No constraints for the full MINLP factory (keep default unconstrained space)
    space = MixedSpace(DiscreteSpec(Dd=Dd), ContinuousSpec(Dc=Dc))
    f = minlp_objective_wrapper(space)
    return space, f


def make_small_problem():
    """Factory for a small synthetic test problem.

    - 3 discrete binary variables (Dd=3)
    - 3 continuous variables in [0,1] (Dc=3)
    Objective: squared error to a fixed target (deterministic, easy to test).
    """
    Dd = 3
    Dc = 3
    # Build a mixed linear constraint over [xd; xc]: sum(xd) + 2*xc[0] <= 3.0
    import torch as _t
    A = _t.zeros(1, Dd + Dc, dtype=_t.double)
    # coefficients: 1 for each discrete bit, 2 for xc[0], 0 for other continuous dims
    A[0, :Dd] = 1.0
    A[0, Dd + 0] = 2.0
    b = _t.tensor([3.0], dtype=_t.double)
    lin_cons = None
    try:
        from MIVABO import LinearConstraint
        lin_cons = LinearConstraint(A=A, b=b)
    except Exception:
        lin_cons = None
    space = MixedSpace(DiscreteSpec(Dd=Dd), ContinuousSpec(Dc=Dc), lin_cons=lin_cons)

    def f(xd_bits, xc_unit):
        # validate shapes
        if xd_bits.numel() != space.disc.Dd:
            raise ValueError(f"xd_bits must have {space.disc.Dd} elements, got {xd_bits.numel()}")
        if xc_unit.numel() != space.cont.Dc:
            raise ValueError(f"xc_unit must have {space.cont.Dc} elements, got {xc_unit.numel()}")

        xd = xd_bits.to(dtype=space.dtype, device=space.device)
        xc = xc_unit.to(dtype=space.dtype, device=space.device)

        target_xd = torch.tensor([1.0, 0.0, 1.0], dtype=space.dtype, device=space.device)
        target_xc = torch.tensor([0.2, 0.8, 0.5], dtype=space.dtype, device=space.device)

        # enforce feasibility via MixedSpace helper if available
        try:
            from MIVABO import InfeasibleError
        except Exception:
            InfeasibleError = Exception

        # enforce mixed linear constraint via MixedSpace helper if available
        try:
            # MixedSpace.disc_feasible only checks discrete constraints; for mixed linear
            # constraints we compute A @ [xd; xc] <= b here.
            if space.lin_cons is not None:
                A = space.lin_cons.A
                b = space.lin_cons.b
                import torch as _t
                vec = _t.cat([xd, xc])
                if (_t.matmul(A, vec) > b + 1e-9).any():
                    raise InfeasibleError('mixed linear constraint violated')
        except Exception as e:
            if isinstance(e, InfeasibleError):
                raise
            # otherwise rethrow
            raise
        val = ((xd - target_xd) ** 2).sum() + ((xc - target_xc) ** 2).sum()
        return float(val)

    return space, f


def make_medium_problem():
    """Factory for a medium synthetic test problem.

    - 8 discrete binary variables (Dd=8)
    - 8 continuous variables in [0,1] (Dc=8)
    Objective: squared error to a fixed target vector.
    """
    Dd = 8
    Dc = 8

    # Build two mixed linear constraints of the form A @ [xd; xc] <= b
    # Constraint 1: sum(xd) + 2*xc[0] + xc[1] <= 6.0
    # Constraint 2: xd[2] + xd[3] + 3*xc[2] + 2*xc[3] <= 3.5
    A = torch.zeros(2, Dd + Dc, dtype=torch.double)
    # Constraint 1
    A[0, :Dd] = 1.0
    A[0, Dd + 0] = 2.0
    A[0, Dd + 1] = 1.0
    b = torch.tensor([6.0, 3.5], dtype=torch.double)
    # Constraint 2
    A[1, 2] = 1.0
    A[1, 3] = 1.0
    A[1, Dd + 2] = 3.0
    A[1, Dd + 3] = 2.0

    lin_cons = None
    try:
        from MIVABO import LinearConstraint
        lin_cons = LinearConstraint(A=A, b=b)
    except Exception:
        lin_cons = None

    space = MixedSpace(DiscreteSpec(Dd=Dd), ContinuousSpec(Dc=Dc), lin_cons=lin_cons)

    def f(xd_bits, xc_unit):
        # validate shapes
        if xd_bits.numel() != space.disc.Dd:
            raise ValueError(f"xd_bits must have {space.disc.Dd} elements, got {xd_bits.numel()}")
        if xc_unit.numel() != space.cont.Dc:
            raise ValueError(f"xc_unit must have {space.cont.Dc} elements, got {xc_unit.numel()}")

        xd = xd_bits.to(dtype=space.dtype, device=space.device)
        xc = xc_unit.to(dtype=space.dtype, device=space.device)

        # New objective: target-based quadratic on continuous variables plus
        # a discrete-target Hamming penalty and sinusoidal ripples for multimodality.
        target_xd = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=space.dtype, device=space.device)
        target_xc = torch.tensor([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.1], dtype=space.dtype, device=space.device)

        # quadratic continuous deviation
        cont_quad = ((xc - target_xc) ** 2).sum()
        # discrete mismatch penalty (squared differences behave like Hamming here)
        disc_pen = 3.0 * ((xd - target_xd) ** 2).sum()
        # sinusoidal ripples on continuous variables to introduce local minima
        ripples = 0.3 * torch.sin(4.0 * math.pi * xc).sum()
        # slight incentive to fewer active bits
        bit_bonus = -0.15 * float(xd.sum().item())

        val = cont_quad + disc_pen + float(ripples.item()) + bit_bonus

        return float(val)

    return space, f


def run_random_baseline(f, space, n_iters=50, trial_idx: int | None = None, log_callback=None, verbosity=0):
    """Simple random-search baseline: evaluate one random point per iteration and track incumbent."""
    if verbosity > 0:
        print(f"Running random-search baseline for {n_iters} iterations.")
    incumbents = []
    best = float('inf')
    for i in range(n_iters):
        # Sample until we find a feasible mixed point (if feasibility check is available)
        max_tries = 1000
        tries = 0
        while True:
            xd_bits = torch.randint(0, 2, (space.disc.Dd,), dtype=space.dtype, device=space.device)
            xc_unit = torch.rand(space.cont.Dc, dtype=space.dtype, device=space.device)
            tries += 1
            try:
                if hasattr(space, 'mixed_feasible'):
                    if space.mixed_feasible(xd_bits, xc_unit):
                        break
                    else:
                        if tries >= max_tries:
                            # Give up and treat as infeasible for this iteration; skip evaluation
                            if verbosity > 0:
                                print(f"  [RANDOM] could not find feasible sample after {max_tries} tries; skipping iteration {i+1}")
                            xd_bits = None
                            xc_unit = None
                            break
                        continue
                else:
                    # No mixed_feasible helper: accept the random sample
                    break
            except Exception:
                # If feasibility check throws, fall back to accepting the sample
                break
        if xd_bits is None:
            incumbents.append(best)
            if verbosity > 0:
                if (i + 1) % max(1, n_iters // 10) == 0 or i < 3:
                    print(f"  [RANDOM] iter {i+1}/{n_iters} | incumbent={best:.6f}")
            elif (i + 1) % 10 == 0:
                print(f"  [RANDOM] iter {i+1}/{n_iters} | incumbent={best:.6f}")
            continue
        t0 = time.time()
        y = f(xd_bits, xc_unit)
        dt = time.time() - t0
        if callable(log_callback):
            try:
                log_callback(trial_idx, 'iter', i + 1, xd_bits, xc_unit, float(y), float(dt))
            except Exception:
                pass
        if y < best:
            best = y
        incumbents.append(best)
        if verbosity > 0:
            if (i + 1) % max(1, n_iters // 10) == 0 or i < 3:
                print(f"  [RANDOM] iter {i+1}/{n_iters} | incumbent={best:.6f}")
        elif (i + 1) % 10 == 0:
            print(f"  [RANDOM] iter {i+1}/{n_iters} | incumbent={best:.6f}")
    return incumbents


# RBFOpt baseline removed


def run_multiple_trials(runner, space_factory, n_trials: int, n_iters: int, verbosity=0, collect_acq=False, log_callback=None):
    """Run `runner(space, f, n_iters)` for n_trials and return a tensor (n_trials, n_iters)

    runner: callable(space, f, n_iters) -> list[float]
    space_factory: callable() -> (space, f)
    """
    import torch as _t
    traces = []
    acq_traces = [] if collect_acq else None
    for trial in range(n_trials):
        if verbosity >= 0:
            print(f"  trial {trial+1}/{n_trials}")
        space, f = space_factory()
        # runner should return a list of incumbents, or (incumbents, acquisitions)
        if collect_acq:
            res = runner(space, f, n_iters, trial, log_callback)
            if isinstance(res, tuple):
                trace, acq = res
            else:
                trace = res
                acq = None
        else:
            trace = runner(space, f, n_iters, trial, log_callback)
            acq = None

        if len(trace) != n_iters:
            trace = (trace + [trace[-1]] * n_iters)[:n_iters]
        traces.append(_t.tensor(trace, dtype=_t.double))
        if collect_acq:
            if acq is None:
                # pad with NaNs
                acq = [float('nan')] * n_iters
            if len(acq) != n_iters:
                acq = (acq + [acq[-1]] * n_iters)[:n_iters]
            acq_traces.append(_t.tensor(acq, dtype=_t.double))

    traces = _t.stack(traces, dim=0)
    if collect_acq:
        return traces, _t.stack(acq_traces, dim=0)
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    verbosity = 1 if args.verbose else 0

    # compare average incumbent history across multiple trials
    n_iters = 100
    n_trials = 50
    # Create a run results output directory for plots
    out_root = os.path.join(os.path.dirname(__file__), 'run_results')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, timestamp)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as _e:
        # fallback to root if timestamped dir fails
        out_dir = out_root
        os.makedirs(out_dir, exist_ok=True)

    # CSV loggers for sampled points per method
    def make_csv_logger(path):
        fh = open(path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(fh)
        writer.writerow(['trial', 'phase', 'idx', 'y', 'time_s', 'active_bits', 'xd_bits', 'xc_values'])
        def _cb(trial_idx, phase, idx, xd, xc, y, dt):
            try:
                import torch as _t
                xd_cpu = xd.detach().cpu().view(-1).double()
                xc_cpu = xc.detach().cpu().view(-1).double()
                # binarize bits to {0,1}
                xd_bits = [int(float(v.round().clamp(0.0, 1.0).item())) for v in xd_cpu]
                active_bits = int(sum(xd_bits))
                xd_str = ''.join(str(b) for b in xd_bits)
                xc_vals = [float(v.item()) for v in xc_cpu]
                xc_str = '[' + ','.join(f'{v:.6f}' for v in xc_vals) + ']'
                writer.writerow([int(trial_idx) if trial_idx is not None else '', phase, int(idx), y, dt, active_bits, xd_str, xc_str])
                fh.flush()
            except Exception:
                pass
        return _cb, fh

    cb_hbnb, fh_hbnb = make_csv_logger(os.path.join(out_dir, 'samples_hbnb.csv'))
    cb_flip, fh_flip = make_csv_logger(os.path.join(out_dir, 'samples_alternating.csv'))
    cb_rand, fh_rand = make_csv_logger(os.path.join(out_dir, 'samples_random.csv'))

    # helper runners accept (space, f, n_iters, trial_idx, log_callback) and return list of incumbents
    def runner_hbnb(space, f, n_iters, trial_idx, log_cb):
        # Heuristic BnB with Tabu warm start is default inside the optimizer
        return run_optimizer("HeuristicBnB", HeuristicBnBOptimizer, space, n_init=5, n_iters=n_iters, n_starts=2, objective=f, verbosity=verbosity, collect_acquisition=True, trial_idx=trial_idx, log_callback=log_cb)

    # Gurobi disabled for current comparison
    # def runner_gurobi(space, f, n_iters):
    #     try:
    #         return run_optimizer("Gurobi", GurobiAcquisitionOptimizer, space, n_init=5, n_iters=n_iters, n_starts=2, objective=f)
    #     except RuntimeError as e:
    #         print(f"[WARN] Gurobi unavailable or failed to construct: {e}\nFalling back to Random for this trial.")
    #         return run_random_baseline(f, space, n_iters=n_iters)

    def runner_rand(space, f, n_iters, trial_idx, log_cb):
        return run_random_baseline(f, space, n_iters=n_iters, trial_idx=trial_idx, log_callback=log_cb, verbosity=verbosity)

    # choose space factory (use a medium test problem for constraints handling)
    space_factory = make_xgb_breast_cancer_problem

    import torch as _t
    # run multiple trials quietly (verbosity=0). collect acquisition traces for the two MIVABO methods
    print(f"Starting HeuristicBnB multi-trial run: n_trials={n_trials}, n_iters={n_iters}")
    traces_hbnb, acq_hbnb = run_multiple_trials(lambda s,f,n,trial,cb: run_optimizer('HeuristicBnB', HeuristicBnBOptimizer, s, n_init=5, n_iters=n, n_starts=2, objective=f, verbosity=verbosity, collect_acquisition=True, trial_idx=trial, log_callback=cb), space_factory, n_trials=n_trials, n_iters=n_iters, verbosity=verbosity, collect_acq=True, log_callback=cb_hbnb)

    print(f"Starting Alternating (Flip&Polish) multi-trial run: n_trials={n_trials}, n_iters={n_iters}")
    traces_flip, acq_flip = run_multiple_trials(lambda s,f,n,trial,cb: run_optimizer('Alternating', FlipAndPolishOptimizer, s, n_init=5, n_iters=n, n_starts=2, objective=f, verbosity=verbosity, collect_acquisition=True, trial_idx=trial, log_callback=cb), space_factory, n_trials=n_trials, n_iters=n_iters, verbosity=verbosity, collect_acq=True, log_callback=cb_flip)

    # (Gurobi multi-trial run removed)

    print(f"Starting Random baseline multi-trial run: n_trials={n_trials}, n_iters={n_iters}")
    traces_rand = run_multiple_trials(runner_rand, space_factory, n_trials=n_trials, n_iters=n_iters, verbosity=verbosity, collect_acq=False, log_callback=cb_rand)
    # compute mean and sem (std / sqrt(n))
    mean_hbnb = traces_hbnb.mean(dim=0)
    sem_hbnb = traces_hbnb.std(dim=0, unbiased=True) / (_t.sqrt(_t.tensor(n_trials, dtype=_t.double)))
    mean_flip = traces_flip.mean(dim=0)
    sem_flip = traces_flip.std(dim=0, unbiased=True) / (_t.sqrt(_t.tensor(n_trials, dtype=_t.double)))
    mean_rand = traces_rand.mean(dim=0)
    sem_rand = traces_rand.std(dim=0, unbiased=True) / (_t.sqrt(_t.tensor(n_trials, dtype=_t.double)))

    # acquisition stats (only for methods that provide acquisitions)
    acq_mean_hbnb = acq_hbnb.mean(dim=0)
    acq_sem_hbnb = acq_hbnb.std(dim=0, unbiased=True) / (_t.sqrt(_t.tensor(n_trials, dtype=_t.double)))
    acq_mean_flip = acq_flip.mean(dim=0)
    acq_sem_flip = acq_flip.std(dim=0, unbiased=True) / (_t.sqrt(_t.tensor(n_trials, dtype=_t.double)))

    # Plot incumbents (skip plotting if NO_PLOT=1)
    if os.environ.get('NO_PLOT') == '1':
        print('NO_PLOT=1 set — skipping plot creation.')
    else:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))

            xs = list(range(1, n_iters + 1))
            plt.plot(xs, mean_hbnb.tolist(), marker='o', label='MIVABO + HeuristicBnB')
            plt.fill_between(xs, (mean_hbnb - sem_hbnb).tolist(), (mean_hbnb + sem_hbnb).tolist(), alpha=0.2)

            plt.plot(xs, mean_flip.tolist(), marker='^', label='MIVABO + Alternating')
            plt.fill_between(xs, (mean_flip - sem_flip).tolist(), (mean_flip + sem_flip).tolist(), alpha=0.2)

            # Gurobi trace omitted

            plt.plot(xs, mean_rand.tolist(), marker='v', label='Random')
            plt.fill_between(xs, (mean_rand - sem_rand).tolist(), (mean_rand + sem_rand).tolist(), alpha=0.2)

            plt.xlabel('Iteration / evaluation group')
            plt.ylabel('Incumbent best y (min so far)')
            plt.title('Incumbent objective per iteration')
            plt.legend()
            plt.grid(alpha=0.3)
            out_path = os.path.join(out_dir, 'incumbent_plot.png')
            plt.tight_layout()
            plt.savefig(out_path)
            print(f"Saved incumbent plot to: {out_path}")
        except Exception as e:
            print(f"Could not create plot (matplotlib / numpy issue): {e}")

        # Additional single-method incumbent plots (show ALL trial trajectories)
        try:
            import matplotlib.pyplot as plt
            xs = list(range(1, n_iters + 1))

            # HBNB only
            plt.figure(figsize=(6, 4))
            for tr in range(traces_hbnb.shape[0]):
                plt.plot(xs, traces_hbnb[tr].tolist(), color='#1f77b4', alpha=0.35, linewidth=1)
            plt.xlabel('Iteration / evaluation group')
            plt.ylabel('Incumbent best y (min so far)')
            plt.title('Incumbent trajectories — HeuristicBnB (all trials)')
            plt.grid(alpha=0.3)
            out_hbnb = os.path.join(out_dir, 'incumbent_hbnb.png')
            plt.tight_layout()
            plt.savefig(out_hbnb)
            plt.close()
            print(f"Saved HBNB incumbent plot to: {out_hbnb}")

            # Alternating only
            plt.figure(figsize=(6, 4))
            for tr in range(traces_flip.shape[0]):
                plt.plot(xs, traces_flip[tr].tolist(), color='#ff7f0e', alpha=0.35, linewidth=1)
            plt.xlabel('Iteration / evaluation group')
            plt.ylabel('Incumbent best y (min so far)')
            plt.title('Incumbent trajectories — Alternating (all trials)')
            plt.grid(alpha=0.3)
            out_flip = os.path.join(out_dir, 'incumbent_alternating.png')
            plt.tight_layout()
            plt.savefig(out_flip)
            plt.close()
            print(f"Saved Alternating incumbent plot to: {out_flip}")

            # Random only
            plt.figure(figsize=(6, 4))
            for tr in range(traces_rand.shape[0]):
                plt.plot(xs, traces_rand[tr].tolist(), color='#2ca02c', alpha=0.35, linewidth=1)
            plt.xlabel('Iteration / evaluation group')
            plt.ylabel('Incumbent best y (min so far)')
            plt.title('Incumbent trajectories — Random (all trials)')
            plt.grid(alpha=0.3)
            out_rand = os.path.join(out_dir, 'incumbent_random.png')
            plt.tight_layout()
            plt.savefig(out_rand)
            plt.close()
            print(f"Saved Random incumbent plot to: {out_rand}")
        except Exception as e:
            print(f"Could not create single-method incumbent plots: {e}")

        # Second plot: acquisition mean ± SEM for HeuristicBnB and Gurobi
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            xs = list(range(1, n_iters + 1))
            plt.plot(xs, acq_mean_hbnb.tolist(), marker='o', label='HeuristicBnB acquisition')
            plt.fill_between(xs, (acq_mean_hbnb - acq_sem_hbnb).tolist(), (acq_mean_hbnb + acq_sem_hbnb).tolist(), alpha=0.2)
            plt.plot(xs, acq_mean_flip.tolist(), marker='^', label='Alternating acquisition')
            plt.fill_between(xs, (acq_mean_flip - acq_sem_flip).tolist(), (acq_mean_flip + acq_sem_flip).tolist(), alpha=0.2)
            # Gurobi acquisition curve omitted
            plt.xlabel('Iteration')
            plt.ylabel('Acquisition value (best per iteration)')
            plt.title('Acquisition (mean ± SEM)')
            plt.legend()
            plt.grid(alpha=0.3)
            out_path2 = os.path.join(out_dir, 'acquisition_plot.png')
            plt.tight_layout()
            plt.savefig(out_path2)
            print(f"Saved acquisition plot to: {out_path2}")
        except Exception as e:
            print(f"Could not create acquisition plot: {e}")

    # Close CSV files
    try:
        fh_hbnb.close()
        fh_flip.close()
        fh_rand.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
