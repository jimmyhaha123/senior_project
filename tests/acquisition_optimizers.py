# acquisition_optimizers.py
from __future__ import annotations
from typing import Optional, Tuple
import math
import random
from collections import deque

import torch
from torch import Tensor
import numpy as np
import heapq

# ===== Common helpers for mixed linear constraints =====

def _split_linear(space):
    lc = space.lin_cons
    if lc is None or not hasattr(lc.A, "shape"):
        return None, None, None
    A, b = lc.A, lc.b
    Dd, Dc = space.disc.Dd, space.cont.Dc
    if A.shape[1] == Dd + Dc:
        return A[:, :Dd], A[:, Dd:], b
    if A.shape[1] == Dd:
        return A, None, b
    return None, None, None

def _to_numpy_array(x):
    """Convert torch.Tensor / list / scalar to a NumPy float array (or None)."""
    if x is None:
        return None
    import numpy as _np
    try:
        # torch.Tensor -> numpy
        return x.detach().cpu().numpy().astype(float)
    except AttributeError:
        # already numpy / list / scalar
        return _np.asarray(x, dtype=float)

def _reduced_disc(space, xc: Tensor):
    Ad, Ac, b = _split_linear(space)
    if Ad is None:
        return None, None
    if Ac is None:
        return Ad, b
    return Ad, (b - Ac @ xc)

def _reduced_cont(space, xd: Tensor):
    Ad, Ac, b = _split_linear(space)
    if Ac is None:
        return None, None
    if Ad is None:
        return Ac, b
    return Ac, (b - Ad @ xd)

def _disc_feasible_given_xc(space, xd: Tensor, xc: Tensor, tol: float = 1e-9) -> bool:
    # Check discrete feasibility under reduced linear constraint Ad xd <= b - Ac xc
    Ad, rhs = _reduced_disc(space, xc)
    if Ad is None:
        # No discrete-relevant linear constraint; fall back to space.disc_feasible (handles quad if any)
        try:
            return space.disc_feasible(xd, tol=tol)
        except Exception:
            return True
    try:
        return bool(((Ad @ xd) <= (rhs + tol)).all())
    except Exception:
        return True

def _project_cont_given_xd(space, xc: Tensor, xd: Tensor, iters: int = 25) -> Tensor:
    # Project xc onto [0,1]^Dc ∩ {A_c xc <= b - A_d xd}
    Ac, rhs = _reduced_cont(space, xd)
    return space.project_polytope(xc, Ac, rhs, iters=iters)

def _acq_value(phi_d, phi_c, phi_m, wd, wc, wm, xd, xc) -> float:
    phid = phi_d(xd)
    phic = phi_c(xc)
    val = wd @ phid + wc @ phic + phi_m(phid, phic) @ wm
    return float(val.item() if hasattr(val, "item") else val)

def _polish_continuous(oracle, space, phi_d, phi_c, phi_m,
                       wd: Tensor, wc: Tensor, wm: Tensor,
                       xd: Tensor, xc_start: Tensor,
                       mode: Optional[str] = None) -> Tensor:
    def obj_fn(z: Tensor) -> Tensor:
        phid = phi_d(xd)
        phic = phi_c(z)
        return wd @ phid + wc @ phic + phi_m(phid, phic) @ wm

    def grad_fn(z: Tensor) -> Tensor:
        z_req = z.clone().detach().requires_grad_(True)
        val = obj_fn(z_req)
        val.backward()
        return z_req.grad.detach()

    # Prefer oracle with reduced-constraint awareness
    try:
        if mode is not None:
            xc = oracle.solve(obj_fn, grad_fn, space, xc_start=xc_start, xd_fixed=xd, mode=mode)  # type: ignore
        else:
            xc = oracle.solve(obj_fn, grad_fn, space, xc_start=xc_start, xd_fixed=xd)  # type: ignore
    except TypeError:
        xc = oracle.solve(obj_fn, grad_fn, space, xc_start=xc_start)

    # Always project once more to the reduced feasible set
    return _project_cont_given_xd(space, xc, xd, iters=25)

# ===== Base interface =====
class AcquisitionOptimizer:
    def solve(self, wd: Tensor, wc: Tensor, wm: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        raise NotImplementedError

# ===== Flip-&-Polish =====
class FlipAndPolishOptimizer(AcquisitionOptimizer):
    def __init__(self, space, phi_d, phi_c, phi_m, continuous_oracle=None, max_passes: int = 50):
        self.space = space
        self.phi_d = phi_d
        self.phi_c = phi_c
        self.phi_m = phi_m
        # Lazy import in case this file is standalone
        try:
            from MIVABO import TieredContinuousOracle
            self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        except Exception:
            self.cont_oracle = continuous_oracle
        self.max_passes = max_passes

    def _best_improving_flip(self, wd, wc, wm, xd, xc):
        base = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
        best_delta, best_i = 0.0, None
        for i in range(self.space.disc.Dd):
            xd[i] = 1.0 - xd[i]
            feas = _disc_feasible_given_xc(self.space, xd, xc)
            if feas:
                v = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
                delta = v - base
                if delta < best_delta - 1e-12:
                    best_delta, best_i = delta, i
            xd[i] = 1.0 - xd[i]
        return best_i, best_delta

    def solve(self, wd, wc, wm, xd0, xc0):
        xd = xd0.clone()
        xc = _project_cont_given_xd(self.space, xc0.clone(), xd)  # project initial xc to reduced feasible set

        # Repair xd if discrete reduced constraint violated
        if not _disc_feasible_given_xc(self.space, xd, xc):
            for i in range(self.space.disc.Dd):
                if not _disc_feasible_given_xc(self.space, xd, xc):
                    if xd[i] == 1.0:  # drop bits until feasible
                        xd[i] = 0.0

        val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)

        for _ in range(self.max_passes):
            i, delta = self._best_improving_flip(wd, wc, wm, xd, xc)
            if i is None or delta >= -1e-12:
                break
            xd[i] = 1.0 - xd[i]
            # polish continuous subject to reduced constraints
            xc = _polish_continuous(self.cont_oracle, self.space, self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc, mode=None)
            # final safety projection
            xc = _project_cont_given_xd(self.space, xc, xd)
            val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)

        # Final feasibility enforcement
        xc = _project_cont_given_xd(self.space, xc, xd)
        if not _disc_feasible_given_xc(self.space, xd, xc) or not self.space.mixed_feasible(xd, xc):
            raise RuntimeError("[Flip&Polish] could not produce a feasible solution.")
        return xd, xc, val

# ===== Simulated Annealing (binary focus) =====
class SAOptimizer(AcquisitionOptimizer):
    def __init__(self, space, phi_d, phi_c, phi_m, continuous_oracle=None,
                 T0: float = 1.0, T_min: float = 1e-3, alpha: float = 0.92,
                 k_reopt: int = 5, max_steps: int = 5000):
        self.space = space; self.phi_d = phi_d; self.phi_c = phi_c; self.phi_m = phi_m
        try:
            from MIVABO import TieredContinuousOracle
            self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        except Exception:
            self.cont_oracle = continuous_oracle
        self.T0, self.T_min, self.alpha = T0, T_min, alpha
        self.k_reopt, self.max_steps = k_reopt, max_steps

    def solve(self, wd, wc, wm, xd0, xc0):
        xd = xd0.clone()
        xc = _project_cont_given_xd(self.space, xc0.clone(), xd)

        # reduced discrete repair
        if not _disc_feasible_given_xc(self.space, xd, xc):
            for i in range(self.space.disc.Dd):
                if not _disc_feasible_given_xc(self.space, xd, xc):
                    if xd[i] == 1.0:
                        xd[i] = 0.0

        cur_val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
        best_val, best_xd, best_xc = cur_val, xd.clone(), xc.clone()

        T = self.T0; steps = 0; accepted_since_polish = 0
        while T > self.T_min and steps < self.max_steps:
            steps += 1
            i = random.randrange(self.space.disc.Dd)
            xd[i] = 1.0 - xd[i]
            if not _disc_feasible_given_xc(self.space, xd, xc):
                xd[i] = 1.0 - xd[i]; T *= self.alpha; continue

            new_val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
            if new_val < cur_val or random.random() < math.exp((cur_val - new_val)/max(1e-12, T)):
                cur_val = new_val
                accepted_since_polish += 1
                if cur_val < best_val:
                    best_val, best_xd, best_xc = cur_val, xd.clone(), xc.clone()

                if accepted_since_polish >= self.k_reopt:
                    accepted_since_polish = 0
                    xc = _polish_continuous(self.cont_oracle, self.space, self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc, mode=None)
                    cur_val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
                    if cur_val < best_val:
                        best_val, best_xd, best_xc = cur_val, xd.clone(), xc.clone()
            else:
                xd[i] = 1.0 - xd[i]
            T *= self.alpha

        # Final projection & check
        best_xc = _project_cont_given_xd(self.space, best_xc, best_xd)
        if not _disc_feasible_given_xc(self.space, best_xd, best_xc) or not self.space.mixed_feasible(best_xd, best_xc):
            raise RuntimeError("[SimAnneal] could not produce a feasible solution.")
        return best_xd, best_xc, best_val

# ===== Tabu Search (binary) =====
class TabuSearchOptimizer(AcquisitionOptimizer):
    def __init__(self, space, phi_d, phi_c, phi_m, continuous_oracle=None,
                 tabu_len: int = 8, max_iters: int = 300, max_no_improve: int = 60):
        self.space = space; self.phi_d = phi_d; self.phi_c = phi_c; self.phi_m = phi_m
        try:
            from MIVABO import TieredContinuousOracle
            self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        except Exception:
            self.cont_oracle = continuous_oracle
        self.tabu_len = tabu_len; self.max_iters = max_iters; self.max_no_improve = max_no_improve

    def solve(self, wd, wc, wm, xd0, xc0):
        xd = xd0.clone()
        xc = _project_cont_given_xd(self.space, xc0.clone(), xd)

        if not _disc_feasible_given_xc(self.space, xd, xc):
            for i in range(self.space.disc.Dd):
                if not _disc_feasible_given_xc(self.space, xd, xc):
                    if xd[i] == 1.0:
                        xd[i] = 0.0

        cur_val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
        best_val, best_xd, best_xc = cur_val, xd.clone(), xc.clone()

        tabu = deque(maxlen=self.tabu_len)
        no_improve = 0

        for _ in range(self.max_iters):
            base = cur_val
            best_i, best_v = None, None

            for i in range(self.space.disc.Dd):
                is_tabu = (i in tabu)
                xd[i] = 1.0 - xd[i]
                if _disc_feasible_given_xc(self.space, xd, xc):
                    v = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)
                    if (not is_tabu) or (v < best_val - 1e-12):  # aspiration
                        if (best_v is None) or (v < best_v - 1e-12):
                            best_v, best_i = v, i
                xd[i] = 1.0 - xd[i]

            if best_i is None:
                break

            xd[best_i] = 1.0 - xd[best_i]
            tabu.append(best_i)

            xc = _polish_continuous(self.cont_oracle, self.space, self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc, mode=None)
            cur_val = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd, xc)

            if cur_val < best_val - 1e-12:
                best_val, best_xd, best_xc = cur_val, xd.clone(), xc.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.max_no_improve:
                    break

        best_xc = _project_cont_given_xd(self.space, best_xc, best_xd)
        if not _disc_feasible_given_xc(self.space, best_xd, best_xc) or not self.space.mixed_feasible(best_xd, best_xc):
            raise RuntimeError("[Tabu] could not produce a feasible solution.")
        return best_xd, best_xc, best_val


class GurobiAcquisitionOptimizer(AcquisitionOptimizer):
    """
    Global MINLP for the *original* acquisition:
        wd·phi_d(xd) + wc·phi_c(xc) + wm·kron(phi_d(xd), phi_c(xc))
    Assumes:
      - DiscreteFeatures: [1, x_i, x_i x_j] (we create z_ij = x_i x_j aux binaries)
      - RFFContinuousFeatures: phi_c = norm * cos(W xc + b)
    Linear constraints A_d xd + A_c xc <= b are added from space.lin_cons.
    Discrete-only quadratic constraints (Q, q, b) are supported.
    If gurobi trig is unavailable, uses PWL for cos on a provable range.
    """

    def __init__(self, space, disc_phi, cont_phi, mix_phi,
                 time_limit: float | None = None,
                 mip_gap: float | None = None,
                 use_pwl_if_needed: bool = True,
                 pwl_segments: int = 101,
                 verbose: bool = False,
                 dump_model_on_infeasible: bool = False,
                 dump_path_prefix: str | None = None):
        try:
            import gurobipy as gp  # noqa: F401
        except Exception as e:
            raise ImportError("GurobiAcquisitionOptimizer requires gurobipy.") from e

        self.space = space
        self.phi_d = disc_phi
        self.phi_c = cont_phi
        self.phi_m = mix_phi
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.use_pwl_if_needed = use_pwl_if_needed
        self.pwl_segments = max(5, int(pwl_segments))
        self.verbose = verbose
        self.dump_model_on_infeasible = dump_model_on_infeasible
        self.dump_path_prefix = dump_path_prefix

    def _add_cos_feature(self, model, xc_vars, w_row, b_sc, norm):
        """
        Create y = norm * cos(w_row^T xc + b_sc) using Gurobi's exact trig constraint.
        Requires Gurobi >= 11 (model.addGenConstrCos must exist).
        Returns the Var y.
        """
        import gurobipy as gp
        from gurobipy import GRB

        if not hasattr(model, "addGenConstrCos"):
            raise RuntimeError(
                "This Gurobi version does not support trigonometric constraints. "
                "Please upgrade to Gurobi >= 11.0 for exact MINLP solving."
            )

        # Build affine expression t = b_sc + sum_j w_row[j] * xc[j]
        t_expr = gp.LinExpr(float(b_sc))
        for j, coeff in enumerate(w_row):
            if coeff != 0.0:
                t_expr.addTerms(float(coeff), xc_vars[j])

        # Create t with tight bounds inferred from xc ∈ [0,1]
        # (these bounds help spatial B&B a lot)
        row = list(w_row)
        t_min = float(b_sc + sum(min(c, 0.0) for c in row))
        t_max = float(b_sc + sum(max(c, 0.0) for c in row))
        if t_min > t_max:
            t_min, t_max = t_max, t_min
        eps = 1e-8 * (1.0 + max(abs(t_min), abs(t_max)))
        t = model.addVar(lb=t_min - eps, ub=t_max + eps, vtype=GRB.CONTINUOUS, name="t_rff")
        model.addConstr(t == t_expr, name="t_rff_def")

        # z = cos(t)  (IMPORTANT: argument order is (x, y) meaning y = cos(x))
        z = model.addVar(lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="cos_aux")
        model.addGenConstrCos(t, z, name="cos_constr")

        # y = norm * z
        y = model.addVar(lb=-abs(norm), ub=abs(norm), vtype=GRB.CONTINUOUS, name="rff_cos")
        model.addConstr(y == float(norm) * z, name="scale_cos")
        return y

    def solve(self, w_d, w_c, w_m, xd0, xc0):
        import time
        import numpy as np
        import torch as _t
        import gurobipy as gp
        from gurobipy import GRB

        t0 = time.time()
        if self.verbose:
            print("\n[GUROBI] === Acquisition solve (exact MINLP) ===")

        # ---- sizes ----
        Dd = self.space.disc.Dd
        Dc = self.space.cont.Dc
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        if self.verbose:
            print(f"[GUROBI] Sizes: Dd={Dd}, Dc={Dc}, Md={Md}, Mc={Mc}")
            print(f"[GUROBI] w_d:{tuple(w_d.shape)}, w_c:{tuple(w_c.shape)}, w_m:{tuple(w_m.shape)}")

        # ---- model + global nonconvex settings ----
        model = gp.Model("acq_min")
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.NonConvex = 2
        if self.time_limit is not None:
            model.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            model.Params.MIPGap = float(self.mip_gap)

        if not hasattr(model, "addGenConstrCos"):
            raise RuntimeError(
                "This Gurobi version does not support trigonometric constraints. "
                "Please upgrade to Gurobi >= 11.0 for exact MINLP solving."
            )

        # ---- decision vars ----
        xd = model.addVars(Dd, vtype=GRB.BINARY, name="xd")
        xc = model.addVars(Dc, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="xc")
        if self.verbose:
            print(f"[GUROBI] Vars added: binaries={Dd}, continuous={Dc}")

        # ---- z_ij = xd_i * xd_j for pairwise pseudo-Boolean features ----
        pair_idx = getattr(self.phi_d, "pairs", [(i, j) for i in range(Dd) for j in range(i + 1, Dd)])
        z_pairs = []
        for (i, j) in pair_idx:
            z = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
            model.addConstr(z <= xd[i]); model.addConstr(z <= xd[j])
            model.addConstr(z >= xd[i] + xd[j] - 1)
            z_pairs.append(z)
        if self.verbose:
            print(f"[GUROBI] Pairwise z_ij count: {len(z_pairs)}")

        # ---- φ_d in the order DiscreteFeatures expects: [1, xd..., z_pairs...] ----
        phi_d_vars = []
        if getattr(self.phi_d, "include_bias", True):
            phi_d_vars.append(1.0)
        for i in range(Dd):
            phi_d_vars.append(xd[i])
        for z in z_pairs:
            phi_d_vars.append(z)
        assert len(phi_d_vars) == Md, f"phi_d_vars({len(phi_d_vars)}) != Md({Md})"

        # ---- φ_c via exact trig RFF: norm * cos(W xc + b) ----
        if getattr(self.phi_c, "W", None) is None or getattr(self.phi_c, "b", None) is None:
            raise RuntimeError("GurobiAcquisitionOptimizer needs RFFContinuousFeatures with tensors W and b.")
        W = self.phi_c.W.detach().cpu().numpy()   # (M, Dc)
        bvec = self.phi_c.b.detach().cpu().numpy()  # (M,)
        norm = float(getattr(self.phi_c, "norm", 1.0))
        rff_M = int(getattr(self.phi_c, "M", len(W)))

        phi_c_vars = []
        if getattr(self.phi_c, "include_bias", False):
            phi_c_vars.append(1.0)

        # Build each RFF exactly: y_m = norm * cos(w_m^T xc + b_m)
        for m in range(rff_M):
            # tight t-bounds (helps a lot)
            row = W[m, :]
            t_min = float(bvec[m] + np.sum(np.minimum(row, 0.0)))
            t_max = float(bvec[m] + np.sum(np.maximum(row, 0.0)))
            if t_min > t_max:
                t_min, t_max = t_max, t_min
            eps = 1e-8 * (1.0 + max(abs(t_min), abs(t_max)))

            # t_m = b_m + w_m^T xc
            t_expr = gp.LinExpr(float(bvec[m]))
            for j in range(Dc):
                c = float(row[j])
                if c != 0.0:
                    t_expr.addTerms(c, xc[j])
            t_m = model.addVar(lb=t_min - eps, ub=t_max + eps, vtype=GRB.CONTINUOUS, name=f"t_{m}")
            model.addConstr(t_m == t_expr, name=f"t_{m}_def")

            # z_m = cos(t_m)   (FIXED ARGUMENT ORDER)
            z_m = model.addVar(lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"cos_{m}")
            model.addGenConstrCos(t_m, z_m, name=f"cos_constr_{m}")

            # y_m = norm * z_m
            y_m = model.addVar(lb=-abs(norm), ub=abs(norm), vtype=GRB.CONTINUOUS, name=f"rff_{m}")
            model.addConstr(y_m == norm * z_m, name=f"scale_{m}")
            phi_c_vars.append(y_m)

        # minimal logging
        assert len(phi_c_vars) == Mc, f"phi_c_vars({len(phi_c_vars)}) != Mc({Mc})"
        if self.verbose:
            try:
                print(f"[GUROBI] Built φ_c with exact cos: count={len(phi_c_vars)} (include_bias={getattr(self.phi_c,'include_bias',False)})")
            except Exception:
                pass

        # ---- objective: wd·φd + wc·φc + wm·kron(φd,φc) ----
        wd_list = [float(w_d[i].item()) for i in range(Md)]
        wc_list = [float(w_c[i].item()) for i in range(Mc)]
        WM = w_m.view(Md, Mc)

        lin = gp.LinExpr()
        quad = gp.QuadExpr()

        for i, c in enumerate(wd_list):
            a = phi_d_vars[i]
            lin += c * (a if not isinstance(a, (int, float)) else float(a))

        for j, c in enumerate(wc_list):
            b = phi_c_vars[j]
            lin += c * (b if not isinstance(b, (int, float)) else float(b))

        for i in range(Md):
            a = phi_d_vars[i]
            for j in range(Mc):
                coef = float(WM[i, j].item())
                if coef == 0.0:
                    continue
                b = phi_c_vars[j]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    lin += coef * a * b
                elif isinstance(a, (int, float)):
                    lin += coef * a * b
                elif isinstance(b, (int, float)):
                    lin += coef * b * a
                else:
                    quad += coef * a * b  # bilinear
        model.setObjective(lin + quad, GRB.MINIMIZE)

        # ---- linear constraints A_d xd + A_c xc <= b ----
        num_lin_mix = num_lin_disc = num_qdisc = 0
        if self.space.lin_cons is not None and hasattr(self.space.lin_cons.A, "shape"):
            Ad, Ac, b_lin = self.space.split_linear()
            if Ad is not None and Ac is not None:
                A_d = Ad.detach().cpu().numpy()
                A_c = Ac.detach().cpu().numpy()
                b_np = b_lin.detach().cpu().numpy()
                K = A_d.shape[0]
                for k in range(K):
                    lhs = gp.LinExpr()
                    for i in range(Dd):
                        coeff = float(A_d[k, i])
                        if coeff != 0.0:
                            lhs += coeff * xd[i]
                    for j in range(Dc):
                        coeff = float(A_c[k, j])
                        if coeff != 0.0:
                            lhs += coeff * xc[j]
                    model.addConstr(lhs <= float(b_np[k]), name=f"lin_mix_{k}")
                num_lin_mix += K
                if self.verbose:
                    print(f"[GUROBI] Added {K} mixed linear constraints.")
            elif Ad is not None:
                A_d = Ad.detach().cpu().numpy()
                b_np = self.space.lin_cons.b.detach().cpu().numpy()
                K = A_d.shape[0]
                for k in range(K):
                    lhs = gp.LinExpr()
                    for i in range(Dd):
                        coeff = float(A_d[k, i])
                        if coeff != 0.0:
                            lhs += coeff * xd[i]
                    model.addConstr(lhs <= float(b_np[k]), name=f"lin_disc_{k}")
                num_lin_disc += K
                if self.verbose:
                    print(f"[GUROBI] Added {K} discrete-only linear constraints.")

        # ---- discrete-only quadratic constraints (if provided) ----
        if self.space.quad_cons is not None:
            qcount = 0
            for qc in self.space.quad_cons:
                Q = getattr(qc, "Q", None); q = getattr(qc, "q", None); rhs = float(getattr(qc, "b", 0.0))
                if Q is None or q is None:
                    continue
                if hasattr(Q, "shape") and Q.shape[0] == Dd and Q.shape[1] == Dd:
                    Qn = Q.detach().cpu().numpy()
                    qn = q.detach().cpu().numpy()
                    qexpr = gp.QuadExpr()
                    for i in range(Dd):
                        for j in range(Dd):
                            c = float(Qn[i, j])
                            if c != 0.0:
                                qexpr += c * xd[i] * xd[j]
                    lin_q = gp.LinExpr()
                    for i in range(Dd):
                        c = float(qn[i])
                        if c != 0.0:
                            lin_q += c * xd[i]
                    model.addQConstr(qexpr + lin_q <= rhs, name=f"qdisc_{qcount}")
                    qcount += 1
            if qcount:
                num_qdisc += qcount
                if self.verbose:
                    print(f"[GUROBI] Added {qcount} discrete-only quadratic constraints.")

        # ---- optimize ----
        if self.verbose:
            print("[GUROBI] Optimizing...")
            if any([num_lin_mix, num_lin_disc, num_qdisc]):
                print(f"[GUROBI] Constraint summary -> lin_mix={num_lin_mix}, lin_disc={num_lin_disc}, q_disc={num_qdisc}")
        model.optimize()
        status = model.Status
        if self.verbose:
            try:
                if hasattr(model, 'MIPGap'):
                    print(f"[GUROBI] MIPGap achieved: {model.MIPGap}")
                if hasattr(model, 'ObjBound'):
                    print(f"[GUROBI] Best bound: {model.ObjBound}")
            except Exception:
                pass

        # ---- no usable solution? return +inf with diagnostics ----
        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            elapsed = time.time() - t0
            print(f"[GUROBI] Infeasible; returning inf. time={elapsed:.3f}s")
            try:
                if status == GRB.INFEASIBLE:
                    if self.verbose:
                        print("[GUROBI] Computing IIS...")
                        try:
                            model.computeIIS()
                        except Exception:
                            pass
                    if getattr(self, 'dump_model_on_infeasible', False):
                        prefix = getattr(self, 'dump_path_prefix', None) or 'acq_model'
                        print(f"[GUROBI] Writing model: {prefix}.lp")
                        model.write(f"{prefix}.lp")
                        try:
                            model.write(f"{prefix}.ilp")
                            print(f"[GUROBI] Writing IIS: {prefix}.ilp")
                        except Exception:
                            pass
            except Exception as e:
                if self.verbose:
                    print(f"[GUROBI] Infeasibility debug error: {e}")

            return xd0.clone(), xc0.clone(), float("inf")

        # ---- extract solution ----
        xd_sol = _t.zeros(Dd, dtype=self.space.dtype, device=self.space.device)
        xc_sol = _t.zeros(Dc, dtype=self.space.dtype, device=self.space.device)
        for i in range(Dd):
            xd_sol[i] = 1.0 if xd[i].X > 0.5 else 0.0
        for j in range(Dc):
            xc_sol[j] = float(xc[j].X)

        # ---- post-check & compute value ----
        try:
            if not self.space.mixed_feasible(xd_sol, xc_sol):
                print("[GUROBI] Post-check: projected xc to satisfy constraints once.")
                Ac, rhs = self.space.reduced_cont_constraints(xd_sol)
                xc_sol = self.space.project_polytope(xc_sol, Ac, rhs, iters=25)
        except Exception as e:
            if self.verbose:
                print(f"[GUROBI] Feasibility check warning: {e}")

        phid = self.phi_d(xd_sol)
        phic = self.phi_c(xc_sol)
        phim = self.phi_m(phid, phic)
        val = float((w_d @ phid + w_c @ phic + w_m @ phim).item())

        elapsed = time.time() - t0
        print(f"[GUROBI] Solution: val={val:.6f}, bits_on={int(xd_sol.sum().item())}, time={elapsed:.3f}s")
        return xd_sol, xc_sol, val
    
class HeuristicBnBOptimizer(AcquisitionOptimizer):
    """
        Best-first branch-and-bound on binaries with a fast, structure-aware lower bound.

        Objective:
                minimize  wd·phi_d(xd) + wc·phi_c(xc) + wm·kron(phi_d(xd), phi_c(xc))

        Node bound (partial assignment on xd):
            - Decompose the objective into a linear φ_d part and a mixed part:
                        lin(φ_d) + sum_j c_j(φ_d) · φ_c_j(xc),
                folding any φ_c bias column into the linear part.
            - Linear φ_d part: minimized over the relaxed box z ∈ [0,1]^Dd honoring fixed bits
                (bias/unary/pair features get interval bounds). This yields L_lb.
            - Mixed part: use ||c||_2 upper bounds via per-dimension intervals on c_j:
                        C2max = sqrt(Σ_j max(|c_j^min|, |c_j^max|)^2).
                Combine with either:
                    • deterministic: sup_x ||φ_c(x)||_2 (tight RFF radius on [0,1]^Dc)
                    • subgaussian:  norm(φ_c) · sqrt(2 log(1/δ))
                Optionally scale by alpha(depth) if include_alpha=True.

                Final bound:  h1 = L_lb − C2max · scale(depth)
                    scale(depth) = { Rc, or norm·sqrt(2 log(1/δ)) } × [alpha(depth) if enabled].

        Mixed linear constraints (Ad xd + Ac xc ≤ b):
            - Before pushing nodes: fast necessary relaxed feasibility check prunes clearly infeasible nodes.
            - Leaves and the seed: project xc to Ac xc ≤ b − Ad xd, polish with the oracle, and verify with space.mixed_feasible.
            - Constraints are not embedded in the bound, keeping it cheap and conservative.
    """

    def __init__(self, space, phi_d, phi_c, phi_m, continuous_oracle=None,
                delta: float = 1e-2, max_nodes: int = 50_000,
                time_limit: float | None = None, verbose: bool = False,
                alpha_root: float = 5, alpha_depth_saturation: int = 12,
                bound_type: str = "deterministic", include_alpha: bool = False,
                warm_start: str = "tabu",
                warm_tabu_len: int = 8,
                warm_tabu_max_iters: int = 200,
                warm_tabu_max_no_improve: int = 50):
        self.space = space
        self.phi_d = phi_d
        self.phi_c = phi_c
        self.phi_m = phi_m
        self.delta = float(delta)
        self.max_nodes = int(max_nodes)
        self.time_limit = time_limit
        self.verbose = verbose
        self.alpha_root = float(alpha_root)
        self.alpha_depth_saturation = int(alpha_depth_saturation)
        self.include_alpha = bool(include_alpha)
        # bound_type: "deterministic" uses sup_x ||phi_c(x)||; "subgaussian" uses SG tail with delta
        self.bound_type = str(bound_type).lower()
        if self.bound_type not in ("deterministic", "subgaussian"):
            self.bound_type = "subgaussian"

        # Warm start configuration
        self.warm_start = str(warm_start).lower() if warm_start is not None else "none"
        if self.warm_start not in ("none", "tabu"):
            self.warm_start = "none"
        self.warm_tabu_len = int(warm_tabu_len)
        self.warm_tabu_max_iters = int(warm_tabu_max_iters)
        self.warm_tabu_max_no_improve = int(warm_tabu_max_no_improve)

        try:
            from MIVABO import TieredContinuousOracle
            self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        except Exception:
            self.cont_oracle = continuous_oracle

        # Cache φ_d structure
        self._d_has_bias = bool(getattr(self.phi_d, "include_bias", True))
        self._pairs = list(getattr(self.phi_d, "pairs",
                                   [(i, j) for i in range(self.space.disc.Dd)
                                            for j in range(i + 1, self.space.disc.Dd)]))

        # Cache φ_c structure
        self._c_has_bias = bool(getattr(self.phi_c, "include_bias", False))

        # Debug flags (one-time prints) for investigating non-finite bounds
        self._printed_nonfinite_Wb_once = False
        self._printed_t_nonfinite_once = False

    # ---------- internal helpers ----------

    def _infer_c_bias_and_cos_cols(self, Dc: int, Mc: int):
        """
        Detect φ_c bias (constant feature) by evaluating at two points.
        Returns (bias_col or None, cos_cols list).
        """
        import numpy as np, torch
        z0 = torch.zeros(Dc, dtype=self.space.dtype, device=self.space.device)
        z1 = torch.rand(Dc, dtype=self.space.dtype, device=self.space.device)
        y0 = self.phi_c(z0).detach().cpu().numpy().ravel()
        y1 = self.phi_c(z1).detach().cpu().numpy().ravel()
        const = np.isclose(y0, y1, rtol=1e-9, atol=1e-12)
        bias_idx = np.where(const)[0].tolist()
        bias_col = bias_idx[0] if len(bias_idx) == 1 else None
        cos_cols = [j for j in range(Mc) if j != bias_col]
        return bias_col, cos_cols

    def _phi_c_radius_tight(self, Dc: int, Mc: int, cos_cols: list[int]) -> float:
        """
        Tight, safe upper bound R_c >= sup_{xc∈[0,1]^Dc} ||φ_c(xc)||_2.
        Uses per-column t-interval: t = b + W xc, xc in [0,1]^Dc.
        Falls back to |norm|*sqrt(M_cos) if W/b missing.
        """
        import math, numpy as np
        norm = float(getattr(self.phi_c, "norm", 1.0))
        W = getattr(self.phi_c, "W", None)
        b = getattr(self.phi_c, "b", None)
        if W is None or b is None:
            return abs(norm) * math.sqrt(len(cos_cols))  # safe fallback

        W_np = W.detach().cpu().numpy()   # shape (M, Dc)
        b_np = b.detach().cpu().numpy()   # shape (M,)

        # Optional one-time diagnostics for non-finite W/b
        if self.verbose and not self._printed_nonfinite_Wb_once:
            import numpy as _np
            n_infW = int(_np.isinf(W_np).sum()); n_nanW = int(_np.isnan(W_np).sum())
            n_infb = int(_np.isinf(b_np).sum()); n_nanb = int(_np.isnan(b_np).sum())
            if any([n_infW, n_nanW, n_infb, n_nanb]):
                ls = getattr(self.phi_c, "ls", None)
                print(f"[H-BNB][diag] Non-finite RFF params detected: W inf={n_infW}, nan={n_nanW}; b inf={n_infb}, nan={n_nanb}; lengthscale={ls}")
                self._printed_nonfinite_Wb_once = True

        def cos_abs_max_on_interval(tL: float, tU: float) -> float:
            # Ensure ordered
            if tL > tU:
                tL, tU = tU, tL
            # Non-finite bounds -> safe fallback: sup |cos| = 1
            if not (math.isfinite(tL) and math.isfinite(tU)):
                return 1.0
            # If the interval spans at least π, there exists kπ inside -> |cos| can be 1
            if (tU - tL) >= math.pi - 1e-15:
                return 1.0
            # Otherwise check if any multiple of π falls inside explicitly
            # Use NumPy ceil/floor on finite floats to avoid rare math.floor issues
            import numpy as _np
            kmin = int(_np.ceil(float(tL) / math.pi))
            kmax = int(_np.floor(float(tU) / math.pi))
            if kmin <= kmax:
                return 1.0
            # Else endpoints dominate
            return max(abs(math.cos(tL)), abs(math.cos(tU)))

        s2 = 0.0
        for j in cos_cols:
            row = W_np[j, :]
            tL = float(b_np[j] + np.minimum(row, 0.0).sum())
            tU = float(b_np[j] + np.maximum(row, 0.0).sum())
            # Optional one-time diagnostics for non-finite t-intervals
            if self.verbose and not self._printed_t_nonfinite_once:
                if not (math.isfinite(tL) and math.isfinite(tU)):
                    ls = getattr(self.phi_c, "ls", None)
                    row_min = float(np.min(row)); row_max = float(np.max(row))
                    l1 = float(np.sum(np.abs(row)))
                    print(
                        f"[H-BNB][diag] Non-finite t-interval detected at row {j}: "
                        f"tL={tL}, tU={tU}, b={float(b_np[j])}, minW={row_min:.3e}, maxW={row_max:.3e}, ||W_row||_1={l1:.3e}, lengthscale={ls}"
                    )
                    self._printed_t_nonfinite_once = True
            try:
                y_abs = abs(norm) * cos_abs_max_on_interval(tL, tU)
            except OverflowError as _e:
                if self.verbose:
                    ls = getattr(self.phi_c, "ls", None)
                    row_min = float(np.min(row)); row_max = float(np.max(row))
                    l1 = float(np.sum(np.abs(row)))
                    print(
                        f"[H-BNB][diag] OverflowError in cos_abs_max_on_interval at row {j}: "
                        f"tL={tL}, tU={tU}, b={float(b_np[j])}, minW={row_min:.3e}, maxW={row_max:.3e}, ||W_row||_1={l1:.3e}, lengthscale={ls}"
                    )
                raise
            s2 += y_abs * y_abs
        return float(math.sqrt(s2))

    def _alpha_for_depth(self, depth: int) -> float:
        """
        Adaptive pruning multiplier (>=1). Safer near the root, →1 deeper.
        You can tune root aggressiveness & saturation depth via attributes.
        """
        k_root = getattr(self, "alpha_root", 1.20)           # 1.20 = 20% safer at very shallow nodes
        d_sat  = getattr(self, "alpha_depth_saturation", 12) # depth where alpha→1
        if depth >= d_sat:
            return 1.0
        # Linear decay from k_root at depth 0 to 1.0 at depth d_sat
        return 1.0 + (k_root - 1.0) * (1.0 - depth / float(d_sat))
    

    def _indices_d(self):
        """Return indices for φ_d features: bias_idx, unary_idx[], pair_idx[] in the row order matching φ_d."""
        Dd = self.space.disc.Dd
        idx = 0
        bias_idx = None
        if self._d_has_bias:
            bias_idx = idx
            idx += 1
        unary_idx = list(range(idx, idx + Dd))
        idx += Dd
        pair_idx = list(range(idx, idx + len(self._pairs)))
        return bias_idx, unary_idx, pair_idx

    def _necessary_relaxed_feasible(
        self,
        xd_fix_mask: np.ndarray,
        xd_fix_vals: np.ndarray
    ) -> bool:
        """
        Fast necessary check for A_d xd + A_c xc <= b with free binaries and all
        continuous relaxed to [0,1]. If even the best-case per-row LHS exceeds b,
        the node is infeasible.
        """
        Ad, Ac, b = _split_linear(self.space)
        if Ad is None and Ac is None:
            return True

        Ad = _to_numpy_array(Ad); Ac = _to_numpy_array(Ac); b = _to_numpy_array(b)
        Dd = self.space.disc.Dd

        # Contribution from fixed binaries
        if Ad is not None:
            xdf = np.zeros(Dd, dtype=float)
            if xd_fix_mask.any():
                xdf[xd_fix_mask] = xd_fix_vals
            lhs_fixed = Ad @ xdf                       # (K,)
            best_free_disc = np.minimum(0.0, Ad[:, ~xd_fix_mask]).sum(axis=1) if (~xd_fix_mask).any() else 0.0
        else:
            lhs_fixed = 0.0
            best_free_disc = 0.0

        # Best-case from continuous (relaxed to [0,1])
        best_cont = np.minimum(0.0, Ac).sum(axis=1) if Ac is not None else 0.0

        lhs_min = lhs_fixed + best_free_disc + best_cont
        return bool(np.all(lhs_min <= b + 1e-9))

    def _linear_lower_bound(self, wd_row: np.ndarray, WM_bias_col: np.ndarray | None,
                            xd_fix_mask: np.ndarray, xd_fix_vals: np.ndarray) -> float:
        """
        Lower bound for the z-dependent linear part:
            wd·phi_d(z) + (WM[:, j_bias]^T phi_d(z)) + wc_bias (the latter added outside).
        Uses relaxation z_free ∈ [0,1] and pair features ∈ [0,1] with reductions for fixed bits.
        """
        Dd = self.space.disc.Dd
        bias_idx, unary_idx, pair_idx = self._indices_d()

        coef_d = wd_row.copy()
        if WM_bias_col is not None:
            coef_d = coef_d + WM_bias_col  # absorb φ_c-bias interaction into linear part

        # Align fixed values to full-length discrete vector for safe indexing
        xdf = np.zeros(Dd, dtype=float)
        if xd_fix_mask.any():
            xdf[xd_fix_mask] = xd_fix_vals

        total = 0.0
        # Bias feature (constant 1)
        if bias_idx is not None:
            total += float(coef_d[bias_idx])

        # Unaries
        for k, irow in enumerate(unary_idx):
            c = float(coef_d[irow])
            if xd_fix_mask[k]:
                if xdf[k] > 0.5:  # fixed 1
                    total += c
                # if fixed 0: contribute 0
            else:
                # free z_k ∈ [0,1] -> min contrib = min(0, c)
                total += min(0.0, c)

        # Pairs
        for (p, q), idx in zip(self._pairs, pair_idx):
            c = float(coef_d[idx])
            p_fixed, q_fixed = xd_fix_mask[p], xd_fix_mask[q]
            if p_fixed and q_fixed:
                # constant 0 or 1
                if (xdf[p] > 0.5) and (xdf[q] > 0.5):
                    total += c
            elif p_fixed and (xdf[p] < 0.5):
                # one is 0 -> product 0
                pass
            elif q_fixed and (xdf[q] < 0.5):
                pass
            else:
                # product is a free [0,1] scalar (either both free, or one fixed=1 & the other free)
                total += min(0.0, c)

        return float(total)

    def _C2max(self, wc: np.ndarray, WM: np.ndarray,
               xd_fix_mask: np.ndarray, xd_fix_vals: np.ndarray,
               cos_cols: list[int]) -> float:
        """
        Compute C2max = sqrt( sum_j max(|c_j^min|, |c_j^max|)^2 ) over the cosine columns.
        Each c_j(z) = wc_j + sum_i WM[i,j] * phi_d_i(z). Bound via intervals on φ_d features.
        """
        Dd = self.space.disc.Dd
        bias_idx, unary_idx, pair_idx = self._indices_d()

        # Prepare per-feature "state" and values:
        # For bias: value = 1 (constant)
        # For unary k: value ∈ {fixed 0, fixed 1, [0,1]}
        # For pair (p,q): value ∈ {0,1,[0,1]} depending on fixed bits.
        # We'll aggregate constant contribution into base and handle variable parts by sign.
        # Align fixed values to full-length discrete vector for safe indexing
        xdf = np.zeros(Dd, dtype=float)
        if xd_fix_mask.any():
            xdf[xd_fix_mask] = xd_fix_vals

        c2 = 0.0
        for j in cos_cols:
            base = float(wc[j])  # starts with wc_j

            # Add constant feature contributions to base
            if bias_idx is not None:
                base += float(WM[bias_idx, j])

            # Unaries
            var_contrib_min = 0.0
            var_contrib_max = 0.0
            for k, irow in enumerate(unary_idx):
                gamma = float(WM[irow, j])
                if xd_fix_mask[k]:
                    if xdf[k] > 0.5:
                        base += gamma
                    # if fixed 0: nothing
                else:
                    # free scalar in [0,1]
                    var_contrib_min += min(0.0, gamma)
                    var_contrib_max += max(0.0, gamma)

            # Pairs
            for (p, q), idx in zip(self._pairs, pair_idx):
                gamma = float(WM[idx, j])
                p_fixed, q_fixed = xd_fix_mask[p], xd_fix_mask[q]
                if p_fixed and q_fixed:
                    if (xdf[p] > 0.5) and (xdf[q] > 0.5):
                        base += gamma
                elif p_fixed and (xdf[p] < 0.5):
                    pass
                elif q_fixed and (xdf[q] < 0.5):
                    pass
                else:
                    # product behaves as a free scalar in [0,1]
                    var_contrib_min += min(0.0, gamma)
                    var_contrib_max += max(0.0, gamma)

            lo = base + var_contrib_min
            hi = base + var_contrib_max
            c_abs = max(abs(lo), abs(hi))
            c2 += c_abs * c_abs

        return float(math.sqrt(max(c2, 0.0)))

    def _branch_score(self, unary_row_idx: int, wd_row: np.ndarray, WM: np.ndarray, cos_cols: list[int]) -> float:
        """
        Simple branching score for a free binary k: combine absolute unary linear coeff
        and its total coupling to cos columns (L1 of the WM row over cos dims).
        """
        c_lin = abs(float(wd_row[unary_row_idx]))
        c_mix = float(np.abs(WM[unary_row_idx, cos_cols]).sum()) if len(cos_cols) else 0.0
        return c_lin + c_mix

    # ---------- main solve ----------
    def solve(self, wd: Tensor, wc: Tensor, wm: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        import time
        t_start = time.time()

        Dd = self.space.disc.Dd
        Md = self.phi_d.dim
        Mc = self.phi_c.dim

        # Convert weights to numpy; WM is (Md, Mc)
        wd_np = _to_numpy_array(wd).reshape(-1)
        wc_np = _to_numpy_array(wc).reshape(-1)
        WM_np = _to_numpy_array(wm).reshape(Md, Mc)

        # Determine φ_c structure and bound scaling
        Dc = self.space.cont.Dc
        c_bias_col, cos_cols = self._infer_c_bias_and_cos_cols(Dc, Mc)
        # Determine multiplier for mixed bound term depending on mode
        if self.bound_type == "deterministic":
            R_c = self._phi_c_radius_tight(Dc, Mc, cos_cols)
            def _mixed_scale(depth: int) -> float:
                alpha = self._alpha_for_depth(depth) if self.include_alpha else 1.0
                return R_c * alpha
        else:
            # Sub-Gaussian scale: ||phi_c||_psi2 <= norm; tail uses sqrt(2 log(1/delta))
            import math as _math
            # sanitize delta to (0,1)
            delta = self.delta
            if not (delta > 0.0 and delta < 1.0):
                delta = 1e-2
            phi_norm = float(getattr(self.phi_c, "norm", 1.0))
            base = phi_norm * _math.sqrt(max(0.0, 2.0 * _math.log(1.0 / delta)))
            def _mixed_scale(depth: int) -> float:
                alpha = self._alpha_for_depth(depth) if self.include_alpha else 1.0
                return base * alpha
        tol_prune = 1e-7   # slightly hardened prune tolerance

        # Linear constants outside of z:
        const_linear = 0.0
        # absorb φ_c bias constant wc_bias
        if c_bias_col is not None:
            const_linear += float(wc_np[c_bias_col])

        # Build the "linear φd coefficients" for bounding:
        # wd + (WM[:, c_bias_col]) if φ_c bias exists
        wm_bias_col = WM_np[:, c_bias_col] if c_bias_col is not None else None
        # NOTE: bias feature contributes as coef_d[bias]*1 in _linear_lower_bound

        # Initial incumbent: optional warm start (Tabu) then polish/project
        xd_seed = xd0.clone()
        xc_seed = _project_cont_given_xd(self.space, xc0.clone(), xd_seed)

        # Repair discrete feasibility for seed if needed
        if not _disc_feasible_given_xc(self.space, xd_seed, xc_seed):
            for i in range(Dd):
                if xd_seed[i] == 1.0:
                    xd_seed[i] = 0.0
                if _disc_feasible_given_xc(self.space, xd_seed, xc_seed):
                    break

        # Warm start with Tabu if enabled
        xd_best, xc_best = xd_seed, xc_seed
        if self.warm_start == "tabu":
            try:
                if self.verbose:
                    print("[H-BNB] Warm-starting with Tabu search...")
                tabu = TabuSearchOptimizer(
                    self.space, self.phi_d, self.phi_c, self.phi_m,
                    continuous_oracle=self.cont_oracle,
                    tabu_len=self.warm_tabu_len,
                    max_iters=self.warm_tabu_max_iters,
                    max_no_improve=self.warm_tabu_max_no_improve,
                )
                xd_ws, xc_ws, val_ws = tabu.solve(wd, wc, wm, xd_seed, xc_seed)
                xd_best, xc_best = xd_ws.clone(), xc_ws.clone()
                if self.verbose:
                    print(f"[H-BNB] Tabu warm start value={val_ws:.6g}")
            except Exception as e:
                if self.verbose:
                    print(f"[H-BNB] Tabu warm start failed: {e}. Falling back to seed.")

        # Polish warm-start incumbent once with continuous oracle (safe)
        try:
            xc_best = _polish_continuous(self.cont_oracle, self.space, self.phi_d, self.phi_c, self.phi_m,
                                         wd, wc, wm, xd_best, xc_best, mode=None)
        except Exception:
            pass
        val_best = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd_best, xc_best)
        if self.verbose:
            print(f"[H-BNB] seed incumbent value={val_best:.6g}")

        # Node structure: (bound, seq_id, fixed_mask(bool[Dd]), fixed_vals(float[n_fixed]))
        # Use a heap (best-first by bound)
        import heapq
        heap = []
        seq = 0

        # Root node (no bits fixed)
        root_mask = np.zeros(Dd, dtype=bool)
        root_vals = np.zeros(0, dtype=float)
        # quick necessary feasibility check
        if not self._necessary_relaxed_feasible(root_mask, root_vals):
            if self.verbose:
                print("[H-BNB] Root infeasible under relaxed check; returning incumbent.")
            return xd_best, xc_best, float(val_best)

        # H1 bound at root
        lin_lb_root = self._linear_lower_bound(wd_np, wm_bias_col, root_mask, root_vals) + const_linear
        C2_root = self._C2max(wc_np, WM_np, root_mask, root_vals, cos_cols)
        h1_root = lin_lb_root - C2_root * _mixed_scale(0)
        heapq.heappush(heap, (h1_root, seq, 0, root_mask, root_vals))  # include depth=0
        seq += 1

        nodes = 0
        tol = 1e-9
        # Diagnostics counters
        pruned_by_bound_pop = 0              # nodes popped then pruned by incumbent bound
        pruned_child_by_bound = 0            # children not pushed due to bound >= incumbent
        pruned_child_relaxed_infeasible = 0  # children pruned by necessary relaxed infeasibility
        pushed_children = 0                  # children pushed to heap
        leaves_polished = 0                  # leaves where we polished continuous

        while heap and nodes < self.max_nodes:
            if (self.time_limit is not None) and (time.time() - t_start > self.time_limit):
                if self.verbose:
                    print("[H-BNB] time limit reached.")
                break

            bound, _, depth, fix_mask, fix_vals = heapq.heappop(heap)
            nodes += 1
            if self.verbose and nodes % 5000 == 0:
                print(f"[H-BNB] pop node #{nodes}, bound={bound:.6g}, best={val_best:.6g}")

            # Prune by incumbent (bounding technique)
            if bound >= val_best - tol_prune:
                pruned_by_bound_pop += 1
                continue

            # Branch or evaluate leaf
            free_idx = np.where(~fix_mask)[0]
            if free_idx.size == 0:
                # Leaf: all binaries fixed -> polish continuous
                xd_leaf = torch.zeros(Dd, dtype=self.space.dtype, device=self.space.device)
                for i in range(Dd):
                    xd_leaf[i] = 1.0 if fix_vals[np.where(fix_mask)[0] == i].size and fix_vals[np.where(fix_mask)[0] == i][0] > 0.5 else 0.0
                # Build xd from mask/vals faster:
                xd_leaf = torch.zeros(Dd, dtype=self.space.dtype, device=self.space.device)
                if fix_mask.any():
                    idxs = np.where(fix_mask)[0]
                    xd_leaf[idxs.tolist()] = torch.tensor(fix_vals, dtype=self.space.dtype, device=self.space.device)

                # Start xc from incumbent's xc (or xc0) and polish
                xc_leaf = _project_cont_given_xd(self.space, xc_best.clone(), xd_leaf)
                try:
                    xc_leaf = _polish_continuous(self.cont_oracle, self.space, self.phi_d, self.phi_c, self.phi_m,
                                                 wd, wc, wm, xd_leaf, xc_leaf, mode="refine")
                except Exception:
                    pass
                # Project and validate
                xc_leaf = _project_cont_given_xd(self.space, xc_leaf, xd_leaf)
                if _disc_feasible_given_xc(self.space, xd_leaf, xc_leaf) and self.space.mixed_feasible(xd_leaf, xc_leaf):
                    leaves_polished += 1
                    v = _acq_value(self.phi_d, self.phi_c, self.phi_m, wd, wc, wm, xd_leaf, xc_leaf)
                    if v < val_best - tol:
                        val_best = v
                        xd_best, xc_best = xd_leaf.clone(), xc_leaf.clone()
                continue

            # Choose a branching variable (free unary index with highest score)
            # Score on raw binary index 'k' -> its row in φ_d is unary_idx[k]
            _, unary_rows, _ = self._indices_d()
            scores = []
            for k in free_idx:
                row_idx = unary_rows[k]
                scores.append((self._branch_score(row_idx, wd_np, WM_np, cos_cols), k))
            scores.sort(reverse=True)
            k_branch = scores[0][1]

            # Create children: set xd[k_branch] = 0 and 1
            for bit in (0.0, 1.0):
                child_mask = fix_mask.copy()
                child_vals = fix_vals.copy()
                # append k_branch to fixed sets
                new_mask = child_mask
                new_vals = None
                if child_mask.any():
                    idxs = np.where(child_mask)[0].tolist()
                    idxs.append(k_branch)
                    idxs = np.array(sorted(idxs), dtype=int)
                    # rebuild arrays in sorted order
                    old_pairs = dict(zip(np.where(child_mask)[0].tolist(), child_vals.tolist()))
                    old_pairs[k_branch] = bit
                    new_mask = np.zeros(Dd, dtype=bool)
                    new_mask[idxs] = True
                    new_vals = np.array([old_pairs[i] for i in idxs], dtype=float)
                else:
                    new_mask = np.zeros(Dd, dtype=bool)
                    new_mask[k_branch] = True
                    new_vals = np.array([bit], dtype=float)

                # Relaxed feasibility (necessary)
                if not self._necessary_relaxed_feasible(new_mask, new_vals):
                    pruned_child_relaxed_infeasible += 1
                    continue

                # H1 bound at child
                lin_lb = self._linear_lower_bound(wd_np, wm_bias_col, new_mask, new_vals) + const_linear
                C2 = self._C2max(wc_np, WM_np, new_mask, new_vals, cos_cols)
                h1 = lin_lb - C2 * _mixed_scale(depth + 1)

                if h1 < val_best - tol_prune:
                    heapq.heappush(heap, (h1, seq, depth + 1, new_mask, new_vals))
                    seq += 1
                    pushed_children += 1
                else:
                    pruned_child_by_bound += 1

        if self.verbose:
            elapsed = time.time() - t_start
            print(f"[H-BNB] done: nodes_explored={nodes}, best={val_best:.6g}, time={elapsed:.3f}s")
            print(
                f"[H-BNB] pruned_by_bound_pop={pruned_by_bound_pop}, "
                f"pruned_child_by_bound={pruned_child_by_bound}, pruned_child_relaxed_infeasible={pruned_child_relaxed_infeasible}, "
                f"pushed_children={pushed_children}, leaves={leaves_polished}"
            )
        return xd_best, xc_best, float(val_best)
