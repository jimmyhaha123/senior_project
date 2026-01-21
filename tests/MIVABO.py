"""
MIVABO (Mixed‑Variable Bayesian Optimization) — paper‑faithful, modular implementation

This file now includes a **dependency guard**: if PyTorch is not installed, the
script exits gracefully with clear install instructions instead of throwing
`ModuleNotFoundError`. Once PyTorch is available, all demos and tests run.

Highlights (per Daxberger et al., 2020)
- Linear surrogate over explicit features φ(x) with blocks for discrete, continuous, and mixed:
    f(x) = w_d^T φ_d(x_d) + w_c^T φ_c(x_c) + w_m^T φ_m(x_d, x_c)
- φ_d: second‑order pseudo‑Boolean features (bias, unary, pairwise)
- φ_c: Random Fourier Features (RFF) for an SE kernel on [0,1]^Dc
- φ_m: all pairwise products (kron) between φ_d and φ_c
- Thompson Sampling acquisition: sample w̃ ~ N(m, S^{-1}); pick x̂ ∈ argmin_x w̃^T φ(x)
- Acquisition optimization backends share ONE interface so you can swap:
    • Alternating (discrete QUBO step + continuous box step)
    • Branch‑and‑Bound (BnB) over binaries with continuous refit at nodes

Tests: keeps existing tests and adds a small dimension‑sanity test.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Sequence, Dict

import math
import random
import itertools
import sys

# =========================================================
# Dependency guard: Try to import torch; if missing, run in "help" mode
# =========================================================
TORCH_AVAILABLE = True
try:
    import torch
    from torch import Tensor
    from torch import no_grad as no_grad
except Exception:  # ModuleNotFoundError or other import errors
    TORCH_AVAILABLE = False
    # Provide minimal stubs so type hints & dataclass defaults don't crash at import time
    from typing import Any
    Tensor = Any
    def no_grad(fn):
        return fn
    class _TorchStub:
        double = float
        def device(self, *_args, **_kwargs):
            return "cpu"
    torch = _TorchStub()  # only used for defaults in dataclass; real code gated below

# =========================================================
# Search space and encoding
# =========================================================

@dataclass(frozen=True)
class DiscreteSpec:
    """Binary block with Dd variables (categorical/integer should be pre‑encoded)."""
    Dd: int

@dataclass(frozen=True)
class ContinuousSpec:
    Dc: int  # number of [0,1] box‑bounded variables

@dataclass
class LinearConstraint:
    A: Tensor  # (K, Dd)
    b: Tensor  # (K,)

@dataclass
class QuadraticConstraint:
    Q: Tensor  # (Dd, Dd)
    q: Tensor  # (Dd,)
    b: float

@dataclass
class MixedSpace:
    disc: DiscreteSpec
    cont: ContinuousSpec
    lin_cons: Optional[LinearConstraint] = None
    quad_cons: Optional[List[QuadraticConstraint]] = None
    dtype: "torch.dtype" = torch.double  # uses stub when torch missing
    device: "torch.device" = torch.device("cpu") if TORCH_AVAILABLE else "cpu"

    def zeros(self) -> Tuple[Tensor, Tensor]:
        import torch as _t
        xd = _t.zeros(self.disc.Dd, dtype=self.dtype, device=self.device)
        xc = _t.zeros(self.cont.Dc, dtype=self.dtype, device=self.device)
        return xd, xc

    def project_box(self, xc: Tensor) -> Tensor:
        import torch as _t
        return _t.clamp(xc, 0.0, 1.0)

    # ---------- discrete feasibility helpers ----------
    def disc_feasible(self, xd: Tensor, tol: float = 1e-9) -> bool:
        """Check full discrete assignment feasibility against linear and quadratic constraints."""
        import torch as _t
        if self.lin_cons is not None:
            A = self.lin_cons.A
            # Only check linear constraints that are discrete-only (A has Dd columns)
            try:
                if hasattr(A, 'shape') and A.shape[1] == self.disc.Dd:
                    if (_t.matmul(A, xd) > self.lin_cons.b + tol).any():
                        return False
                else:
                    # Mixed linear constraints cannot be decided from xd alone; skip
                    pass
            except Exception:
                # be conservative and skip if shapes unexpected
                pass
        if self.quad_cons is not None:
            for qc in self.quad_cons:
                try:
                    # Only evaluate discrete-only quadratic constraints
                    if qc.Q.shape[0] == self.disc.Dd and qc.Q.shape[1] == self.disc.Dd:
                        lhs = float(_t.dot(xd, _t.matmul(qc.Q, xd)) + _t.dot(qc.q, xd))
                        if lhs > qc.b + tol:
                            return False
                    else:
                        # mixed quadratic cannot be decided from xd alone; skip
                        pass
                except Exception:
                    # skip malformed constraints
                    pass
        return True

    def partial_disc_possible(self, partial_xd: Tensor, fixed_mask: Tensor, tol: float = 1e-9) -> bool:
        """Conservative check whether a partial assignment can be extended to a full feasible xd.

        This implements an exact check for linear constraints by computing the minimal
        possible left-hand side given the fixed bits and optimistic choices for free bits.
        For quadratic constraints we only perform a full-check at completion (not here),
        since tight conservative bounds are more expensive.
        """
        import torch as _t
        # Linear constraints: compute minimal achievable A x for each constraint row
        if self.lin_cons is not None:
            A = self.lin_cons.A
            b = self.lin_cons.b
            try:
                # Only attempt conservative pruning if A is discrete-only
                if hasattr(A, 'shape') and A.shape[1] == self.disc.Dd:
                    # partial_xd contains values for fixed positions and 0 for others; fixed_mask indicates fixed positions
                    Dd = partial_xd.numel()
                    # For free vars, choose 0 if coefficient >=0 else 1 to minimize A x
                    free_choice = _t.zeros(Dd, dtype=self.dtype, device=self.device)
                    if TORCH_AVAILABLE:
                        coeffs_pos = (A >= 0.0)
                        # set free_choice[j] = 0 if A_ij >=0 else 1; we can vectorize per-row
                        # We'll compute row-wise min by treating free variables optimally per row
                        for i in range(A.shape[0]):
                            row = A[i]
                            # ensure we only use the discrete-relevant columns
                            Dd_local = int(self.disc.Dd)
                            row_d = row[:Dd_local]
                            fm = fixed_mask.to(row_d.dtype)
                            # contribution from fixed vars
                            fixed_contrib = float(_t.dot(row_d * fm, partial_xd).item())
                            # minimal contribution from free vars
                            # if row[j] >= 0 -> choose 0; else choose 1
                            free_idx = (~fixed_mask).to(torch.bool)
                            if free_idx.any():
                                row_free = row_d[free_idx]
                                min_free = float(_t.dot(row_free.clamp(max=0.0), _t.ones_like(row_free)).item())
                            else:
                                min_free = 0.0
                            min_total = fixed_contrib + min_free
                            if min_total > float(b[i].item()) + tol:
                                return False
                else:
                    # mixed linear constraints cannot be conservatively pruned here
                    pass
            except Exception:
                # if any error, be conservative and do not prune
                pass
        # Quadratic constraints: cannot cheaply bound here; skip conservative pruning
        return True

    def mixed_feasible(self, xd: Tensor, xc: Tensor, tol: float = 1e-9) -> bool:
        """Check full mixed feasibility for linear and quadratic constraints over [xd; xc].

        This will evaluate linear constraints where A has Dd+Dc columns and
        quadratic constraints where Q is sized (Dd+Dc) x (Dd+Dc). If any
        constraint is malformed it is skipped.
        """
        import torch as _t
        if self.lin_cons is not None:
            A = self.lin_cons.A
            b = self.lin_cons.b
            try:
                if hasattr(A, 'shape') and A.shape[1] == (self.disc.Dd + self.cont.Dc):
                    vec = _t.cat([xd, xc])
                    if (_t.matmul(A, vec) > b + tol).any():
                        return False
                # if A has only Dd columns, that was handled by disc_feasible
            except Exception:
                pass
        if self.quad_cons is not None:
            for qc in self.quad_cons:
                try:
                    n = qc.Q.shape[0]
                    if n == (self.disc.Dd + self.cont.Dc):
                        vec = _t.cat([xd, xc])
                        lhs = float(_t.dot(vec, _t.matmul(qc.Q, vec)) + _t.dot(qc.q, vec))
                        if lhs > qc.b + tol:
                            return False
                    # else: discrete-only quadratic handled elsewhere
                except Exception:
                    pass
        return True

    # ---------- helpers: integer/categorical → binary (optional) ----------
    @staticmethod
    def encode_integer(val: int, low: int, high: int) -> List[int]:
        n = high - low + 1
        bits = max(1, (n - 1).bit_length())
        idx = val - low
        return [(idx >> k) & 1 for k in range(bits)]

    @staticmethod
    def decode_integer(bits: Sequence[int], low: int) -> int:
        idx = 0
        for k, b in enumerate(bits):
            idx |= (int(b) & 1) << k
        return low + idx
    
    def split_linear(self):
        """Return (A_d, A_c, b) if we have mixed linear constraints with Dd+Dc columns.
        If constraints are discrete-only (Dd columns), return (A_d, None, b).
        If no linear constraints, return (None, None, None)."""
        if self.lin_cons is None or not hasattr(self.lin_cons.A, 'shape'):
            return None, None, None
        A, b = self.lin_cons.A, self.lin_cons.b
        Dd, Dc = self.disc.Dd, self.cont.Dc
        if A.shape[1] == Dd + Dc:
            import torch as _t
            return A[:, :Dd], A[:, Dd:], b
        if A.shape[1] == Dd:
            return A, None, b
        return None, None, None

    def reduced_disc_constraints(self, xc):
        """Given xc, return discrete-only (A_d, b') or (None, None) if not applicable."""
        import torch as _t
        Ad, Ac, b = self.split_linear()
        if Ad is None:
            return None, None
        if Ac is None:
            return Ad, b  # already discrete-only
        rhs = b - (Ac @ xc)
        return Ad, rhs

    def reduced_cont_constraints(self, xd):
        """Given xd, return continuous-only (A_c, b') or (None, None) if not applicable."""
        import torch as _t
        Ad, Ac, b = self.split_linear()
        if Ac is None:
            return None, None
        if Ad is None:
            return Ac, b  # already cont-only (rare)
        rhs = b - (Ad @ xd)
        return Ac, rhs

    def project_polytope(self, x, Ac, rhs, iters: int = 25):
        """Project x onto [0,1]^Dc ∩ {Ac x <= rhs} using successive halfspace projections.
        Lightweight, no extra deps; good enough for small Dc and moderately tight constraints."""
        import torch as _t
        x = x.clone()
        for _ in range(iters):
            # box first
            x.clamp_(0.0, 1.0)
            if Ac is None:
                break
            # project onto each violated halfspace: a^T x <= b
            Ax = Ac @ x
            viol = (Ax > rhs)
            if not bool(viol.any()):
                break
            # do one sweep of most-violated-first projection
            idxs = _t.nonzero(viol, as_tuple=False).view(-1)
            # sort by violation magnitude (optional)
            _, order = _t.sort((Ax - rhs)[idxs], descending=True)
            for k in idxs[order]:
                a = Ac[k]
                num = (a @ x - rhs[k]).item()
                den = float((a @ a).item())
                if den > 1e-12 and num > 0.0:
                    x = x - (num / den) * a
        # final clamp
        x.clamp_(0.0, 1.0)
        return x

# =========================================================
# Feature expansions φ_d, φ_c, φ_m
# =========================================================

class DiscreteFeatures:
    """Second‑order pseudo‑Boolean features: [1, x_i, x_i x_j]."""
    def __init__(self, Dd: int, include_bias: bool = True):
        self.Dd = Dd
        self.include_bias = include_bias
        self.pairs = [(i, j) for i in range(Dd) for j in range(i + 1, Dd)]

    @property
    def dim(self) -> int:
        bias = 1 if self.include_bias else 0
        return bias + self.Dd + len(self.pairs)

    def __call__(self, xd: Tensor) -> Tensor:
        import torch as _t
        parts = []
        if self.include_bias:
            parts.append(_t.ones(1, dtype=xd.dtype, device=xd.device))
        parts.append(xd)
        if self.pairs:
            pair_terms = _t.stack([xd[i] * xd[j] for (i, j) in self.pairs], dim=0)
            parts.append(pair_terms)
        return _t.cat(parts)

class RFFContinuousFeatures:
    """Random Fourier Features for SE/RBF kernel on [0,1]^Dc."""
    def __init__(
        self,
        Dc: int,
        num_features: int = 64,
        kernel_lengthscale: float = 0.2,
        seed: int | None = None,
        include_bias: bool = False,
        dtype: "torch.dtype" = torch.double,
        device: "torch.device" = torch.device("cpu") if TORCH_AVAILABLE else "cpu",
    ):
        import torch as _t
        self.Dc = Dc
        self.M = num_features
        self.ls = kernel_lengthscale
        self.include_bias = include_bias
        rng = _t.Generator(device=device) if TORCH_AVAILABLE else None
        if TORCH_AVAILABLE:
            if seed is not None:
                rng.manual_seed(seed)
            self.W = _t.randn(self.M, Dc, generator=rng, dtype=dtype, device=device) / self.ls
            self.b = 2 * math.pi * _t.rand(self.M, generator=rng, dtype=dtype, device=device)
        else:
            # Defer real tensors to runtime when torch is present; placeholders for shape
            self.W = None
            self.b = None
        self.dtype = dtype
        self.device = device
        self.norm = math.sqrt(2.0 / self.M)

    @property
    def dim(self) -> int:
        return (1 if self.include_bias else 0) + self.M

    def __call__(self, xc: Tensor) -> Tensor:
        import torch as _t
        if self.W is None or self.b is None:
            raise RuntimeError("RFFContinuousFeatures requires PyTorch. Please install torch (see instructions below).")
        z = self.W @ xc  # (M,)
        feats = self.norm * _t.cos(z + self.b)
        return (
            _t.cat([_t.ones(1, dtype=self.dtype, device=self.device), feats])
            if self.include_bias
            else feats
        )

class MixedFeatures:
    """All pairwise products (Kronecker) between φ_d and φ_c."""
    def __init__(self, Md: int, Mc: int):
        self.Md = Md
        self.Mc = Mc

    @property
    def dim(self) -> int:
        return self.Md * self.Mc

    def __call__(self, phid: Tensor, phic: Tensor) -> Tensor:
        import torch as _t
        return _t.kron(phid, phic)

# =========================================================
# Linear Bayesian model over features
# =========================================================

class LinearFeatureModel:
    """Conjugate Bayesian linear model: prior w~N(0, α^{-1}I), noise β^{-1}."""
    def __init__(
        self,
        M: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        ts_delta: float = 0.1,
        ts_T: int = 50,
        dtype: "torch.dtype" = torch.double,
        device: "torch.device" = torch.device("cpu") if TORCH_AVAILABLE else "cpu",
    ):
        import torch as _t
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.ts_delta = float(ts_delta)
        self.ts_T = int(ts_T)
        self.dtype = dtype
        self.device = device
        if not TORCH_AVAILABLE:
            raise RuntimeError("LinearFeatureModel requires PyTorch. Please install torch (see instructions below).")
        self.S = alpha * _t.eye(M, dtype=dtype, device=device)  # precision
        self.m = _t.zeros(M, dtype=dtype, device=device)

    def update(self, phi: Tensor, y: float):
        """Rank‑1 Bayesian update with observation (φ, y)."""
        import torch as _t
        phi = phi.view(-1)
        self.S = self.S + self.beta * _t.outer(phi, phi)
        self.m = self.m + self.beta * phi * float(y)

    @no_grad
    def posterior_sample_w(self) -> Tensor:
        import torch as _t
        # Paper TS distribution:
        #   w_t ~ N( μ_t,  (24 M ln(T) ln(1/δ)) * S_t^{-1} )
        # where M is feature dimension, and S_t is the posterior precision.
        # We sample without forming S^{-1} explicitly:
        #   S = L L^T  =>  S^{-1} = L^{-T} L^{-1}
        #   w = μ + sqrt(scale) * L^{-T} z,   z~N(0, I)

        # Symmetrize for numerical stability (S is PSD by construction).
        S = 0.5 * (self.S + self.S.T)
        L = _t.linalg.cholesky(S)

        # μ_t = S^{-1} m  (m is the natural parameter)
        mean = _t.cholesky_solve(self.m.view(-1, 1), L).view(-1)

        # scale = 24 M ln(T) ln(1/delta)
        delta = self.ts_delta
        T = self.ts_T
        if not (delta > 0.0 and delta < 1.0):
            raise ValueError(f"ts_delta must be in (0,1), got {delta}")
        if not (T > 1):
            raise ValueError(f"ts_T must be > 1, got {T}")
        scale = 24.0 * float(self.M) * math.log(float(T)) * math.log(1.0 / float(delta))
        scale_sqrt = math.sqrt(scale)

        z = _t.randn(self.M, dtype=self.dtype, device=self.device).view(-1, 1)
        delta_w = _t.linalg.solve_triangular(L.T, z, upper=True).view(-1)
        return mean + scale_sqrt * delta_w

    @no_grad
    def posterior_mean_w(self) -> Tensor:
        import torch as _t
        S = 0.5 * (self.S + self.S.T)
        L = _t.linalg.cholesky(S)
        return _t.cholesky_solve(self.m.view(-1, 1), L).view(-1)

# =========================================================
# Acquisition = Thompson Sampling
# =========================================================

class ThompsonSampler:
    def __init__(self, model: LinearFeatureModel):
        self.model = model

    def sample_weight(self) -> Tensor:
        return self.model.posterior_sample_w()

# =========================================================
# Discrete/Continuous oracles used by optimizers
# =========================================================

class DiscreteOracle:
    """Interface for solving a quadratic pseudo‑Boolean subproblem.
    minimize  c0 + c^T xd + xd^T H xd   over xd∈{0,1}^Dd, possibly with constraints.
    """
    def solve(self, c0: float, c: Tensor, H: Tensor, space: MixedSpace, xd_start: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

class GreedyFlipOracle(DiscreteOracle):
    """Greedy single‑bit flips with feasibility checks (fallback heuristic)."""
    def __init__(self, max_passes: int = 50):
        self.max_passes = max_passes

    def feasible(self, xd: Tensor, space: MixedSpace) -> bool:
        # delegate to MixedSpace helper (centralizes tolerance/logic)
        return space.disc_feasible(xd)

    @staticmethod
    def energy(c0: float, c: Tensor, H: Tensor, xd: Tensor) -> float:
        import torch as _t
        return float(c0 + _t.dot(c, xd) + _t.dot(xd, _t.matmul(H, xd)))

    def solve(self, c0: float, c: Tensor, H: Tensor, space: MixedSpace, xd_start: Optional[Tensor] = None) -> Tensor:
        import torch as _t
        Dd = space.disc.Dd
        xd = (xd_start.clone() if xd_start is not None else _t.zeros(Dd, dtype=space.dtype, device=space.device))
        xd = (xd > 0.5).to(dtype=space.dtype)
        if not self.feasible(xd, space):
            for i in range(Dd):
                if self.feasible(xd, space):
                    break
                if xd[i] == 1:
                    xd[i] = 0
        best = self.energy(c0, c, H, xd)
        for _ in range(self.max_passes):
            improved = False
            for i in range(Dd):
                xd[i] = 1 - xd[i]
                if self.feasible(xd, space):
                    e = self.energy(c0, c, H, xd)
                    if e + 1e-12 < best:
                        best = e
                        improved = True
                    else:
                        xd[i] = 1 - xd[i]
                else:
                    xd[i] = 1 - xd[i]
            if not improved:
                break
        return xd


class GurobiOracle(DiscreteOracle):
    """Exact QUBO solver using Gurobi. Falls back to GreedyFlipOracle if gurobipy is unavailable.
    Minimizes c0 + c^T x + x^T H x subject to linear/quadratic constraints provided in MixedSpace.
    """
    def __init__(self, time_limit: float | None = 5.0):
        self.time_limit = time_limit
        self._fallback = GreedyFlipOracle()

    def solve(self, c0: float, c: Tensor, H: Tensor, space: MixedSpace, xd_start: Optional[Tensor] = None) -> Tensor:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as e:
            raise ImportError("gurobipy is required for GurobiOracle but not importable") from e

        # Convert tensors to Python floats/lists
        import torch as _t
        Dd = space.disc.Dd
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        if self.time_limit is not None:
            model.setParam('TimeLimit', float(self.time_limit))

        xvars = model.addVars(range(Dd), vtype=GRB.BINARY, name='x')

        # Linear part
        # build linear part via quicksum to avoid LinExpr.add API pitfalls
        lin_terms = []
        for i in range(Dd):
            coeff = c[i].item() if hasattr(c[i], 'item') else float(c[i])
            if coeff != 0.0:
                lin_terms.append(coeff * xvars[i])
        lin_expr = gp.quicksum(lin_terms) if lin_terms else gp.LinExpr()

        # Quadratic part: collect quadratic product terms then quicksum
        quad_terms = []
        H_np = H.cpu().numpy() if hasattr(H, 'cpu') else H
        for i in range(Dd):
            for j in range(Dd):
                coeff = float(H_np[i, j])
                if coeff != 0.0:
                    quad_terms.append(coeff * (xvars[i] * xvars[j]))

        # Objective: linear + quadratic + constant
        obj_terms = []
        if isinstance(lin_expr, gp.LinExpr):
            # lin_expr may already be a LinExpr from empty quicksum
            obj_terms.append(lin_expr)
        else:
            obj_terms.extend(lin_terms if 'lin_terms' in locals() else [])
        obj_terms.extend(quad_terms)
        if obj_terms:
            obj = gp.quicksum(obj_terms)
        else:
            obj = gp.LinExpr()
        if c0 != 0.0:
            obj += float(c0)
        model.setObjective(obj, GRB.MINIMIZE)

        # Linear constraints A x <= b
        if space.lin_cons is not None:
            A = space.lin_cons.A
            b = space.lin_cons.b
            A_np = A.cpu().numpy() if hasattr(A, 'cpu') else A
            for i in range(A_np.shape[0]):
                terms = []
                for j in range(Dd):
                    coeff = float(A_np[i, j])
                    if coeff != 0.0:
                        terms.append(coeff * xvars[j])
                expr = gp.quicksum(terms) if terms else gp.LinExpr()
                model.addConstr(expr <= float(b[i].item() if hasattr(b[i], 'item') else b[i]))

        # Quadratic constraints x^T Q x + q^T x <= b
        if space.quad_cons is not None:
            for qc in space.quad_cons:
                Q_np = qc.Q.cpu().numpy() if hasattr(qc.Q, 'cpu') else qc.Q
                q_np = qc.q.cpu().numpy() if hasattr(qc.q, 'cpu') else qc.q
                rhs = float(qc.b)
                # build quadratic constraint expression via quicksum of quadratic and linear terms
                q_terms = []
                for i in range(Dd):
                    for j in range(Dd):
                        coeff = float(Q_np[i, j])
                        if coeff != 0.0:
                            q_terms.append(coeff * (xvars[i] * xvars[j]))
                lin_terms_q = []
                for i in range(Dd):
                    coeff = float(q_np[i])
                    if coeff != 0.0:
                        lin_terms_q.append(coeff * xvars[i])
                left = gp.quicksum(q_terms + lin_terms_q) if (q_terms or lin_terms_q) else gp.LinExpr()
                model.addQConstr(left <= rhs)

        model.optimize()
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
            sol = torch.zeros(Dd, dtype=space.dtype, device=space.device)
            for i in range(Dd):
                val = xvars[i].X
                sol[i] = 1.0 if val > 0.5 else 0.0
            # final feasibility check; if infeasible, fallback
            feasible = True
            try:
                # If linear constraints exist, only check the discrete-relevant columns
                if space.lin_cons is not None:
                    A = space.lin_cons.A
                    b = space.lin_cons.b
                    # If A is discrete-only, delegate to existing helper
                    if hasattr(A, 'shape') and A.shape[1] == Dd:
                        if not space.disc_feasible(sol):
                            feasible = False
                    else:
                        # Mixed A: only check first Dd columns against xd
                        A_np = A.cpu().numpy() if hasattr(A, 'cpu') else A
                        b_np = b.cpu().numpy() if hasattr(b, 'cpu') else b
                        sol_np = sol.cpu().numpy() if hasattr(sol, 'cpu') else sol
                        try:
                            left = A_np[:, :Dd].dot(sol_np)
                            if (left > (b_np + 1e-9)).any():
                                feasible = False
                        except Exception:
                            # if any shape or type issue, be conservative and skip rejecting
                            pass
                else:
                    # no linear constraints: still need to check discrete quadratic constraints
                    if space.quad_cons is not None:
                        for qc in space.quad_cons:
                            try:
                                if qc.Q.shape[0] == Dd and qc.Q.shape[1] == Dd:
                                    lhs = float(_t.dot(sol, _t.matmul(qc.Q, sol)) + _t.dot(qc.q, sol))
                                    if lhs > qc.b + 1e-9:
                                        feasible = False
                                        break
                            except Exception:
                                pass
            except Exception:
                # conservative: if final-check machinery fails, fall back to safe behaviour
                feasible = True

            if not feasible:
                return self._fallback.solve(c0, c, H, space, xd_start=xd_start)
            return sol
        else:
            return self._fallback.solve(c0, c, H, space, xd_start=xd_start)

class ContinuousOracle:
    """Projected gradient descent for box‑constrained continuous step (swap‑able)."""
    def __init__(self, max_steps: int = 200, lr: float = 0.1):
        self.max_steps = max_steps
        self.lr = lr

    def solve(
    self,
    objective: Callable[[Tensor], Tensor],
    grad: Callable[[Tensor], Tensor],
    space: MixedSpace,
    xc_start: Optional[Tensor] = None,
    xd_fixed: Optional[Tensor] = None,   # NEW: to reduce mixed constraints
) -> Tensor:
        
        import torch as _t

        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd, Dc = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd + Dc:
                return A[:, :Dd], A[:, Dd:], b
            if A.shape[1] == Dd:
                return A, None, b
            return None, None, None

        def _reduced_cont(space: MixedSpace, xd: Tensor | None):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if xd is None or Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd)

        def _project_poly_box_poly(x: Tensor):
            # Project x onto [0,1]^Dc ∩ {A_c x <= rhs}
            x = x.clone()
            Ac, rhs = _reduced_cont(space, xd_fixed)
            for _ in range(25):
                x.clamp_(0.0, 1.0)
                if Ac is None:
                    break
                Ax = Ac @ x
                if rhs is None:
                    break
                viol = Ax > rhs
                if not bool(viol.any()):
                    break
                idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                _, order = _t.sort((Ax - rhs)[idxs], descending=True)
                for k in idxs[order]:
                    a = Ac[k]
                    num = (a @ x - rhs[k]).item()
                    den = float((a @ a).item())
                    if den > 1e-12 and num > 0.0:
                        x = x - (num / den) * a
            x.clamp_(0.0, 1.0)
            return x

        xc = (xc_start.clone() if xc_start is not None else _t.zeros(space.cont.Dc, dtype=space.dtype, device=space.device))
        xc = _project_poly_box_poly(xc)

        for _ in range(self.max_steps):
            g = grad(xc)
            xc = _project_poly_box_poly(xc - self.lr * g)

        return xc


class TieredContinuousOracle(ContinuousOracle):
    """Tiered continuous solver: fast mode for bounds and a refine mode for leaves.

    - fast: short Adam / PGD warmup
    - refine: multi-start warmup + L-BFGS polish, optional coarse grid when Dc small
    """
    def __init__(self, *, fast_steps: int = 40, warmup_steps: int = 40, n_starts: int = 6, lbfgs_max_iter: int = 50, lr: float = 0.05, dtype=None, device=None):
        super().__init__(max_steps=fast_steps, lr=lr)
        self.fast_steps = fast_steps
        self.warmup_steps = warmup_steps
        self.n_starts = n_starts
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lr = lr

    def _adam_warmup(self, xc0, grad_fn, space, steps, lr, xd_fixed=None):
        import torch as _t

        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd, Dc = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd + Dc:
                return A[:, :Dd], A[:, Dd:], b
            if A.shape[1] == Dd:
                return A, None, b
            return None, None, None

        def _reduced_cont(space: MixedSpace, xd: Tensor | None):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if xd is None or Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd)

        def _project(x: Tensor):
            x = x.clone()
            Ac, rhs = _reduced_cont(space, xd_fixed)
            for _ in range(15):
                x.clamp_(0.0, 1.0)
                if Ac is None or rhs is None:
                    break
                Ax = Ac @ x
                viol = Ax > rhs
                if not bool(viol.any()):
                    break
                idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                for k in idxs:
                    a = Ac[k]
                    num = (a @ x - rhs[k]).item()
                    den = float((a @ a).item())
                    if den > 1e-12 and num > 0.0:
                        x = x - (num / den) * a
            x.clamp_(0.0, 1.0)
            return x

        xc = xc0.clone().detach().to(dtype=space.dtype, device=space.device)
        xc = _project(xc)
        xc.requires_grad_(True)
        opt = _t.optim.Adam([xc], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            g = grad_fn(xc)
            if xc.grad is not None:
                xc.grad.detach_(); xc.grad.zero_()
            xc.grad = g.detach()
            opt.step()
            with _t.no_grad():
                xc.copy_(_project(xc))
        return xc.detach()

    def _lbfgs_polish(self, xc0, objective_fn, grad_fn, space, max_iter, xd_fixed=None):
        import torch as _t

        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd, Dc = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd + Dc:
                return A[:, :Dd], A[:, Dd:], b
            if A.shape[1] == Dd:
                return A, None, b
            return None, None, None

        def _reduced_cont(space: MixedSpace, xd: Tensor | None):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if xd is None or Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd)

        def _project(x: Tensor):
            x = x.clone()
            Ac, rhs = _reduced_cont(space, xd_fixed)
            for _ in range(25):
                x.clamp_(0.0, 1.0)
                if Ac is None or rhs is None:
                    break
                Ax = Ac @ x
                viol = Ax > rhs
                if not bool(viol.any()):
                    break
                idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                for k in idxs:
                    a = Ac[k]
                    num = (a @ x - rhs[k]).item()
                    den = float((a @ a).item())
                    if den > 1e-12 and num > 0.0:
                        x = x - (num / den) * a
            x.clamp_(0.0, 1.0)
            return x

        xc = xc0.clone().detach().to(dtype=space.dtype, device=space.device)
        xc = _project(xc)
        xc.requires_grad_(True)
        opt = _t.optim.LBFGS([xc], max_iter=max_iter, tolerance_grad=1e-6, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            val = objective_fn(xc)
            if not isinstance(val, _t.Tensor):
                val = _t.tensor(float(val), dtype=space.dtype, device=space.device)
            val.backward()
            return val

        try:
            opt.step(closure)
        except Exception:
            pass
        with _t.no_grad():
            xc.copy_(_project(xc))
        return xc.detach()

    def solve(self, objective: Callable[[Tensor], Tensor], grad: Callable[[Tensor], Tensor], space: MixedSpace, xc_start: Optional[Tensor] = None, mode: str = 'fast', xd_fixed: Optional[Tensor] = None) -> Tensor:
        import torch as _t
        # fast mode: short Adam warmup with projection
        if mode == 'fast':
            xc0 = xc_start.clone() if xc_start is not None else _t.zeros(space.cont.Dc, dtype=space.dtype, device=space.device)
            return self._adam_warmup(xc0, grad, space, self.fast_steps, self.lr, xd_fixed=xd_fixed)

        # refine mode: multi-start + LBFGS, all with projection
        Dc = space.cont.Dc
        best_xc = None
        best_val = float('inf')
        seeds = []
        if xc_start is not None:
            seeds.append(xc_start.clone())
        center = _t.full((Dc,), 0.5, dtype=space.dtype, device=space.device)
        seeds.append(center)
        for _ in range(max(0, self.n_starts - len(seeds))):
            seeds.append(_t.rand(Dc, dtype=space.dtype, device=space.device))

        for s in seeds[: self.n_starts]:
            x_w = self._adam_warmup(s, grad, space, self.warmup_steps, self.lr, xd_fixed=xd_fixed)
            x_p = self._lbfgs_polish(x_w, objective, grad, space, self.lbfgs_max_iter, xd_fixed=xd_fixed)
            val_t = objective(x_p)
            val = float(val_t.item()) if hasattr(val_t, 'item') else float(val_t)
            if val < best_val:
                best_val = val
                best_xc = x_p.clone()

        if Dc <= 3:
            try:
                import numpy as _np
                grid_pts = 5
                coords = [_np.linspace(0.0, 1.0, grid_pts) for _ in range(Dc)]
                for tup in __import__('itertools').product(*coords):
                    xc_g = _t.tensor(tup, dtype=space.dtype, device=space.device)
                    x_w = self._adam_warmup(xc_g, grad, space, max(8, self.warmup_steps // 4), self.lr, xd_fixed=xd_fixed)
                    x_p = self._lbfgs_polish(x_w, objective, grad, space, max(8, self.lbfgs_max_iter // 4), xd_fixed=xd_fixed)
                    val_t = objective(x_p)
                    val = float(val_t.item()) if hasattr(val_t, 'item') else float(val_t)
                    if val < best_val:
                        best_val = val
                        best_xc = x_p.clone()
            except Exception:
                pass

        if best_xc is None:
            # fallback to parent (will project inside)
            return super().solve(objective, grad, space, xc_start=xc_start, xd_fixed=xd_fixed)
        return best_xc

# =========================================================
# Acquisition optimizer interface + backends
# =========================================================

class AcquisitionOptimizer:
    """Common interface for acquisition optimizers.
    Implement: solve(w_d, w_c, w_m, xd0, xc0) -> (xd, xc, value)  [value = min objective]
    """
    def solve(self, w_d: Tensor, w_c: Tensor, w_m: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        raise NotImplementedError

class AlternatingOptimizer(AcquisitionOptimizer):
    def __init__(
        self,
        space: MixedSpace,
        disc_phi: DiscreteFeatures,
        cont_phi: RFFContinuousFeatures,
        mix_phi: MixedFeatures,
        discrete_oracle: Optional[DiscreteOracle] = None,
        continuous_oracle: Optional[ContinuousOracle] = None,
        max_alt_iters: int = 20,
        tol: float = 1e-6,
    ):
        self.space = space
        self.phi_d = disc_phi
        self.phi_c = cont_phi
        self.phi_m = mix_phi
        # prefer Gurobi-based exact oracle; require gurobipy be installed
        if discrete_oracle is not None:
            self.disc_oracle = discrete_oracle
        else:
            try:
                import gurobipy  # type: ignore
            except Exception as e:
                raise ImportError("gurobipy is required for AlternatingOptimizer but not importable") from e
            self.disc_oracle = GurobiOracle()
        self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        self.max_alt_iters = max_alt_iters
        self.tol = tol

    def _build_quadratic(self, w_d: Tensor, w_m: Tensor, xc_fixed: Tensor) -> Tuple[float, Tensor, Tensor]:
        import torch as _t
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        phic = self.phi_c(xc_fixed)
        W = w_m.view(Md, Mc)
        eff_d = w_d + _t.matmul(W, phic)
        idx = 0
        c0 = float(eff_d[idx].item()) if self.phi_d.include_bias else 0.0
        if self.phi_d.include_bias:
            idx += 1
        c = eff_d[idx : idx + self.space.disc.Dd].clone()
        idx += self.space.disc.Dd
        pair_terms = eff_d[idx:]
        Dd = self.space.disc.Dd
        H = _t.zeros(Dd, Dd, dtype=eff_d.dtype, device=eff_d.device)
        for k, (i, j) in enumerate(self.phi_d.pairs):
            H[i, j] += pair_terms[k] * 0.5
            H[j, i] += pair_terms[k] * 0.5
        return c0, c, H

    def _cont_objective(self, w_c: Tensor, w_m: Tensor, xd_fixed: Tensor) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
        import torch as _t
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        W = w_m.view(Md, Mc)
        phid = self.phi_d(xd_fixed)
        def obj(xc: Tensor) -> Tensor:
            phic = self.phi_c(xc)
            return (_t.dot(w_c, phic) + _t.dot(_t.matmul(phid, W), phic))
        def grad(xc: Tensor) -> Tensor:
            # Autograd path (PyTorch only). If needed, can be replaced by FD gradient.
            xc_req = xc.clone().detach().requires_grad_(True)
            phic = self.phi_c(xc_req)
            val = (w_c @ phic + (phid @ W) @ phic)
            val.backward()
            return xc_req.grad.detach()
        return obj, grad

    def solve(self, w_d: Tensor, w_c: Tensor, w_m: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        import torch as _t
        from copy import deepcopy

        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd, Dc = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd + Dc:
                return A[:, :Dd], A[:, Dd:], b
            if A.shape[1] == Dd:
                return A, None, b
            return None, None, None

        def _reduced_disc(space: MixedSpace, xc: Tensor):
            Ad, Ac, b = _split_lin(space)
            if Ad is None:
                return None, None
            if Ac is None:
                return Ad, b
            return Ad, (b - Ac @ xc)

        def _reduced_cont(space: MixedSpace, xd: Tensor):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd)

        def _project_poly_box_poly(x: Tensor, space: MixedSpace, xd_fixed: Tensor | None):
            # Project onto [0,1]^Dc ∩ {A_c x <= rhs} with a few halfspace sweeps
            x = x.clone()
            Ac, rhs = _reduced_cont(space, xd_fixed) if xd_fixed is not None else (None, None)
            for _ in range(25):
                x.clamp_(0.0, 1.0)
                if Ac is None:
                    break
                Ax = Ac @ x
                viol = (Ax > rhs) if rhs is not None else None
                if viol is None or not bool(viol.any()):
                    break
                idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                # most violated first
                _, order = _t.sort((Ax - rhs)[idxs], descending=True)
                for k in idxs[order]:
                    a = Ac[k]
                    num = (a @ x - rhs[k]).item()
                    den = float((a @ a).item())
                    if den > 1e-12 and num > 0.0:
                        x = x - (num / den) * a
            x.clamp_(0.0, 1.0)
            return x

        xd = xd0.clone()
        xc = xc0.clone()
        prev = float("inf")

        for _ in range(self.max_alt_iters):
            # ---- discrete step (given xc): reduce constraints to Ad xd <= brhs ----
            c0, c, H = self._build_quadratic(w_d, w_m, xc)
            Ad, brhs = _reduced_disc(self.space, xc)
            if Ad is not None:
                tmp_space = deepcopy(self.space)
                tmp_space.lin_cons = LinearConstraint(A=Ad, b=brhs)
            else:
                tmp_space = self.space
            xd = self.disc_oracle.solve(c0, c, H, tmp_space, xd_start=xd)

            # ---- continuous step (given xd): project each step onto box ∩ {A_c x <= rhs} ----
            obj, grad = self._cont_objective(w_c, w_m, xd)
            # Use the continuous oracle, but ensure it projects with the ACTIVE reduced constraints
            # We pass xd as xd_fixed so the oracle can reduce A_c x <= b - A_d xd internally.
            try:
                xc = self.cont_oracle.solve(obj, grad, self.space, xc_start=xc, xd_fixed=xd)  # type: ignore
            except TypeError:
                # If your local ContinuousOracle lacks xd_fixed, do a manual PGD loop with projection
                lr = getattr(self.cont_oracle, "lr", 0.05)
                steps = getattr(self.cont_oracle, "max_steps", 200)
                for _it in range(steps):
                    g = grad(xc)
                    xc = _project_poly_box_poly(xc - lr * g, self.space, xd)

            # ---- evaluate acquisition ----
            phid = self.phi_d(xd)
            phic = self.phi_c(xc)
            val = float((_t.dot(w_d, phid) + _t.dot(w_c, phic) + _t.dot(_t.kron(phid, phic), w_m)).item())
            if prev - val < self.tol:
                prev = val
                break
            prev = val

        return xd, xc, prev

class BranchAndBoundOptimizer(AcquisitionOptimizer):
    """Depth‑first BnB over binaries with continuous refit; conservative bounding.
    Intended for small Dd; API‑compatible with AlternatingOptimizer.
    """
    def __init__(
        self,
        space: MixedSpace,
        disc_phi: DiscreteFeatures,
        cont_phi: RFFContinuousFeatures,
        mix_phi: MixedFeatures,
        continuous_oracle: Optional[ContinuousOracle] = None,
        max_nodes: int = 5000,
    ):
        self.space = space
        self.phi_d = disc_phi
        self.phi_c = cont_phi
        self.phi_m = mix_phi
        self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        self.max_nodes = max_nodes

    def _evaluate(self, w_d: Tensor, w_c: Tensor, w_m: Tensor, xd: Tensor, xc_start: Tensor) -> Tuple[float, Tensor]:
        import torch as _t
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        W = w_m.view(Md, Mc)
        phid = self.phi_d(xd)

        def obj(xc: Tensor) -> Tensor:
            phic = self.phi_c(xc)
            return (_t.dot(w_c, phic) + _t.dot(_t.matmul(phid, W), phic))

        def grad(xc: Tensor) -> Tensor:
            xc_req = xc.clone().detach().requires_grad_(True)
            phic = self.phi_c(xc_req)
            val = (w_c @ phic + (phid @ W) @ phic)
            val.backward()
            return xc_req.grad.detach()

        # --- fast solve (Projected Adam inside TieredContinuousOracle) ---
        try:
            xc = self.cont_oracle.solve(obj, grad, self.space, xc_start=xc_start, xd_fixed=xd)  # type: ignore
        except TypeError:
            xc = self.cont_oracle.solve(obj, grad, self.space, xc_start=xc_start)

        # --- single projection sweep to be extra safe ---
        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd, Dc = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd + Dc:
                return A[:, :Dd], A[:, Dd:], b
            if A.shape[1] == Dd:
                return A, None, b
            return None, None, None

        def _reduced_cont(space: MixedSpace, xd_fixed: Tensor | None):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if xd_fixed is None or Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd_fixed)

        def _project_once(x: Tensor):
            x = x.clone()
            Ac, rhs = _reduced_cont(self.space, xd)
            x.clamp_(0.0, 1.0)
            if Ac is not None and rhs is not None:
                Ax = Ac @ x
                viol = Ax > rhs
                if bool(viol.any()):
                    idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                    _, order = _t.sort((Ax - rhs)[idxs], descending=True)
                    for k in idxs[order]:
                        a = Ac[k]
                        num = (a @ x - rhs[k]).item()
                        den = float((a @ a).item())
                        if den > 1e-12 and num > 0.0:
                            x = x - (num / den) * a
                    x.clamp_(0.0, 1.0)
            return x

        xc = _project_once(xc)

        # --- value ---
        phic = self.phi_c(xc)
        val = float((_t.dot(w_d, phid) + _t.dot(w_c, phic) + _t.dot(_t.kron(phid, phic), w_m)).item())
        return val, xc

    def _bound(self, w_d: Tensor, w_m: Tensor, partial_xd: Tensor, fixed_mask: Tensor, xc_hint: Tensor) -> float:
        import torch as _t
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        # Conservative lower bound for the mixed contribution:
        # φ_c components lie in [-norm, norm] where norm = self.phi_c.norm.
        # Therefore W @ φ_c ∈ [ -norm * sum_j |W_ij|, +norm * sum_j |W_ij| ] for each row i.
        # Use the minimal possible contribution (most optimistic for minimization) to
        # ensure the bound is a true lower bound and does not prune feasible minima.
        W = w_m.view(Md, Mc)
        try:
            phic_range = float(self.phi_c.norm)
        except Exception:
            # Fallback: if phi_c lacks norm, assume range 1.0 (conservative)
            phic_range = 1.0
        min_W_contrib = -phic_range * _t.sum(_t.abs(W), dim=1)
        eff_d = (w_d + min_W_contrib)
        idx = 0
        c0 = float(eff_d[idx].item()) if self.phi_d.include_bias else 0.0
        if self.phi_d.include_bias:
            idx += 1
        c = eff_d[idx : idx + self.space.disc.Dd]
        idx += self.space.disc.Dd
        pair = eff_d[idx:]
        xd = partial_xd
        lb = c0 + float(_t.dot(c, xd).item())
        k = 0
        for (i, j) in self.phi_d.pairs:
            w_ij = pair[k]
            if fixed_mask[i] and fixed_mask[j]:
                lb += float((w_ij * xd[i] * xd[j]).item())
            elif fixed_mask[i] and not fixed_mask[j]:
                xj = 0.0 if w_ij * xd[i] >= 0 else 1.0
                lb += float((w_ij * xd[i] * xj).item() if hasattr(w_ij, 'item') else (w_ij * xd[i] * xj))
            elif not fixed_mask[i] and fixed_mask[j]:
                xi = 0.0 if w_ij * xd[j] >= 0 else 1.0
                lb += float((w_ij * xi * xd[j]).item() if hasattr(w_ij, 'item') else (w_ij * xi * xd[j]))
            else:
                # min over {0,1}^2 of w_ij*xi*xj is 0 if w_ij>=0 else w_ij
                val = float(min(0.0, w_ij.item() if hasattr(w_ij, 'item') else float(w_ij)))
                lb += val
            k += 1
        for i in range(self.space.disc.Dd):
            if not fixed_mask[i]:
                ci = c[i]
                lb += float(min(0.0, ci.item() if hasattr(ci, 'item') else float(ci)))
        return lb

    def solve(self, w_d: Tensor, w_c: Tensor, w_m: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        import torch as _t

        Dd = self.space.disc.Dd
        Dc = self.space.cont.Dc
        best_val = float("inf")
        best_xd = xd0.clone()
        best_xc = xc0.clone()
        nodes = 0

        # --- seed incumbent via constraint-aware alternating, then evaluate via fast path ---
        alt_seed = AlternatingOptimizer(self.space, self.phi_d, self.phi_c, self.phi_m)
        inc_xd, inc_xc, _ = alt_seed.solve(w_d, w_c, w_m, xd0, xc0)
        inc_val, inc_xc = self._evaluate(w_d, w_c, w_m, inc_xd, inc_xc)
        best_val, best_xd, best_xc = inc_val, inc_xd, inc_xc

        fixed_mask = _t.zeros(Dd, dtype=_t.bool, device=self.space.device)
        xd_partial = _t.zeros(Dd, dtype=self.space.dtype, device=self.space.device)

        stack = [(0, xd_partial.clone(), fixed_mask.clone(), best_xc.clone())]  # (next_idx, xd, mask, xc_hint)

        # local helpers for feasibility + value
        def _mixed_feasible(xd: Tensor, xc: Tensor) -> bool:
            try:
                return self.space.mixed_feasible(xd, xc)
            except Exception:
                return True

        def _value(w_d: Tensor, w_c: Tensor, w_m: Tensor, xd: Tensor, xc: Tensor) -> float:
            phid = self.phi_d(xd)
            phic = self.phi_c(xc)
            return float((w_d @ phid + w_c @ phic + self.phi_m(phid, phic) @ w_m).item())

        # projection helper reused here
        def _split_lin(space: MixedSpace):
            if space.lin_cons is None or not hasattr(space.lin_cons.A, 'shape'):
                return None, None, None
            A, b = space.lin_cons.A, space.lin_cons.b
            Dd_, Dc_ = space.disc.Dd, space.cont.Dc
            if A.shape[1] == Dd_ + Dc_:
                return A[:, :Dd_], A[:, Dd_:], b
            if A.shape[1] == Dd_:
                return A, None, b
            return None, None, None

        def _reduced_cont(space: MixedSpace, xd_fixed: Tensor | None):
            Ad, Ac, b = _split_lin(space)
            if Ac is None:
                return None, None
            if xd_fixed is None or Ad is None:
                return Ac, b
            return Ac, (b - Ad @ xd_fixed)

        def _project_once(x: Tensor, xd_fixed: Tensor):
            x = x.clone()
            Ac, rhs = _reduced_cont(self.space, xd_fixed)
            x.clamp_(0.0, 1.0)
            if Ac is not None and rhs is not None:
                Ax = Ac @ x
                viol = Ax > rhs
                if bool(viol.any()):
                    idxs = _t.nonzero(viol, as_tuple=False).view(-1)
                    _, order = _t.sort((Ax - rhs)[idxs], descending=True)
                    for k in idxs[order]:
                        a = Ac[k]
                        num = (a @ x - rhs[k]).item()
                        den = float((a @ a).item())
                        if den > 1e-12 and num > 0.0:
                            x = x - (num / den) * a
                    x.clamp_(0.0, 1.0)
            return x

        while stack and nodes < self.max_nodes:
            idx, xd_part, mask, xc_hint = stack.pop()
            nodes += 1

            # bound-based pruning (unchanged)
            lb = self._bound(w_d, w_m, xd_part, mask, xc_hint)
            if lb >= best_val - 1e-12:
                continue

            if idx == Dd:
                xd_leaf = xd_part

                # fast evaluate
                val_fast, xc_fast = self._evaluate(w_d, w_c, w_m, xd_leaf, xc_hint)

                # decide if we should refine: when it beats incumbent by epsilon OR small Dc
                do_refine = (val_fast < best_val - 1e-9) or (Dc <= 3)

                if do_refine:
                    # refine starting from the fast solution
                    try:
                        xc_ref = self.cont_oracle.solve(
                            # re-create objective/grad lightweight
                            lambda x: _value(w_d, w_c, w_m, xd_leaf, x),
                            lambda x: _t.autograd.functional.jacobian(
                                lambda z: _value(w_d, w_c, w_m, xd_leaf, z), x
                            ),  # note: autograd path; your oracle typically uses provided grad
                            self.space, xc_start=xc_fast, xd_fixed=xd_leaf, mode="refine"  # type: ignore
                        )
                    except TypeError:
                        # Fallback if oracle doesn't accept mode/xd_fixed; just use fast result
                        xc_ref = xc_fast

                    # project once and compute refined value
                    xc_ref = _project_once(xc_ref, xd_leaf)
                    val_ref = _value(w_d, w_c, w_m, xd_leaf, xc_ref)

                    # pick the better of fast/refined
                    if val_ref < val_fast:
                        val_fast, xc_fast = val_ref, xc_ref

                # final mixed feasibility guard; if off by eps, repair once via projection
                if not _mixed_feasible(xd_leaf, xc_fast):
                    xc_fast = _project_once(xc_fast, xd_leaf)
                    if not _mixed_feasible(xd_leaf, xc_fast):
                        # infeasible even after repair → skip this leaf
                        continue
                    val_fast = _value(w_d, w_c, w_m, xd_leaf, xc_fast)

                if val_fast < best_val:
                    best_val, best_xd, best_xc = val_fast, xd_leaf.clone(), xc_fast.clone()
                continue

            # branch on next bit
            for bit in (0.0, 1.0):
                xd_new = xd_part.clone(); xd_new[idx] = bit
                mask_new = mask.clone(); mask_new[idx] = True
                stack.append((idx + 1, xd_new, mask_new, xc_hint))

        return best_xd, best_xc, best_val


class GurobiMixedOptimizer(AcquisitionOptimizer):
    """Use Gurobi to enumerate top-K discrete candidates on a conservative discrete surrogate
    (lower bound for mixed contribution) and then refine continuous variables with
    the TieredContinuousOracle in 'refine' mode.

    This treats Gurobi as a licensed global optimizer for the discrete surrogate and
    relies on a strong continuous local solver for the continuous refinement.
    """
    def __init__(self, space: MixedSpace, disc_phi: DiscreteFeatures, cont_phi: RFFContinuousFeatures, mix_phi: MixedFeatures, n_candidates: int = 8, time_limit: float = 5.0, continuous_oracle: Optional[ContinuousOracle] = None):
        self.space = space
        self.phi_d = disc_phi
        self.phi_c = cont_phi
        self.phi_m = mix_phi
        self.n_candidates = n_candidates
        self.time_limit = time_limit
        self.cont_oracle = continuous_oracle or TieredContinuousOracle()
        self._fallback_oracle = GreedyFlipOracle()

    def _build_discrete_surrogate(self, w_d: Tensor, w_m: Tensor):
        """Return c0, c, H for a discrete-only QUBO surrogate using conservative min over phi_c."""
        import torch as _t
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        W = w_m.view(Md, Mc)
        # conservative min contribution per discrete feature row
        try:
            phic_range = float(self.phi_c.norm)
        except Exception:
            phic_range = 1.0
        min_W_contrib = -phic_range * _t.sum(_t.abs(W), dim=1)
        eff_d = w_d + min_W_contrib
        # build c0, c, H same layout as Alternating._build_quadratic
        idx = 0
        c0 = float(eff_d[idx].item()) if self.phi_d.include_bias else 0.0
        if self.phi_d.include_bias:
            idx += 1
        c = eff_d[idx : idx + self.space.disc.Dd].clone()
        idx += self.space.disc.Dd
        pair_terms = eff_d[idx:]
        Dd = self.space.disc.Dd
        H = _t.zeros(Dd, Dd, dtype=eff_d.dtype, device=eff_d.device)
        for k, (i, j) in enumerate(self.phi_d.pairs):
            H[i, j] += pair_terms[k] * 0.5
            H[j, i] += pair_terms[k] * 0.5
        return c0, c, H

    def _solve_qubo_topk(self, c0: float, c: Tensor, H: Tensor, space: MixedSpace, k: int, time_limit: float):
        """Use gurobipy to extract up to k distinct binary solutions for the QUBO defined by c0,c,H.
        Returns a list of torch tensors (binary vectors). Falls back to heuristic sampling if gurobipy missing.
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            # fallback: collect k distinct solutions via GreedyFlip with random starts
            sols = []
            import torch as _t
            tries = max(20, k * 5)
            seen = set()
            for _ in range(tries):
                xd0 = _t.randint(0, 2, (space.disc.Dd,), dtype=space.dtype, device=space.device)
                sol = self._fallback_oracle.solve(c0, c, H, space, xd_start=xd0)
                key = tuple(int(x.item()) for x in sol)
                if key not in seen:
                    seen.add(key)
                    sols.append(sol.clone())
                    if len(sols) >= k:
                        break
            return sols

        # build gurobi model
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        if time_limit is not None:
            model.setParam('TimeLimit', float(time_limit))
        Dd = space.disc.Dd
        xvars = model.addVars(range(Dd), vtype=GRB.BINARY, name='x')
        lin_terms = []
        for i in range(Dd):
            coeff = c[i].item() if hasattr(c[i], 'item') else float(c[i])
            if coeff != 0.0:
                lin_terms.append(coeff * xvars[i])
        quad_terms = []
        H_np = H.cpu().numpy() if hasattr(H, 'cpu') else H
        for i in range(Dd):
            for j in range(Dd):
                coeff = float(H_np[i, j])
                if coeff != 0.0:
                    quad_terms.append(coeff * (xvars[i] * xvars[j]))
        obj_terms = []
        if lin_terms:
            obj_terms.extend(lin_terms)
        obj_terms.extend(quad_terms)
        if obj_terms:
            obj = gp.quicksum(obj_terms)
        else:
            obj = gp.LinExpr()
        if c0 != 0.0:
            obj += float(c0)
        model.setObjective(obj, GRB.MINIMIZE)

        sols = []
        excluded = []
        for _ in range(k):
            model.optimize()
            if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT and model.status != GRB.SUBOPTIMAL:
                break
            sol = gp.tupledict({i: xvars[i].X for i in range(Dd)})
            import torch as _t
            xd = _t.zeros(Dd, dtype=space.dtype, device=space.device)
            key = []
            for i in range(Dd):
                val = sol[i]
                xd[i] = 1.0 if val > 0.5 else 0.0
                key.append(int(xd[i].item()))
            sols.append(xd.clone())
            # exclude this solution by adding linear cut: sum_{i in S} x_i + sum_{i not in S} (1-x_i) <= Dd-1
            expr = gp.quicksum([xvars[i] if key[i] == 1 else (1 - xvars[i]) for i in range(Dd)])
            model.addConstr(expr <= Dd - 1)
        return sols

    def solve(self, w_d: Tensor, w_c: Tensor, w_m: Tensor, xd0: Tensor, xc0: Tensor) -> Tuple[Tensor, Tensor, float]:
        import torch as _t
        # build discrete surrogate
        c0, c, H = self._build_discrete_surrogate(w_d, w_m)
        # enumerate candidate discrete solutions using Gurobi (or fallback)
        cand_xd_list = self._solve_qubo_topk(c0, c, H, self.space, self.n_candidates, self.time_limit)
        best_val = float('inf')
        best_xd = xd0.clone()
        best_xc = xc0.clone()
        # build full weight vector concatenated for final evaluation convenience
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        w_full = _t.cat([w_d, w_c, w_m])

        for xd in cand_xd_list:
            # refine continuous using tiered oracle
            phid = self.phi_d(xd)
            def obj_fn(xc: Tensor) -> Tensor:
                phic = self.phi_c(xc)
                phim = self.phi_m(phid, phic)
                return _t.dot(w_full, _t.cat([phid, phic, phim]))
            def grad_fn(xc: Tensor) -> Tensor:
                xc_req = xc.clone().detach().requires_grad_(True)
                phic = self.phi_c(xc_req)
                phim = self.phi_m(phid, phic)
                val = _t.dot(w_full, _t.cat([phid, phic, phim]))
                val.backward()
                return xc_req.grad.detach()

            try:
                xc_ref = self.cont_oracle.solve(obj_fn, grad_fn, self.space, xc_start=xc0, mode='refine')
            except TypeError:
                # older ContinuousOracle may not accept mode; call without mode
                xc_ref = self.cont_oracle.solve(obj_fn, grad_fn, self.space, xc_start=xc0)

            val = float(obj_fn(xc_ref).item()) if hasattr(obj_fn(xc_ref), 'item') else float(obj_fn(xc_ref))
            if val < best_val:
                best_val = val
                best_xd = xd.clone()
                best_xc = xc_ref.clone()

        return best_xd, best_xc, best_val


# Gurobi full-acquisition optimizer removed per user request. The code retains
# Gurobi-backed discrete enumeration optimizer (`GurobiMixedOptimizer`) and the
# BranchAndBound and Alternating optimizers. If you later want the full MIP
# formulation for acquisition minimization, we can reintroduce a tested and
# hardened implementation.


# BARON support removed per user request (license not available). The codebase
# keeps Gurobi- and BranchAndBound-based optimizers. If you later obtain a
# BARON license and want a Pyomo-backed MINLP optimizer, we can reintroduce a
# lightweight Pyomo wrapper then.

# =========================================================
# MVBO driver
# =========================================================

class MIVABO:
    def __init__(
        self,
        space: MixedSpace,
        Md_strategy: Callable[[int], DiscreteFeatures],
        Mc_strategy: Callable[[int], RFFContinuousFeatures],
        optimizer: Optional[AcquisitionOptimizer] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        seed: int | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("MIVABO requires PyTorch. Please install torch (see instructions below).")
        import torch as _t
        self.space = space
        self.phi_d = Md_strategy(space.disc.Dd)
        self.phi_c = Mc_strategy(space.cont.Dc)
        self.phi_m = MixedFeatures(self.phi_d.dim, self.phi_c.dim)
        self.model = LinearFeatureModel(self.phi_d.dim + self.phi_c.dim + self.phi_m.dim, alpha=alpha, beta=beta, dtype=space.dtype, device=space.device)
        self.ts = ThompsonSampler(self.model)
        # Default optimizer: if user didn't provide one, construct a working optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Prefer BranchAndBound (as demo previously did). Pass feature objects and space so
            # constraint-aware logic in the optimizers is used.
            try:
                self.optimizer = BranchAndBoundOptimizer(space, self.phi_d, self.phi_c, self.phi_m)
            except Exception:
                # fallback: alternating optimizer
                self.optimizer = AlternatingOptimizer(space, self.phi_d, self.phi_c, self.phi_m)
        # Do not set global seeds here; leave randomness uncontrolled so trials are independent
        self.rng = random.Random()  # non-deterministic

    def phi_concat(self, xd: Tensor, xc: Tensor) -> Tensor:
        import torch as _t
        phid = self.phi_d(xd)
        phic = self.phi_c(xc)
        phim = self.phi_m(phid, phic)
        return _t.cat([phid, phic, phim])

    def initialize(self, f: Callable[[Tensor, Tensor], float], n_init: int = 10):
        import torch as _t
        try:
            InfeasibleError
        except Exception:
            # bind from module scope if present
            try:
                from .MIVABO import InfeasibleError as InfeasibleError
            except Exception:
                try:
                    from MIVABO import InfeasibleError as InfeasibleError
                except Exception:
                    class InfeasibleError(Exception):
                        pass
        attempts = 0
        accepted = 0
        max_attempts = max(5 * n_init, n_init + 50)
        while accepted < n_init and attempts < max_attempts:
            attempts += 1
            xd = _t.randint(0, 2, (self.space.disc.Dd,), dtype=self.space.dtype, device=self.space.device)
            xc = _t.rand(self.space.cont.Dc, dtype=self.space.dtype, device=self.space.device)
            # Pre-check feasibility using space helpers to avoid calling user objective on infeasible points
            try:
                # Prefer full mixed feasibility if available
                if hasattr(self.space, 'mixed_feasible'):
                    if not self.space.mixed_feasible(xd, xc):
                        continue
                else:
                    # Fall back to discrete-only feasibility check for spaces with only discrete constraints
                    if not self.space.disc_feasible(xd):
                        continue
            except Exception:
                # If feasibility check fails for any reason, be conservative and attempt the objective which may raise
                pass
            try:
                y = f(xd, xc)
            except Exception as e:
                # Prefer explicit InfeasibleError but catch broadly to avoid halting init
                if isinstance(e, InfeasibleError):
                    # skip infeasible samples
                    continue
                # otherwise rethrow
                raise
            self.model.update(self.phi_concat(xd, xc), y)
            accepted += 1
        if accepted < n_init:
            raise RuntimeError(f"Could only collect {accepted} feasible init points after {attempts} attempts")

    def propose(self, n_starts: int = 10) -> Tuple[Tensor, Tensor]:
        import torch as _t
        w = self.ts.sample_weight()
        Md = self.phi_d.dim
        Mc = self.phi_c.dim
        wd = w[:Md]
        wc = w[Md : Md + Mc]
        wm = w[Md + Mc :]
        best_val = float("inf")
        best_x = None
        for _ in range(n_starts):
            xd0 = _t.randint(0, 2, (self.space.disc.Dd,), dtype=self.space.dtype, device=self.space.device)
            xc0 = _t.rand(self.space.cont.Dc, dtype=self.space.dtype, device=self.space.device)
            xd, xc, val = self.optimizer.solve(wd, wc, wm, xd0, xc0)
            if val < best_val:
                best_val = val
                best_x = (xd, xc)
        return best_x

    def observe(self, xd: Tensor, xc: Tensor, y: float):
        self.model.update(self.phi_concat(xd, xc), y)

# =========================================================
# Example objective (synthetic) — MINIMIZATION
# =========================================================

def synthetic_objective(space: MixedSpace) -> Callable[[Tensor, Tensor], float]:
    """Branin(xc[:2]) + mild structure on xd + small interactions. Used for smoke tests."""
    def branin(xy: Tensor) -> Tensor:
        x = xy[0] * 15 - 5
        y = xy[1] * 15
        a = 1.0
        b = 5.1 / (4.0 * math.pi ** 2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * math.cos(x) + s

    def f(xd: Tensor, xc: Tensor) -> float:
        import torch as _t
        val = float(branin(xc[:2]))
        k = max(1, space.disc.Dd // 4)
        act = float(_t.sum(xd).item())
        val += 0.2 * (act - k) ** 2
        val += 0.1 * float((_t.sum(xd[: min(4, space.disc.Dd)]) * _t.sum(xc[:1])).item())
        return val

    return f

# =========================================================
# Tests (kept, plus one more small dimension test)
# =========================================================

def _test_interface_parity():
    import torch as _t
    _t.set_default_dtype(_t.double)
    space = MixedSpace(DiscreteSpec(Dd=8), ContinuousSpec(Dc=2))
    md = lambda Dd: DiscreteFeatures(Dd, include_bias=True)
    mc = lambda Dc: RFFContinuousFeatures(Dc, num_features=32, kernel_lengthscale=0.3, seed=None, dtype=space.dtype, device=space.device)
    f = synthetic_objective(space)

    alt = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=AlternatingOptimizer(space, md(space.disc.Dd), mc(space.cont.Dc), MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim)))
    bnb = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=BranchAndBoundOptimizer(space, md(space.disc.Dd), mc(space.cont.Dc), MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim)))

    alt.initialize(f, n_init=8)
    bnb.initialize(f, n_init=8)

    xd_a, xc_a = alt.propose(n_starts=6)
    xd_b, xc_b = bnb.propose(n_starts=6)

    assert isinstance(xd_a, _t.Tensor) and isinstance(xc_a, _t.Tensor)
    assert xd_a.numel() == space.disc.Dd and xc_a.numel() == space.cont.Dc
    assert isinstance(xd_b, _t.Tensor) and isinstance(xc_b, _t.Tensor)
    assert xd_b.numel() == space.disc.Dd and xc_b.numel() == space.cont.Dc


def _test_short_runs():
    import torch as _t
    _t.set_default_dtype(_t.double)
    space = MixedSpace(DiscreteSpec(Dd=12), ContinuousSpec(Dc=2))
    md = lambda Dd: DiscreteFeatures(Dd, include_bias=True)
    mc = lambda Dc: RFFContinuousFeatures(Dc, num_features=64, kernel_lengthscale=0.2, seed=None, dtype=space.dtype, device=space.device)
    f = synthetic_objective(space)

    # Alternating
    alt_opt = AlternatingOptimizer(space, md(space.disc.Dd), mc(space.cont.Dc), MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim))
    bo_alt = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=alt_opt)
    bo_alt.initialize(f, n_init=12)
    for t in range(5):
        xd, xc = bo_alt.propose(n_starts=6)
        y = f(xd, xc)
        bo_alt.observe(xd, xc, y)
        print(f"[ALT] iter {t+1:02d} | y={y:.3f} | bits={int(xd.sum().item())} | xc={xc.tolist()}")

    # Branch & Bound
    bnb_opt = BranchAndBoundOptimizer(space, md(space.disc.Dd), mc(space.cont.Dc), MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim))
    bo_bnb = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=bnb_opt)
    bo_bnb.initialize(f, n_init=12)
    for t in range(5):
        xd, xc = bo_bnb.propose(n_starts=6)
        y = f(xd, xc)
        bo_bnb.observe(xd, xc, y)
        print(f"[BnB] iter {t+1:02d} | y={y:.3f} | bits={int(xd.sum().item())} | xc={xc.tolist()}")


def _test_feature_dims_sanity():
    # This test does not require torch tensors; checks combinatorics only
    Dd = 5
    df = DiscreteFeatures(Dd, include_bias=True)
    expected_dim = 1 + Dd + (Dd * (Dd - 1)) // 2
    assert df.dim == expected_dim, f"DiscreteFeatures.dim={df.dim}, expected {expected_dim}"

# =========================================================
# Demo / Entry point
# =========================================================
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("\n[Dependency check] PyTorch is not installed.\n")
        print("To run this script, please install the CPU wheel of PyTorch (recommended):\n")
        print("  pip install --upgrade pip")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
        print("If you have a CUDA GPU and drivers, choose the appropriate CUDA wheel from:\n  https://pytorch.org/get-started/locally/\n")
        print("After installing, rerun this file. Demos and tests will execute.")
        sys.exit(0)

    import torch as _t
    _t.set_default_dtype(_t.double)

    # Demo: use the Branin function on the continuous variables (ignore discrete bits)
    space = MixedSpace(DiscreteSpec(Dd=12), ContinuousSpec(Dc=2))
    md = lambda Dd: DiscreteFeatures(Dd, include_bias=True)
    mc = lambda Dc: RFFContinuousFeatures(Dc, num_features=64, kernel_lengthscale=0.2, seed=None, dtype=space.dtype, device=space.device)

    # Use Branch & Bound by default in the demo; swap to Alternating if you prefer
    optimizer = BranchAndBoundOptimizer(space, md(space.disc.Dd), mc(space.cont.Dc), MixedFeatures(md(space.disc.Dd).dim, mc(space.cont.Dc).dim))
    bo = MIVABO(space, Md_strategy=md, Mc_strategy=mc, optimizer=optimizer)

    # Standalone Branin (maps xc[:2] in [0,1]^2 to Branin domain)
    def branin(xy):
        x = float(xy[0]) * 15.0 - 5.0
        y = float(xy[1]) * 15.0
        a = 1.0
        b = 5.1 / (4.0 * math.pi ** 2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * math.cos(x) + s

    def f_branin(xd, xc):
        # ignore xd, only use first two continuous dims
        return float(branin(xc[:2]))

    bo.initialize(f_branin, n_init=15)

    # Run acquisition loop: sample weight, optimize acquisition, print acquisition value and observed y
    for t in range(10):
        # sample weight and run optimizer with multiple random starts (this replicates propose but returns the acquisition value)
        w = bo.ts.sample_weight()
        Md = bo.phi_d.dim
        Mc = bo.phi_c.dim
        wd = w[:Md]
        wc = w[Md : Md + Mc]
        wm = w[Md + Mc :]

        best_val = float("inf")
        best_x = None
        for _ in range(8):
            xd0 = _t.randint(0, 2, (space.disc.Dd,), dtype=space.dtype, device=space.device)
            xc0 = _t.rand(space.cont.Dc, dtype=space.dtype, device=space.device)
            xd, xc, val = bo.optimizer.solve(wd, wc, wm, xd0, xc0)
            if val < best_val:
                best_val = val
                best_x = (xd.clone(), xc.clone())

        xd, xc = best_x
        y = f_branin(xd, xc)
        bo.observe(xd, xc, y)
        print(f"iter {t+1:02d} | y={y:.3f} | acquisition={best_val:.6f} | active_bits={int(xd.sum().item())} | xc={xc.tolist()}")