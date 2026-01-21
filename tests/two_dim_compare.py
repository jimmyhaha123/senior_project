import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tests.MIVABO import (
    MixedSpace,
    DiscreteSpec,
    ContinuousSpec,
    DiscreteFeatures,
    RFFContinuousFeatures,
    MixedFeatures,
    BranchAndBoundOptimizer,
    MIVABO,
)

# Small 2D problem: one integer (1..10) encoded as 4 bits, one continuous in [0,1]
BITS_PER_INT = 4
Dd = BITS_PER_INT
Dc = 1
space = MixedSpace(DiscreteSpec(Dd=Dd), ContinuousSpec(Dc=Dc))
phid = DiscreteFeatures(Dd)
phic = RFFContinuousFeatures(Dc, num_features=8, seed=None, dtype=space.dtype, device=space.device)
phim = MixedFeatures(phid.dim, phic.dim)

# Helpers: decode bits -> integer and continuous mapping
def decode_bits_to_int(xd_bits: torch.Tensor) -> int:
    idx = 0
    for b_i, b in enumerate(xd_bits):
        idx |= (int(b.item()) & 1) << b_i
    val = 1 + idx
    if val < 1:
        val = 1
    if val > 10:
        val = 10
    return int(val)


def decode_continuous(xc_unit: torch.Tensor) -> float:
    return float(1.0 + 9.0 * xc_unit.item())


# Strongly irregular nonconvex objective
def f_obj(xd_bits: torch.Tensor, xc_unit: torch.Tensor) -> float:
    v = float(decode_bits_to_int(xd_bits))
    x = float(decode_continuous(xc_unit))

    # Several narrow and wide wells at different (v,x) locations
    wells = [
        (2.5, 0.15, 0.08, 3.0),
        (3.7, 0.33, 0.12, 2.5),
        (5.0, 0.55, 0.18, 4.0),
        (7.8, 0.72, 0.10, 3.5),
        (9.2, 0.25, 0.06, 5.0),
    ]
    val = 0.0
    for vc, xc_c, width, amp in wells:
        dv = (v - vc) ** 2
        dx = (x - xc_c) ** 2
        val += -amp * math.exp(-0.5 * (dv + dx) / (width * width))

    # Bits-interaction: Hamming weight penalty and specific bit patterns
    bits = [int(b) for b in xd_bits.tolist()]
    hw = sum(bits)
    val += 0.4 * (math.sin(math.pi * hw) * (1.0 - 0.2 * hw))

    # High-frequency ripples in both v and x to create many local minima
    val += 0.6 * math.sin(2.5 * v) * math.cos(9.0 * x)
    val += 0.25 * math.sin(12.0 * x * v)

    # Sharp ridge / cusp behavior via absolute differences from several anchors
    anchors = [(1.0, 0.2), (4.0, 0.6), (8.0, 0.4)]
    for av, ax in anchors:
        val += 0.8 * abs((v - av) * (x - ax)) / (1.0 + (v - av) ** 2 + (x - ax) ** 2)

    # Small quadratic trend to keep objective bounded
    val += 0.02 * (v - 5.5) ** 2 + 0.02 * (x - 0.5) ** 2

    return float(val)


def run_trials_compare(n_trials: int = 30, n_iters: int = 8, n_init: int = 5):
    traces_bnb = np.zeros((n_trials, n_iters), dtype=float)
    traces_rand = np.zeros((n_trials, n_iters), dtype=float)

    for trial in range(n_trials):
        # Fresh models per trial; do NOT set any random seed
        opt_bnb = BranchAndBoundOptimizer(space, phid, phic, phim)
        bo_bnb = MIVABO(space, lambda Dd: phid, lambda Dc: phic, optimizer=opt_bnb)
        bo_rand = MIVABO(space, lambda Dd: phid, lambda Dc: phic, optimizer=None)

        # initialize with a few random points
        for _ in range(n_init):
            xd0 = torch.randint(0, 2, (Dd,), dtype=space.dtype, device=space.device)
            xc0 = torch.rand(Dc, dtype=space.dtype, device=space.device)
            y0 = f_obj(xd0, xc0)
            bo_bnb.model.update(bo_bnb.phi_concat(xd0, xc0), y0)
            bo_rand.model.update(bo_rand.phi_concat(xd0, xc0), y0)

        best_b = float('inf')
        best_r = float('inf')
        for t in range(n_iters):
            # BnB propose
            w = bo_bnb.ts.sample_weight()
            Md = phid.dim
            Mc = phic.dim
            wd = w[:Md]
            wc = w[Md : Md + Mc]
            wm = w[Md + Mc :]
            xd0 = torch.randint(0, 2, (Dd,), dtype=space.dtype, device=space.device)
            xc0 = torch.rand(Dc, dtype=space.dtype, device=space.device)
            xd_sel, xc_sel, _ = bo_bnb.optimizer.solve(wd, wc, wm, xd0, xc0)
            y_sel = f_obj(xd_sel, xc_sel)
            bo_bnb.model.update(bo_bnb.phi_concat(xd_sel, xc_sel), y_sel)
            if y_sel < best_b:
                best_b = y_sel
            traces_bnb[trial, t] = best_b

            # Random baseline
            xr = torch.randint(0, 2, (Dd,), dtype=space.dtype, device=space.device)
            xc_r = torch.rand(Dc, dtype=space.dtype, device=space.device)
            y_r = f_obj(xr, xc_r)
            bo_rand.model.update(bo_rand.phi_concat(xr, xc_r), y_r)
            if y_r < best_r:
                best_r = y_r
            traces_rand[trial, t] = best_r

    # Compute mean and SEM across trials
    mean_b = traces_bnb.mean(axis=0)
    sem_b = traces_bnb.std(axis=0, ddof=1) / np.sqrt(n_trials)
    mean_r = traces_rand.mean(axis=0)
    sem_r = traces_rand.std(axis=0, ddof=1) / np.sqrt(n_trials)

    # Plot averaged incumbents
    try:
        plt.figure(figsize=(8,4))
        xs = list(range(1, n_iters + 1))
        plt.plot(xs, mean_b, marker='x', color='red', label='BnB mean incumbent')
        plt.fill_between(xs, mean_b - sem_b, mean_b + sem_b, color='red', alpha=0.2)
        plt.plot(xs, mean_r, marker='o', color='black', label='Random mean incumbent')
        plt.fill_between(xs, mean_r - sem_r, mean_r + sem_r, color='black', alpha=0.2)
        plt.xlabel('Iteration')
        plt.ylabel('Mean incumbent (lower is better)')
        plt.title(f'Average incumbent over {n_trials} trials')
        plt.legend()
        out_avg = os.path.join(os.path.dirname(__file__), 'two_dim_incumbents_avg.png')
        plt.tight_layout()
        plt.savefig(out_avg)
        print('Saved averaged incumbents plot to', out_avg)
        print('Final mean incumbents: BnB=', mean_b[-1], 'Random=', mean_r[-1])
    except Exception as e:
        print('Could not save averaged incumbents plot:', e)


if __name__ == '__main__':
    run_trials_compare(n_trials=30, n_iters=30, n_init=5)

