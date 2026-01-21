from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch as t

from xgb_breast_cancer_objective import make_xgb_breast_cancer_problem


def main():
    parser = argparse.ArgumentParser(description="2D slice plots of the XGB breast-cancer objective (hyperparameter space)")
    parser.add_argument("--plots", type=int, default=10, help="Number of random 2D plots to generate (default: 10)")
    parser.add_argument("--grid", type=int, default=5, help="Grid resolution for continuous variables (default: 5 ⇒ 5x5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baseline and variable selection")
    parser.add_argument("--metric", type=str, default="misclassification", choices=["misclassification", "logloss"], help="Objective metric to plot")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "figures", help="Output directory for PNGs")
    parser.add_argument("--cache-file", type=Path, default=Path("breast_cancer_xgb_split.npz"), help="Cached split path (.npz)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build objective once (caches the dataset split)
    space, f = make_xgb_breast_cancer_problem(metric=args.metric, cache_file=args.cache_file)
    Dd = space.disc.Dd
    Dc = space.cont.Dc
    n_vars = Dd + Dc

    # Baseline point: random binary for xd, random [0,1] for xc (fixed across plots for consistency)
    baseline_xd = t.tensor(rng.integers(0, 2, size=Dd), dtype=t.float32)
    baseline_xc = t.tensor(rng.random(size=Dc), dtype=t.float32)

    # Variable names
    var_names = [f"d{i}" for i in range(Dd)] + [f"c{j}" for j in range(Dc)]

    # Levels for each variable type
    cont_levels = np.linspace(0.0, 1.0, max(2, args.grid))
    disc_levels = np.array([0.0, 1.0])

    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib is required to plot. Please install it (pip install matplotlib). Error: {e}")

    start_all = time.perf_counter()
    for k in range(args.plots):
        # choose two distinct variable indices
        i, j = rng.choice(n_vars, size=2, replace=False)
        name_i, name_j = var_names[i], var_names[j]
        is_disc_i = i < Dd
        is_disc_j = j < Dd
        levels_i = disc_levels if is_disc_i else cont_levels
        levels_j = disc_levels if is_disc_j else cont_levels

        Z = np.zeros((len(levels_j), len(levels_i)), dtype=float)

        # Evaluate grid (nested loops to keep memory small)
        grid_t0 = time.perf_counter()
        evals = 0
        for r, vj in enumerate(levels_j):
            for c, vi in enumerate(levels_i):
                xd = baseline_xd.clone()
                xc = baseline_xc.clone()
                if is_disc_i:
                    xd[i] = float(vi)
                else:
                    xc[i - Dd] = float(vi)
                if is_disc_j:
                    xd[j] = float(vj)
                else:
                    xc[j - Dd] = float(vj)
                Z[r, c] = f(xd, xc)
                evals += 1
        grid_dt = time.perf_counter() - grid_t0

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"XGB objective: {name_i} vs {name_j} ({args.metric})\n{evals} evals in {grid_dt:.1f}s")
        # Tick labels
        ax.set_xlabel(name_i)
        ax.set_ylabel(name_j)
        # map ticks to level values (limit to ~6 ticks to avoid clutter)
        def _tick_vals(levels):
            n = len(levels)
            if n <= 6:
                idxs = np.arange(n)
            else:
                idxs = np.linspace(0, n - 1, 6).round().astype(int)
            return idxs, levels[idxs]
        xticks, xvals = _tick_vals(levels_i)
        yticks, yvals = _tick_vals(levels_j)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{v:.2f}" for v in xvals])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.2f}" for v in yvals])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(args.metric)
        fig.tight_layout()
        out_path = args.outdir / f"xgb_objective_2d_{k+1:02d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path} ({evals} evals, {grid_dt:.1f}s)")

    total_dt = time.perf_counter() - start_all
    print(f"Done. Generated {args.plots} plots in {total_dt:.1f}s. Output: {args.outdir}")


if __name__ == "__main__":
    main()
