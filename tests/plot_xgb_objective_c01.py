from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch as t

from xgb_breast_cancer_objective import make_xgb_breast_cancer_problem


def main():
    parser = argparse.ArgumentParser(description="2D landscape of XGB objective over continuous c0 vs c1")
    parser.add_argument("--grid", type=int, default=32, help="Grid resolution per axis (default: 32 ⇒ ~1024 evals)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baseline point")
    parser.add_argument("--metric", type=str, default="misclassification", choices=["misclassification", "logloss"], help="Objective metric to plot")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures" / "xgb_objective_2d_c0_c1.png", help="Output PNG path")
    parser.add_argument("--cache-file", type=Path, default=Path("breast_cancer_xgb_split.npz"), help="Cached split path (.npz)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build objective once (caches the dataset split)
    space, f = make_xgb_breast_cancer_problem(metric=args.metric, cache_file=args.cache_file)
    Dd = space.disc.Dd
    Dc = space.cont.Dc
    if Dc < 2:
        raise SystemExit("This objective exposes fewer than 2 continuous variables; cannot plot c0 vs c1.")

    # Fixed baseline point
    baseline_xd = t.tensor(rng.integers(0, 2, size=Dd), dtype=t.float32)
    baseline_xc = t.tensor(rng.random(size=Dc), dtype=t.float32)

    # Indices for c0 and c1 in the concatenated [d..., c...]
    i = Dd + 0  # c0
    j = Dd + 1  # c1

    # Levels: shrink c0 range to [0.2, 1.0], keep c1 in [0.0, 1.0]
    g = max(2, args.grid)
    levels_x = np.linspace(0.2, 1.0, g)  # c0
    levels_y = np.linspace(0.0, 1.0, g)  # c1

    # Evaluate grid (vary c0 horizontally, c1 vertically)
    Z = np.zeros((g, g), dtype=float)
    start = time.perf_counter()
    evals = 0
    for r, vj in enumerate(levels_y):
        for c, vi in enumerate(levels_x):
            xd = baseline_xd.clone()
            xc = baseline_xc.clone()
            xc[0] = float(vi)  # c0
            xc[1] = float(vj)  # c1
            Z[r, c] = f(xd, xc)
            evals += 1
    dt = time.perf_counter() - start

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib is required to plot. Please install it (pip install matplotlib). Error: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", extent=[0.2,1.0,0,1])
    ax.set_xlabel('c0 (0.2 → 1.0)')
    ax.set_ylabel('c1 (0.0 → 1.0)')
    ax.set_title(f"XGB objective over (c0, c1) — {args.metric}\n{evals} evals in {dt:.1f}s")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved 2D landscape to: {args.out}  ({evals} evals, {dt:.1f}s)")


if __name__ == "__main__":
    main()
