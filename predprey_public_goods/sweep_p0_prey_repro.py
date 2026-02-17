#!/usr/bin/env python3
"""
Parameter sweep for cooperation model (public sharing).

- Averages mean_coop over N successful runs for each (P0, PREY_REPRO_PROB)
- Produces a heatmap over PREY_REPRO_PROB vs P0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics as stats
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Support both direct script execution and module execution.
if __package__:
    from . import emerging_cooperation as eco
else:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import predprey_public_goods.emerging_cooperation as eco


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


@dataclass(frozen=True)
class CellResult:
    i: int
    j: int
    p0: float
    prey_repro_prob: float
    successes: int
    mean: float


def _run_cell(
    i: int,
    j: int,
    p0: float,
    prey_repro_prob: float,
    successes_target: int,
    max_attempts: int,
    tail_window: int,
    steps: int,
    seed_base: int,
) -> CellResult:

    eco.ANIMATE = False
    eco.RESTART_ON_EXTINCTION = False
    eco.STEPS = steps
    eco.P0 = p0
    eco.PREY_REPRO_PROB = prey_repro_prob

    successes = 0
    means: List[float] = []

    for attempt in range(max_attempts):
        seed = seed_base + attempt
        with contextlib.redirect_stdout(io.StringIO()):
            (
                pred_hist,
                prey_hist,
                mean_coop_hist,
                var_coop_hist,
                preds_snaps,
                preys_snaps,
                preds_final,
                success,
                extinction_step,
            ) = eco.run_sim(seed_override=seed)

        if success and mean_coop_hist:
            tail_n = min(tail_window, len(mean_coop_hist))
            tail_mean = sum(mean_coop_hist[-tail_n:]) / tail_n
            means.append(tail_mean)
            successes += 1
            if successes >= successes_target:
                break

    mean = stats.mean(means) if means else float("nan")
    return CellResult(
        i=i,
        j=j,
        p0=p0,
        prey_repro_prob=prey_repro_prob,
        successes=successes,
        mean=mean,
    )


def run_grid(
    p0_vals: List[float],
    repro_vals: List[float],
    args: argparse.Namespace,
) -> Tuple[List[CellResult], np.ndarray, np.ndarray]:
    heat = np.full((len(p0_vals), len(repro_vals)), np.nan, dtype=float)
    counts = np.zeros((len(p0_vals), len(repro_vals)), dtype=int)

    jobs = []
    for i, p0 in enumerate(p0_vals):
        for j, repro in enumerate(repro_vals):
            seed_base = args.seed + i * 1000 + j * 100
            jobs.append((i, j, p0, repro, seed_base))

    worker_args = [
        (i, j, p0, repro, args.successes, args.max_attempts, args.tail_window, args.steps, seed_base)
        for i, j, p0, repro, seed_base in jobs
    ]

    results: List[CellResult] = []
    if args.workers == 1:
        for a in worker_args:
            results.append(_run_cell(*a))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_cell, *a) for a in worker_args]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (r.i, r.j))
    for r in results:
        counts[r.i, r.j] = r.successes
        heat[r.i, r.j] = r.mean
        print(
            f"P0={r.p0:.2f} PREY_REPRO_PROB={r.prey_repro_prob:.4f} "
            f"success={r.successes}/{args.successes} mean_coop={r.mean:.3f}"
        )

    return results, heat, counts


def save_heatmap(
    heat: np.ndarray,
    p0_vals: List[float],
    repro_vals: List[float],
    title: str,
    outfile: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[min(repro_vals), max(repro_vals), min(p0_vals), max(p0_vals)],
        cmap=cmap,
    )
    ax.set_xlabel("PREY_REPRO_PROB")
    ax.set_ylabel("P0")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean coop")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved heatmap to {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-min", type=float, default=0.00)
    ap.add_argument("--p0-max", type=float, default=1.00)
    ap.add_argument("--p0-step", type=float, default=0.05)
    ap.add_argument("--repro-min", type=float, default=0.00)
    ap.add_argument("--repro-max", type=float, default=1.00)
    ap.add_argument("--repro-step", type=float, default=0.05)
    ap.add_argument("--successes", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--tail-window", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--outfile", type=str, default="./predprey_public_goods/images/p0_prey_repro_heatmap.png")
    args = ap.parse_args()

    p0_vals = frange(args.p0_min, args.p0_max, args.p0_step)
    repro_vals = frange(args.repro_min, args.repro_max, args.repro_step)

    print(
        "\n=== Sweep ===\n"
        f"P0 range:             [{args.p0_min:.4f}, {args.p0_max:.4f}] step {args.p0_step:.4f}\n"
        f"PREY_REPRO_PROB range:[{args.repro_min:.4f}, {args.repro_max:.4f}] step {args.repro_step:.4f}\n"
    )

    _, heat, _ = run_grid(p0_vals, repro_vals, args)
    title = f"Mean coop (avg over {args.successes} successes; tail {args.tail_window})"

    if args.outfile:
        save_heatmap(heat, p0_vals, repro_vals, title, args.outfile)


if __name__ == "__main__":
    main()
