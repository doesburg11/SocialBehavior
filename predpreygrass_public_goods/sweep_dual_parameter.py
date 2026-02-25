#!/usr/bin/env python3
"""
General 2D parameter sweep for the cooperation model (no CLI).

Edit the configuration block below to choose:
- which 2 parameters to sweep,
- value ranges (or explicit value lists),
- runtime and adaptive refinement settings,
- output location and naming.
"""

from __future__ import annotations

import contextlib
import csv
import io
import math
import os
import re
import statistics as stats
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple

import numpy as np

# Support both direct script execution and module execution.
if __package__:
    from . import emerging_cooperation as eco
else:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import predpreygrass_public_goods.emerging_cooperation as eco


# ============================================================
# SWEEP CONFIG (edit here)
# ============================================================

# Parameters to sweep (must exist in emerging_cooperation.py and be bool/int/float).
X_PARAM = "COOP_COST"
Y_PARAM = "P0"

# Range mode (used when *_VALUES is None).
X_MIN = 0.00
X_MAX = 1.00
X_STEP = 0.01
Y_MIN = 0.00
Y_MAX = 1.00
Y_STEP = 0.01

# Optional explicit value lists (override range mode).
X_VALUES: List[float] | None = None
Y_VALUES: List[float] | None = None

# Cell evaluation.
SUCCESSES = 10
MAX_ATTEMPTS = 100
TAIL_WINDOW = 200
STEPS = 1500
SEED = 4000
WORKERS = 1

# Output.
OUT_DIR = "./predprey_public_goods/images"
NAME_PREFIX = "sweep"

# Adaptive refinement (applies to range mode only).
ADAPTIVE = False
ROUNDS = 3
TOP_K = 5
MIN_SUCCESS_RATE = 1.0
REFINE_SPAN_MULT = 2.0
REFINE_STEP_FACTOR = 0.5
MIN_STEP = 0.0025
SAVE_ALL = True
CLAMP_TO_INITIAL = True


@dataclass
class SweepConfig:
    x_param: str
    y_param: str
    x_min: float
    x_max: float
    x_step: float
    y_min: float
    y_max: float
    y_step: float
    x_values: List[float] | None
    y_values: List[float] | None
    successes: int
    max_attempts: int
    tail_window: int
    steps: int
    seed: int
    workers: int
    out_dir: str
    name_prefix: str
    adaptive: bool
    rounds: int
    top_k: int
    min_success_rate: float
    refine_span_mult: float
    refine_step_factor: float
    min_step: float
    save_all: bool
    clamp_to_initial: bool


def load_config() -> SweepConfig:
    return SweepConfig(
        x_param=X_PARAM,
        y_param=Y_PARAM,
        x_min=X_MIN,
        x_max=X_MAX,
        x_step=X_STEP,
        y_min=Y_MIN,
        y_max=Y_MAX,
        y_step=Y_STEP,
        x_values=X_VALUES,
        y_values=Y_VALUES,
        successes=SUCCESSES,
        max_attempts=MAX_ATTEMPTS,
        tail_window=TAIL_WINDOW,
        steps=STEPS,
        seed=SEED,
        workers=WORKERS,
        out_dir=OUT_DIR,
        name_prefix=NAME_PREFIX,
        adaptive=ADAPTIVE,
        rounds=ROUNDS,
        top_k=TOP_K,
        min_success_rate=MIN_SUCCESS_RATE,
        refine_span_mult=REFINE_SPAN_MULT,
        refine_step_factor=REFINE_STEP_FACTOR,
        min_step=MIN_STEP,
        save_all=SAVE_ALL,
        clamp_to_initial=CLAMP_TO_INITIAL,
    )


def frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    if start > stop:
        raise ValueError("start must be <= stop")
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-") or "sweep"


def cast_value(kind: str, value: float) -> float | int | bool:
    if kind == "bool":
        return bool(int(round(value)))
    if kind == "int":
        return int(round(value))
    return float(value)


def dedupe_preserve_order(values: Sequence[float | int | bool]) -> List[float | int | bool]:
    seen = set()
    out: List[float | int | bool] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_axis_values(
    min_v: float,
    max_v: float,
    step_v: float,
    explicit_values: List[float] | None,
    kind: str,
) -> List[float | int | bool]:
    raw = explicit_values if explicit_values is not None else frange(min_v, max_v, step_v)
    casted = [cast_value(kind, v) for v in raw]
    vals = dedupe_preserve_order(casted)
    if not vals:
        raise ValueError("axis values must not be empty")
    return vals


def detect_param_kind(param_name: str) -> str:
    if not hasattr(eco, param_name):
        raise ValueError(f"Unknown parameter '{param_name}' in emerging_cooperation.py")
    val = getattr(eco, param_name)
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    raise TypeError(
        f"Parameter '{param_name}' has unsupported type {type(val).__name__}; "
        "supported: bool, int, float."
    )


def fmt_value(v: float | int | bool, kind: str) -> str:
    if kind == "bool":
        return str(int(bool(v)))
    if kind == "int":
        return str(int(v))
    return f"{float(v):.4f}"


@dataclass(frozen=True)
class CellResult:
    i: int
    j: int
    x_val: float | int | bool
    y_val: float | int | bool
    successes: int
    mean: float


def _run_cell(
    i: int,
    j: int,
    x_val: float | int | bool,
    y_val: float | int | bool,
    x_param: str,
    y_param: str,
    successes_target: int,
    max_attempts: int,
    tail_window: int,
    steps: int,
    seed_base: int,
) -> CellResult:

    eco.ANIMATE = False
    eco.RESTART_ON_EXTINCTION = False
    eco.STEPS = steps
    setattr(eco, x_param, x_val)
    setattr(eco, y_param, y_val)

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
    return CellResult(i=i, j=j, x_val=x_val, y_val=y_val, successes=successes, mean=mean)


def run_grid(
    x_vals: List[float | int | bool],
    y_vals: List[float | int | bool],
    cfg: SweepConfig,
    round_idx: int,
    x_kind: str,
    y_kind: str,
) -> Tuple[List[CellResult], np.ndarray, np.ndarray]:
    heat = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    counts = np.zeros((len(y_vals), len(x_vals)), dtype=int)

    jobs = []
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            seed_base = cfg.seed + round_idx * 100000 + i * 1000 + j * 100
            jobs.append((i, j, xv, yv, seed_base))

    worker_args = [
        (
            i,
            j,
            xv,
            yv,
            cfg.x_param,
            cfg.y_param,
            cfg.successes,
            cfg.max_attempts,
            cfg.tail_window,
            cfg.steps,
            seed_base,
        )
        for i, j, xv, yv, seed_base in jobs
    ]

    results: List[CellResult] = []
    if cfg.workers == 1:
        for a in worker_args:
            results.append(_run_cell(*a))
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = [ex.submit(_run_cell, *a) for a in worker_args]
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: (r.i, r.j))
    for r in results:
        counts[r.i, r.j] = r.successes
        heat[r.i, r.j] = r.mean
        print(
            f"{cfg.y_param}={fmt_value(r.y_val, y_kind)} "
            f"{cfg.x_param}={fmt_value(r.x_val, x_kind)} "
            f"success={r.successes}/{cfg.successes} mean_coop={r.mean:.3f}"
        )

    return results, heat, counts


def pick_refine_bounds(
    results: List[CellResult],
    cfg: SweepConfig,
    step_x: float,
    step_y: float,
    base_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float, float, float] | None:
    min_successes = math.ceil(cfg.min_success_rate * cfg.successes)
    candidates = [r for r in results if r.successes >= min_successes and not math.isnan(r.mean)]

    if not candidates:
        candidates = [r for r in results if r.successes > 0 and not math.isnan(r.mean)]

    if not candidates:
        return None

    candidates.sort(key=lambda r: r.mean)
    top = candidates[: cfg.top_k]

    min_x = min(float(r.x_val) for r in top)
    max_x = max(float(r.x_val) for r in top)
    min_y = min(float(r.y_val) for r in top)
    max_y = max(float(r.y_val) for r in top)

    span_x = cfg.refine_span_mult * step_x
    span_y = cfg.refine_span_mult * step_y

    new_x_min = min_x - span_x
    new_x_max = max_x + span_x
    new_y_min = min_y - span_y
    new_y_max = max_y + span_y

    if cfg.clamp_to_initial:
        base_x_min, base_x_max, base_y_min, base_y_max = base_bounds
        new_x_min = max(base_x_min, new_x_min)
        new_x_max = min(base_x_max, new_x_max)
        new_y_min = max(base_y_min, new_y_min)
        new_y_max = min(base_y_max, new_y_max)

    new_step_x = max(step_x * cfg.refine_step_factor, cfg.min_step)
    new_step_y = max(step_y * cfg.refine_step_factor, cfg.min_step)

    # Ensure ranges are valid
    if new_x_max - new_x_min < new_step_x * 0.5:
        center = 0.5 * (new_x_min + new_x_max)
        new_x_min = center - new_step_x
        new_x_max = center + new_step_x
    if new_y_max - new_y_min < new_step_y * 0.5:
        center = 0.5 * (new_y_min + new_y_max)
        new_y_min = center - new_step_y
        new_y_max = center + new_step_y

    return new_x_min, new_x_max, new_step_x, new_y_min, new_y_max, new_step_y


def axis_extent(vals: List[float | int | bool]) -> Tuple[float, float]:
    fv = [float(v) for v in vals]
    if len(fv) == 1:
        return fv[0] - 0.5, fv[0] + 0.5
    return min(fv), max(fv)


def save_heatmap(
    heat: np.ndarray,
    x_vals: List[float | int | bool],
    y_vals: List[float | int | bool],
    x_label: str,
    y_label: str,
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
        extent=[*axis_extent(x_vals), *axis_extent(y_vals)],
        cmap=cmap,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean coop")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved heatmap to {outfile}")


def save_round_csv(
    results: List[CellResult],
    outfile: str,
    x_param: str,
    y_param: str,
    round_idx: int,
    successes_target: int,
) -> None:
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "i",
                "j",
                "x_param",
                "y_param",
                "x_value",
                "y_value",
                "successes_target",
                "successes",
                "mean_coop",
            ]
        )
        for r in results:
            w.writerow(
                [
                    round_idx,
                    r.i,
                    r.j,
                    x_param,
                    y_param,
                    r.x_val,
                    r.y_val,
                    successes_target,
                    r.successes,
                    r.mean,
                ]
            )
    print(f"Saved CSV to {outfile}")


def save_all_rounds_csv(
    all_rows: List[Tuple[int, CellResult]],
    outfile: str,
    x_param: str,
    y_param: str,
    successes_target: int,
) -> None:
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "i",
                "j",
                "x_param",
                "y_param",
                "x_value",
                "y_value",
                "successes_target",
                "successes",
                "mean_coop",
            ]
        )
        for round_idx, r in all_rows:
            w.writerow(
                [
                    round_idx,
                    r.i,
                    r.j,
                    x_param,
                    y_param,
                    r.x_val,
                    r.y_val,
                    successes_target,
                    r.successes,
                    r.mean,
                ]
            )
    print(f"Saved CSV to {outfile}")


def main() -> None:
    cfg = load_config()

    x_kind = detect_param_kind(cfg.x_param)
    y_kind = detect_param_kind(cfg.y_param)

    x_vals0 = build_axis_values(cfg.x_min, cfg.x_max, cfg.x_step, cfg.x_values, x_kind)
    y_vals0 = build_axis_values(cfg.y_min, cfg.y_max, cfg.y_step, cfg.y_values, y_kind)

    x_min = float(min(x_vals0))
    x_max = float(max(x_vals0))
    x_step = float(cfg.x_step)
    y_min = float(min(y_vals0))
    y_max = float(max(y_vals0))
    y_step = float(cfg.y_step)

    base_bounds = (x_min, x_max, y_min, y_max)

    adaptive = cfg.adaptive
    if adaptive and (cfg.x_values is not None or cfg.y_values is not None):
        print("Adaptive mode disabled because X_VALUES/Y_VALUES were provided.")
        adaptive = False

    out_dir = cfg.out_dir
    base_prefix = cfg.name_prefix
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_tag = (
        f"{sanitize_token(base_prefix)}_"
        f"{sanitize_token(cfg.x_param)}_vs_{sanitize_token(cfg.y_param)}_"
        f"{timestamp}"
    )

    rounds = cfg.rounds if adaptive else 1
    all_rows: List[Tuple[int, CellResult]] = []

    for r in range(rounds):
        if r == 0 and (cfg.x_values is not None or cfg.y_values is not None):
            x_vals = x_vals0
            y_vals = y_vals0
        else:
            x_vals = build_axis_values(x_min, x_max, x_step, None, x_kind)
            y_vals = build_axis_values(y_min, y_max, y_step, None, y_kind)

        print(
            f"\n=== Round {r + 1}/{rounds} ===\n"
            f"{cfg.x_param} range: [{float(min(x_vals)):.4f}, {float(max(x_vals)):.4f}] "
            f"step ~{x_step:.4f}\n"
            f"{cfg.y_param} range: [{float(min(y_vals)):.4f}, {float(max(y_vals)):.4f}] "
            f"step ~{y_step:.4f}\n"
        )

        results, heat, counts = run_grid(x_vals, y_vals, cfg, r, x_kind, y_kind)
        all_rows.extend((r + 1, row) for row in results)

        title = (
            f"Mean coop (avg over {cfg.successes} successes; tail {cfg.tail_window})\n"
            f"{cfg.y_param} vs {cfg.x_param}"
        )
        round_stem = f"{base_tag}_r{r + 1}"
        heat_out = os.path.join(out_dir, f"{round_stem}_heatmap.png")
        csv_out = os.path.join(out_dir, f"{round_stem}_cells.csv")

        if cfg.save_all or r == rounds - 1:
            save_heatmap(heat, x_vals, y_vals, cfg.x_param, cfg.y_param, title, heat_out)
            save_round_csv(results, csv_out, cfg.x_param, cfg.y_param, r + 1, cfg.successes)

        if not adaptive or r == rounds - 1:
            break

        refined = pick_refine_bounds(
            results,
            cfg,
            x_step,
            y_step,
            base_bounds,
        )
        if refined is None:
            print("No successful cells found; cannot refine further.")
            break

        (x_min, x_max, x_step, y_min, y_max, y_step) = refined

    all_csv = os.path.join(out_dir, f"{base_tag}_all_rounds.csv")
    save_all_rounds_csv(all_rows, all_csv, cfg.x_param, cfg.y_param, cfg.successes)


if __name__ == "__main__":
    main()
