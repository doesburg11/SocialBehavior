#!/usr/bin/env python3
"""
Create a one-tick visualization for the worked cooperation example.

This version writes an SVG using only the Python standard library, so it
does not require matplotlib.

Output:
  assets/predprey_public_goods/tick_logic_example.svg
  assets/predprey_public_goods/tick_logic_gridworld.svg
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import numpy as np


def build_tick_example():
    # Parameters from emerging_cooperation.py defaults
    p0 = 0.18
    kill_energy = 4.0
    metab = 0.06
    move_cost = 0.008
    coop_cost = 0.20
    birth_thresh = 3.0

    # Worked example values from the explanation
    names = ["A", "B", "C"]
    coop = np.array([0.2, 0.6, 0.9], dtype=float)
    e0 = np.array([1.8, 2.4, 1.1], dtype=float)

    s = float(np.sum(coop))
    p_kill = 1.0 - (1.0 - p0) ** s
    random_draw = 0.21
    kill_success = random_draw < p_kill

    share = kill_energy / len(names) if kill_success else 0.0
    costs = metab + move_cost + coop_cost * coop
    e1 = e0 + share - costs
    repro = e1 >= birth_thresh

    return {
        "p0": p0,
        "kill_energy": kill_energy,
        "birth_thresh": birth_thresh,
        "names": names,
        "coop": coop,
        "e0": e0,
        "s": s,
        "p_kill": p_kill,
        "random_draw": random_draw,
        "kill_success": kill_success,
        "share": share,
        "costs": costs,
        "e1": e1,
        "repro": repro,
    }


def _panel_frame(parts: list[str], x: int, y: int, w: int, h: int, title: str) -> None:
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="#ffffff" stroke="#d6d6d6" stroke-width="1.5" rx="8" />'
    )
    parts.append(
        f'<text x="{x + 14}" y="{y + 28}" font-family="Arial" font-size="18" '
        f'font-weight="700" fill="#1f1f1f">{escape(title)}</text>'
    )


def _text(parts: list[str], x: float, y: float, txt: str, size: int = 14, weight: str = "400", color: str = "#222222", family: str = "Arial") -> None:
    parts.append(
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(txt)}</text>'
    )


def _line(parts: list[str], x1: float, y1: float, x2: float, y2: float, color: str = "#999999", width: float = 1.0, dash: str | None = None) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{dash_attr} />'
    )


def _rect(parts: list[str], x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", sw: float = 0.0, rx: int = 0) -> None:
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}" />'
    )


def _circle(parts: list[str], x: float, y: float, r: float, fill: str, stroke: str = "none", sw: float = 0.0) -> None:
    parts.append(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
    )


def _cell_center(x0: float, y0: float, cell: float, ix: int, iy: int) -> tuple[float, float]:
    return x0 + (ix + 0.5) * cell, y0 + (iy + 0.5) * cell


def plot_tick_example(outfile: Path) -> None:
    ex = build_tick_example()

    width, height = 1280, 820
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    _rect(parts, 0, 0, width, height, "#f7f8fb")
    _text(
        parts,
        width / 2,
        42,
        "Predator-Public-Goods Model: One Tick Worked Example",
        size=28,
        weight="700",
        color="#101820",
        family="Arial",
    )

    # Center title text by anchoring to middle
    parts[-1] = parts[-1].replace("<text ", '<text text-anchor="middle" ')

    p1 = (40, 80, 570, 300)
    p2 = (650, 80, 570, 300)
    p3 = (40, 420, 570, 320)
    p4 = (650, 420, 570, 320)

    _panel_frame(parts, *p1, "1) Inputs + Formula")
    _panel_frame(parts, *p2, "2) Hunt Decision")
    _panel_frame(parts, *p3, "3) Energy Update")
    _panel_frame(parts, *p4, "4) Reproduction Check")

    # Panel 1 content
    x, y, w, h = p1
    _text(parts, x + 20, y + 62, "Predators in same cell:", size=16, weight="600")
    for i, name in enumerate(ex["names"]):
        cx = x + 90 + i * 120
        cy = y + 112
        _circle(parts, cx, cy, 22, "#ffffff", stroke="#2f5d8c", sw=2.5)
        _text(parts, cx, cy + 5, name, size=14, weight="700", color="#2f5d8c")
        parts[-1] = parts[-1].replace("<text ", '<text text-anchor="middle" ')
        _text(parts, cx - 36, cy + 40, f"coop={ex['coop'][i]:.1f}", size=12, color="#3a3a3a")
        _text(parts, cx - 36, cy + 58, f"E0={ex['e0'][i]:.1f}", size=12, color="#3a3a3a")

    rule_y = y + 200
    _text(parts, x + 20, rule_y, "Hunt rule: p_kill = 1 - (1 - P0)^S", size=15, family="Courier New")
    _text(parts, x + 20, rule_y + 26, f"P0={ex['p0']:.2f}, S=sum(coop)={ex['s']:.1f}", size=15, family="Courier New")
    _text(parts, x + 20, rule_y + 52, f"p_kill={ex['p_kill']:.3f}", size=15, family="Courier New", weight="700", color="#1c4e80")

    # Panel 2 content
    x, y, w, h = p2
    bar_x = x + 70
    bar_y = y + 145
    bar_w = 420
    bar_h = 34
    _rect(parts, bar_x, bar_y, bar_w, bar_h, "#e8edf3", stroke="#bcc8d6", sw=1.2, rx=4)
    _rect(parts, bar_x, bar_y, bar_w * ex["p_kill"], bar_h, "#4c78a8", rx=4)
    draw_x = bar_x + bar_w * ex["random_draw"]
    _line(parts, draw_x, bar_y - 20, draw_x, bar_y + bar_h + 20, color="#e45756", width=2.5, dash="8,6")
    _text(parts, bar_x, bar_y - 28, "0.0", size=12, color="#616161")
    _text(parts, bar_x + bar_w - 18, bar_y - 28, "1.0", size=12, color="#616161")
    _text(parts, bar_x, bar_y + 62, f"random draw = {ex['random_draw']:.2f}", size=14, color="#e45756", weight="600")
    _text(parts, bar_x + 220, bar_y + 62, f"p_kill = {ex['p_kill']:.3f}", size=14, color="#2f5d8c", weight="600")

    outcome = "SUCCESS (one prey removed)" if ex["kill_success"] else "FAIL (no prey removed)"
    outcome_color = "#2e7d32" if ex["kill_success"] else "#c62828"
    _text(parts, x + 70, y + 250, outcome, size=18, weight="700", color=outcome_color)
    _text(parts, x + 70, y + 280, f"Reward share per predator: {ex['share']:.3f}", size=15)

    # Panel 3 content
    x, y, w, h = p3
    ch_x, ch_y = x + 70, y + 80
    ch_w, ch_h = 450, 210
    y_max = 4.2
    _line(parts, ch_x, ch_y + ch_h, ch_x + ch_w, ch_y + ch_h, color="#666666", width=1.5)
    _line(parts, ch_x, ch_y, ch_x, ch_y + ch_h, color="#666666", width=1.5)

    for t in [0, 1, 2, 3, 4]:
        yy = ch_y + ch_h - (t / y_max) * ch_h
        _line(parts, ch_x - 6, yy, ch_x + ch_w, yy, color="#e5e5e5", width=1.0)
        _text(parts, ch_x - 30, yy + 4, f"{t}", size=11, color="#555555")

    group_step = ch_w / 3.0
    bar_w = 38
    for i, name in enumerate(ex["names"]):
        gx = ch_x + group_step * (i + 0.5)
        h0 = (ex["e0"][i] / y_max) * ch_h
        h1 = (ex["e1"][i] / y_max) * ch_h
        _rect(parts, gx - 44, ch_y + ch_h - h0, bar_w, h0, "#72b7b2")
        _rect(parts, gx + 6, ch_y + ch_h - h1, bar_w, h1, "#54a24b")
        _text(parts, gx - 52, ch_y + ch_h + 24, "Initial", size=11, color="#3f6f6a")
        _text(parts, gx + 2, ch_y + ch_h + 24, "Final", size=11, color="#2e6a2c")
        _text(parts, gx - 13, ch_y + ch_h + 44, name, size=13, weight="700")
        _text(parts, gx - 46, ch_y + ch_h - max(h0, h1) - 10, f"+{ex['share']:.3f}  -{ex['costs'][i]:.3f}", size=11, color="#333333")

    _text(parts, x + 20, y + 300, "Per-tick cost = METAB + MOVE + COOP_COST * coop", size=13, family="Courier New")

    # Panel 4 content
    x, y, w, h = p4
    ch_x, ch_y = x + 70, y + 80
    ch_w, ch_h = 450, 210
    y_max = 4.2
    _line(parts, ch_x, ch_y + ch_h, ch_x + ch_w, ch_y + ch_h, color="#666666", width=1.5)
    _line(parts, ch_x, ch_y, ch_x, ch_y + ch_h, color="#666666", width=1.5)
    thresh_y = ch_y + ch_h - (ex["birth_thresh"] / y_max) * ch_h
    _line(parts, ch_x, thresh_y, ch_x + ch_w, thresh_y, color="#424242", width=2.0, dash="7,6")
    _text(parts, ch_x + 8, thresh_y - 8, f"birth threshold = {ex['birth_thresh']:.1f}", size=12, color="#424242")

    group_step = ch_w / 3.0
    bar_w = 56
    for i, name in enumerate(ex["names"]):
        gx = ch_x + group_step * (i + 0.5)
        h1 = (ex["e1"][i] / y_max) * ch_h
        color = "#2e7d32" if ex["repro"][i] else "#ef6c00"
        status = "reproduce" if ex["repro"][i] else "no birth"
        _rect(parts, gx - bar_w / 2, ch_y + ch_h - h1, bar_w, h1, color)
        _text(parts, gx - 6, ch_y + ch_h + 24, name, size=14, weight="700")
        _text(parts, gx - 30, ch_y + ch_h - h1 - 8, f"{ex['e1'][i]:.3f}", size=12, color="#222222")
        _text(parts, gx - 36, ch_y + ch_h - h1 - 26, status, size=12, color=color, weight="700")

    parts.append("</svg>")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(parts), encoding="utf-8")


def plot_tick_gridworld(outfile: Path) -> None:
    ex = build_tick_example()

    width, height = 1320, 860
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    _rect(parts, 0, 0, width, height, "#f7f8fb")

    _text(
        parts,
        width / 2,
        44,
        "One Tick Gridworld: Before and After Hunt",
        size=30,
        weight="700",
        color="#101820",
    )
    parts[-1] = parts[-1].replace("<text ", '<text text-anchor="middle" ')

    left = (40, 80, 600, 660)
    right = (680, 80, 600, 660)
    _panel_frame(parts, *left, "A) Before Hunt")
    _panel_frame(parts, *right, "B) After Hunt")

    grid_n = 9
    cell = 52
    center_cell = (4, 4)
    hunt_r = 1

    pred_offsets = [(-12, -10), (12, -10), (0, 12)]
    pred_colors = ["#4E79A7", "#59A14F", "#E15759"]
    candidate_prey = [("P1", 3, 4), ("P2", 5, 4), ("P3", 4, 3), ("P4", 5, 5), ("P5", 4, 5)]
    outside_prey = [("Q1", 1, 1), ("Q2", 7, 2), ("Q3", 7, 7)]
    killed_id = "P1"

    def draw_panel(px: int, py: int, remove_killed: bool) -> None:
        gx = px + 44
        gy = py + 68
        gw = grid_n * cell
        gh = grid_n * cell

        _rect(parts, gx, gy, gw, gh, "#ffffff", stroke="#b5bdc8", sw=1.5)

        # Highlight the HUNT_R neighborhood around the predator cell.
        hx = gx + (center_cell[0] - hunt_r) * cell
        hy = gy + (center_cell[1] - hunt_r) * cell
        hw = (2 * hunt_r + 1) * cell
        _rect(parts, hx, hy, hw, hw, "#e3f2fd", stroke="#1f77b4", sw=2.0)

        for k in range(grid_n + 1):
            x = gx + k * cell
            y = gy + k * cell
            _line(parts, x, gy, x, gy + gh, color="#d8dce3", width=1.0)
            _line(parts, gx, y, gx + gw, y, color="#d8dce3", width=1.0)

        # Grid labels
        for idx in range(grid_n):
            _text(parts, gx + idx * cell + 20, gy - 10, str(idx), size=11, color="#70757d")
            _text(parts, gx - 18, gy + idx * cell + 31, str(idx), size=11, color="#70757d")

        _text(parts, gx + 12, gy + gh + 26, "x", size=12, color="#444444", weight="600")
        _text(parts, gx - 28, gy + 14, "y", size=12, color="#444444", weight="600")

        # Draw prey outside neighborhood
        for prey_id, ix, iy in outside_prey:
            cx, cy = _cell_center(gx, gy, cell, ix, iy)
            _circle(parts, cx, cy, 8, "#c6c9cf", stroke="#888c94", sw=1.2)
            _text(parts, cx + 10, cy + 4, prey_id, size=10, color="#6f747d")

        # Draw candidate prey
        for prey_id, ix, iy in candidate_prey:
            if remove_killed and ex["kill_success"] and prey_id == killed_id:
                continue
            cx, cy = _cell_center(gx, gy, cell, ix, iy)
            stroke = "#8c4a00" if prey_id == killed_id else "#935b17"
            sw = 2.2 if prey_id == killed_id else 1.4
            _circle(parts, cx, cy, 8.5, "#f28e2b", stroke=stroke, sw=sw)
            _text(parts, cx + 11, cy + 4, prey_id, size=10, color="#6d3d02")

        # Draw the predator group in the same center cell.
        cx0, cy0 = _cell_center(gx, gy, cell, center_cell[0], center_cell[1])
        for i, name in enumerate(ex["names"]):
            ox, oy = pred_offsets[i]
            cx = cx0 + ox
            cy = cy0 + oy
            _circle(parts, cx, cy, 10.5, "#ffffff", stroke=pred_colors[i], sw=2.2)
            _text(parts, cx - 4, cy + 4, name, size=10, color=pred_colors[i], weight="700")

        # Mark removed prey in after panel.
        if remove_killed and ex["kill_success"]:
            kx, ky = _cell_center(gx, gy, cell, 3, 4)
            _line(parts, kx - 9, ky - 9, kx + 9, ky + 9, color="#b00020", width=2.2)
            _line(parts, kx - 9, ky + 9, kx + 9, ky - 9, color="#b00020", width=2.2)
            _text(parts, kx - 14, ky - 10, "P1 removed", size=11, color="#b00020", weight="700")
            parts[-1] = parts[-1].replace("<text ", '<text text-anchor="end" ')

        # Keep legend and explanatory text inside panel bottom area to avoid overlap.
        ly = gy + gh + 24
        _text(parts, gx, ly, "Legend:", size=12, weight="700")
        _circle(parts, gx + 60, ly - 3, 6, "#f28e2b", stroke="#935b17", sw=1.2)
        _text(parts, gx + 72, ly + 1, "candidate prey", size=11)
        _circle(parts, gx + 220, ly - 3, 6, "#c6c9cf", stroke="#888c94", sw=1.2)
        _text(parts, gx + 232, ly + 1, "outside prey", size=11)
        _circle(parts, gx + 360, ly - 3, 6, "#ffffff", stroke="#4E79A7", sw=1.8)
        _text(parts, gx + 372, ly + 1, "predator", size=11)

        _text(parts, gx, gy + gh + 78, "Highlighted blue box: HUNT_R = 1 neighborhood", size=12, color="#1f4e79")
        _text(parts, gx, gy + gh + 100, "Predators A,B,C share one cell and one hunt attempt", size=12, color="#1f1f1f")

    draw_panel(left[0], left[1], remove_killed=False)
    draw_panel(right[0], right[1], remove_killed=True)

    # Global equations and outcome under the two panels.
    _text(
        parts,
        70,
        790,
        (
            f"S=sum(coop)={ex['s']:.1f}, "
            f"p_kill = 1 - (1 - {ex['p0']:.2f})^S = {ex['p_kill']:.3f}, "
            f"draw={ex['random_draw']:.2f} -> kill"
        ),
        size=15,
        family="Courier New",
        weight="700",
        color="#213547",
    )
    _text(
        parts,
        70,
        816,
        f"Kill reward shared equally: {ex['kill_energy']:.1f} / 3 = {ex['share']:.3f} energy per predator",
        size=14,
        color="#213547",
    )

    parts.append("</svg>")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outfile",
        type=Path,
        default=Path("assets/predprey_public_goods/tick_logic_example.svg"),
    )
    ap.add_argument(
        "--grid-outfile",
        type=Path,
        default=Path("assets/predprey_public_goods/tick_logic_gridworld.svg"),
    )
    args = ap.parse_args()

    plot_tick_example(args.outfile)
    plot_tick_gridworld(args.grid_outfile)
    print(f"Saved: {args.outfile}")
    print(f"Saved: {args.grid_outfile}")


if __name__ == "__main__":
    main()
