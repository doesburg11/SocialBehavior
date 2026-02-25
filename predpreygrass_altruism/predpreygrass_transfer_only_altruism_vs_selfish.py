#!/usr/bin/env python3
"""
Simplified predator-prey-grass simulation for transfer-only altruism vs selfishness.

Predators live in one shared world and carry a heritable strategy:
- altruistic: can donate surplus energy during a transfer phase
- selfish: never donates

This simplified copy intentionally removes extra optionality:
- no post-hunt helper sharing
- no kin-recognition targeting
- only one transfer targeting switch:
  - `altruist_only`
  - `any_predator`
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

if TYPE_CHECKING:
    import pygame as pygame_types


# ============================================================
# CONFIG (edit in file; no CLI parameters)
# ============================================================

SEED = 122913
REPLICATES = 1
STEPS = 500
REPORT_EVERY = 1
TAIL_WINDOW = 220

# World
W, H = 48, 48
GRASS_INIT = 1.45
GRASS_MAX = 6.2
GRASS_REGROWTH = 0.30

# Initial populations
PRED_INIT = 90
PREY_INIT = 3600
ALTRUIST_INIT_FRACTION = 0.5

PRED_ENERGY_INIT_MEAN = 2.2
PRED_ENERGY_INIT_SD = 0.22
PREY_ENERGY_INIT_MEAN = 1.05
PREY_ENERGY_INIT_SD = 0.20

# Prey dynamics
PREY_MOVE_PROB = 0.38
PREY_MOVE_COST = 0.01
PREY_METAB = 0.022
PREY_BITE = 0.72
PREY_ENERGY_CAP = 3.9
PREY_MAX_AGE = 17
PREY_REPRO_MIN_AGE = 2
PREY_REPRO_MAX_AGE = 10
PREY_REPRO_THRESH = 1.12
PREY_REPRO_PROB = 0.88
PREY_CHILD_SHARE = 0.41
PREY_MAX_POP = 15000
PREY_BASE_MORTALITY = 0.0015
PREY_AGE_MORT_ONSET = 10
PREY_AGE_MORT_RATE = 0.004

# Predator dynamics
PRED_MOVE_PROB = 0.75
PRED_MOVE_COST = 0.016
PRED_METAB = 0.050
PRED_ENERGY_CAP = 5.6
PRED_MAX_AGE = 24
PRED_REPRO_MIN_AGE = 3
PRED_REPRO_MAX_AGE = 11
# Tuning lock (transfer-only split objective):
# These shared predator values are tuned together with the transfer settings below
# for the paired 10-seed comparison objective (`SEED + 7919*i`, `i=0..9`,
# `STEPS = 500`) in this simplified transfer-only model:
# - `TRANSFER_RECIPIENT_MODE = "altruist_only"` -> selfish extinct in 10/10
# - `TRANSFER_RECIPIENT_MODE = "any_predator"` -> altruists extinct in 10/10
# If you change these, the exact split usually breaks and must be retuned.
PRED_REPRO_THRESH = 1.34
PRED_REPRO_PROB = 0.30
PRED_CHILD_SHARE = 0.40
PRED_MAX_POP = 750
PRED_BASE_MORTALITY = 0.0023
PRED_AGE_MORT_ONSET = 9
PRED_AGE_MORT_RATE = 0.011

# Fairness-controlled comparison: no built-in survival/reproduction multipliers.
ALTRUIST_MORTALITY_MULT = 1.0
SELFISH_MORTALITY_MULT = 1.0
ALTRUIST_REPRO_MULT = 1.0
SELFISH_REPRO_MULT = 1.0

# Strategy inheritance
STRATEGY_MUTATION_PROB = 0.0

# Hunting
# Fairness-controlled comparison: same baseline hunting reach.
ALTRUIST_HUNT_RADIUS = 2
SELFISH_HUNT_RADIUS = 2
# Same baseline hunt success for both strategies (no hunt support bonus in this simplified copy).
ALTRUIST_HUNT_BASE_SUCCESS = 0.22
SELFISH_HUNT_SUCCESS = 0.22
HUNT_MAX_SUCCESS = 0.90
# Strict energy accounting: predator gain is tied to prey energy at capture.
# 1.0 means full conversion of prey energy into predator energy (before sharing).
HUNT_ASSIMILATION_EFFICIENCY = 1.0
HUNT_ATTEMPT_COST = 0.002
# Strict comparison: remove extra selfish-specific hunting penalty.
SELFISH_CONFLICT_COST = 0.0

# Altruistic energy transfer (transfer phase only)
# Options:
# - "altruist_only": altruists give only to nearby altruist recipients
# - "any_predator": altruists give to nearby predators regardless of strategy
TRANSFER_RECIPIENT_MODE = "any_predator"
TRANSFER_RADIUS = 1
TRANSFER_RESERVE = 1.00
TRANSFER_CHUNK = 0.30
TRANSFER_RECIPIENT_TARGET_ENERGY = 3.40
# Recipient need-threshold removed in this simplified transfer-only variant:
# recipients are filtered only by local proximity and `TRANSFER_RECIPIENT_MODE`.
MAX_TRANSFERS_PER_DONOR = 16

# Fallback foraging (low-intensity non-prey intake)
# Fairness-controlled comparison: no strategy-specific exogenous energy.
ALTRUIST_FORAGE_GAIN = 0.0
SELFISH_FORAGE_GAIN = 0.0

# Extinction / coexistence metrics
SELFISH_NEAR_ZERO_THRESHOLD = 3

# Output
OUTPUT_DIR = Path("predpreygrass_altruism/output_transfer_only_simple")
SAVE_SUMMARY_CSV = True
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_DPI = 180

# Minimal paired comparison helper for the two transfer recipient modes.
RUN_TWO_MODE_COMPARISON = False
SAVE_TWO_MODE_COMPARISON_ARTIFACTS = True
# Example archived robustness seed bases used with this helper:
# - default (`None`) -> `SEED = 122913`
# - `COMPARISON_SEED_BASE = 20260224`
# - `COMPARISON_SEED_BASE = 20270401`
COMPARISON_SEED_BASE: int | None = None
TWO_MODE_COMPARISON_MD_PATH = Path("predpreygrass_altruism/COMPARISON_TRANSFER_ONLY.md")
TWO_MODE_COMPARISON_CSV_FILENAME = "two_mode_comparison_summary.csv"

# Optional live visualization
LIVE_GRID = True
LIVE_GRID_REPLICATE_INDEX = 1
LIVE_GRID_EVERY = 1
LIVE_GRID_FPS = 4
LIVE_GRID_CELL_SIZE = 14
LIVE_GRID_SIDE_PANEL_WIDTH = 390
LIVE_GRID_PREY_RADIUS = 2
LIVE_GRID_PREDATOR_RADIUS = 4
LIVE_GRID_CHART_POINTS = 260
LIVE_GRID_CHART_HEIGHT = 68
LIVE_GRID_CHART_GAP = 8
LIVE_GRID_TERMINAL_HOLD_SEC = 3.0
LIVE_GRID_SHOW_WORLD_LEGEND = True


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Predator:
    aid: int
    x: int
    y: int
    energy: float
    age: int
    altruistic: bool
    alive: bool = True


@dataclass
class Prey:
    aid: int
    x: int
    y: int
    energy: float
    age: int
    alive: bool = True


@dataclass
class TickStats:
    altruists: int
    selfish: int
    prey: int
    mean_grass: float
    hunts: int
    cooperative_hunts: int
    transfers: int
    transferred_energy: float


@dataclass
class RunResult:
    altruist_trace: np.ndarray
    selfish_trace: np.ndarray
    prey_trace: np.ndarray
    grass_trace: np.ndarray
    freq_trace: np.ndarray
    transfer_trace: np.ndarray
    final_altruists: int
    final_selfish: int
    final_prey: int
    selfish_extinct: bool
    selfish_extinction_step: int | None
    selfish_absorbing_extinction_step: int | None
    selfish_near_zero_tail_fraction: float
    coexistence_tail_fraction: float
    predator_prey_coexistence_tail_fraction: float
    mean_tail_altruists: float
    mean_tail_selfish: float
    mean_tail_prey: float
    mean_tail_altruist_frequency: float


class LiveGridRenderer:
    """Simple pygame renderer for the predator-prey-grass grid."""

    def __init__(self, title: str) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required when LIVE_GRID=True")

        pygame.init()
        self._cell = LIVE_GRID_CELL_SIZE
        self._side_panel_w = LIVE_GRID_SIDE_PANEL_WIDTH
        self._world_w_px = W * self._cell
        self._world_h_px = H * self._cell
        self._screen = pygame.display.set_mode((self._world_w_px + self._side_panel_w, self._world_h_px))
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 16)
        self._small_font = pygame.font.SysFont("consolas", 13)

        self._alt_pred_hist: List[int] = []
        self._self_pred_hist: List[int] = []
        self._prey_hist: List[int] = []
        self._alt_freq_hist: List[float] = []

    def _grass_color(self, grass_value: float) -> Tuple[int, int, int]:
        frac = float(np.clip(grass_value / max(1e-9, GRASS_MAX), 0.0, 1.0))
        # Grayscale background so predator/prey markers carry the visual emphasis.
        v = int(42 + (96 * frac))
        return (v, v, v)

    def _poll_events(self) -> bool:
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _append_history(
        self,
        alt_pred_count: int,
        self_pred_count: int,
        prey_count: int,
        alt_frequency: float,
    ) -> None:
        self._alt_pred_hist.append(alt_pred_count)
        self._self_pred_hist.append(self_pred_count)
        self._prey_hist.append(prey_count)
        self._alt_freq_hist.append(alt_frequency)

        keep = max(12, LIVE_GRID_CHART_POINTS)
        if len(self._alt_pred_hist) > keep:
            self._alt_pred_hist = self._alt_pred_hist[-keep:]
        if len(self._self_pred_hist) > keep:
            self._self_pred_hist = self._self_pred_hist[-keep:]
        if len(self._prey_hist) > keep:
            self._prey_hist = self._prey_hist[-keep:]
        if len(self._alt_freq_hist) > keep:
            self._alt_freq_hist = self._alt_freq_hist[-keep:]

    def _draw_series_chart(
        self,
        values: List[float],
        rect: "pygame_types.Rect",
        color: Tuple[int, int, int],
        label: str,
        percent: bool = False,
    ) -> None:
        assert pygame is not None
        screen = self._screen
        pygame.draw.rect(screen, (26, 26, 34), rect)
        pygame.draw.rect(screen, (84, 84, 98), rect, 1)

        if not values:
            return

        vmax = 1.0 if percent else max(1.0, max(values))
        n = len(values)
        points: List[Tuple[int, int]] = []
        for i, v in enumerate(values):
            x = rect.left + int(i * (rect.width - 1) / max(1, n - 1))
            y = rect.bottom - 2 - int((float(v) / vmax) * max(1, rect.height - 4))
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(screen, color, False, points, 2)
        else:
            pygame.draw.circle(screen, color, points[0], 2)

        if percent:
            now_value = 100.0 * float(values[-1])
            label_text = f"{label}: {now_value:.1f}%"
        else:
            now_value = int(round(float(values[-1])))
            peak = int(round(max(values)))
            label_text = f"{label}: now={now_value} max={peak}"

        surface = self._small_font.render(label_text, True, (235, 235, 235))
        screen.blit(surface, (rect.left + 6, rect.top + 4))

    def _draw_legend_item(
        self,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        label: str,
    ) -> int:
        assert pygame is not None
        screen = self._screen
        pygame.draw.circle(screen, color, (x, y), 5)
        label_surface = self._small_font.render(label, True, (235, 235, 235))
        screen.blit(label_surface, (x + 10, y - 8))
        return x + 10 + label_surface.get_width() + 14

    def _draw_world_legend_panel(self) -> None:
        """Prominent legend overlay inside the world view for quick color decoding."""
        assert pygame is not None
        panel_x = 10
        panel_y = 10
        panel_w = 236
        panel_h = 104
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((10, 10, 14, 215))
        self._screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(self._screen, (86, 86, 102), panel_rect, 1, border_radius=8)

        title = self._font.render("Legend", True, (240, 240, 240))
        self._screen.blit(title, (panel_x + 10, panel_y + 8))

        rows = [
            ((49, 173, 255), "Altruistic predator"),
            ((245, 92, 92), "Selfish predator"),
            ((242, 220, 94), "Prey"),
        ]
        row_y = panel_y + 34
        for color, label in rows:
            pygame.draw.circle(self._screen, color, (panel_x + 16, row_y), 5)
            txt = self._small_font.render(label, True, (232, 232, 232))
            self._screen.blit(txt, (panel_x + 28, row_y - 8))
            row_y += 20

        grass_txt = self._small_font.render("Grass: darker = low, lighter = high", True, (212, 212, 212))
        self._screen.blit(grass_txt, (panel_x + 10, panel_y + panel_h - 22))

    def draw(
        self,
        grass: np.ndarray,
        predators: Dict[int, Predator],
        prey: Dict[int, Prey],
        step: int,
    ) -> bool:
        assert pygame is not None
        if not self._poll_events():
            return False

        screen = self._screen
        cell = self._cell

        alt_pred_count = sum(1 for pred in predators.values() if pred.alive and pred.altruistic)
        pred_count = len(predators)
        self_pred_count = pred_count - alt_pred_count
        prey_count = len(prey)
        grass_mean = float(np.mean(grass))
        altruist_freq = (alt_pred_count / pred_count) if pred_count > 0 else 0.0

        self._append_history(alt_pred_count, self_pred_count, prey_count, altruist_freq)

        for y in range(H):
            py = y * cell
            for x in range(W):
                px = x * cell
                pygame.draw.rect(
                    screen,
                    self._grass_color(float(grass[y, x])),
                    (px, py, cell, cell),
                )

        prey_radius = max(1, LIVE_GRID_PREY_RADIUS)
        pred_radius = max(2, LIVE_GRID_PREDATOR_RADIUS)

        for animal in prey.values():
            if not animal.alive:
                continue
            cx = animal.x * cell + (cell // 2)
            cy = animal.y * cell + (cell // 2)
            pygame.draw.circle(screen, (242, 220, 94), (cx, cy), prey_radius)

        for pred in predators.values():
            if not pred.alive:
                continue
            cx = pred.x * cell + (cell // 2)
            cy = pred.y * cell + (cell // 2)
            color = (49, 173, 255) if pred.altruistic else (245, 92, 92)
            pygame.draw.circle(screen, color, (cx, cy), pred_radius)

        pygame.draw.rect(screen, (49, 173, 255), (0, 0, self._world_w_px, self._world_h_px), 2)
        if LIVE_GRID_SHOW_WORLD_LEGEND:
            self._draw_world_legend_panel()

        side_x = self._world_w_px
        side_rect = pygame.Rect(side_x, 0, self._side_panel_w, self._world_h_px)
        pygame.draw.rect(screen, (12, 12, 16), side_rect)
        pygame.draw.rect(screen, (62, 62, 74), side_rect, 1)

        title_surface = self._font.render("Mixed Predator Strategies", True, (235, 235, 235))
        screen.blit(title_surface, (side_x + 10, 10))

        status_lines = [
            f"tick={step + 1}/{STEPS}   ESC=quit",
            f"altruists={alt_pred_count}   selfish={self_pred_count}",
            f"altruist_freq={100.0 * altruist_freq:.1f}%   prey={prey_count}",
            f"mean_grass={grass_mean:.2f}",
        ]
        status_y = 34
        for line in status_lines:
            line_surface = self._small_font.render(line, True, (224, 224, 224))
            screen.blit(line_surface, (side_x + 10, status_y))
            status_y += 18

        legend_panel_rect = pygame.Rect(side_x + 8, status_y + 4, self._side_panel_w - 16, 54)
        pygame.draw.rect(screen, (20, 20, 26), legend_panel_rect)
        pygame.draw.rect(screen, (78, 78, 92), legend_panel_rect, 1)
        legend_title = self._small_font.render("Agents", True, (231, 231, 231))
        screen.blit(legend_title, (legend_panel_rect.left + 8, legend_panel_rect.top + 5))
        legend_y = legend_panel_rect.top + 30
        lx = legend_panel_rect.left + 10
        lx = self._draw_legend_item(lx, legend_y, (242, 220, 94), "prey")
        lx = self._draw_legend_item(lx, legend_y, (49, 173, 255), "altruistic")
        self._draw_legend_item(lx, legend_y, (245, 92, 92), "selfish")

        chart_margin = 8
        chart_width = self._side_panel_w - (2 * chart_margin)
        chart_gap = max(4, LIVE_GRID_CHART_GAP)
        chart_top = legend_panel_rect.bottom + 8
        chart_area_h = self._world_h_px - chart_top - 8
        max_height_each = max(20, (chart_area_h - (3 * chart_gap)) // 4)
        chart_height = min(max_height_each, max(24, LIVE_GRID_CHART_HEIGHT))

        alt_pred_rect = pygame.Rect(side_x + chart_margin, chart_top, chart_width, chart_height)
        self_pred_rect = pygame.Rect(
            side_x + chart_margin,
            chart_top + chart_height + chart_gap,
            chart_width,
            chart_height,
        )
        prey_rect = pygame.Rect(
            side_x + chart_margin,
            chart_top + (2 * (chart_height + chart_gap)),
            chart_width,
            chart_height,
        )
        freq_rect = pygame.Rect(
            side_x + chart_margin,
            chart_top + (3 * (chart_height + chart_gap)),
            chart_width,
            chart_height,
        )

        self._draw_series_chart(
            [float(v) for v in self._alt_pred_hist],
            alt_pred_rect,
            (77, 166, 255),
            "Altruistic predators",
        )
        self._draw_series_chart(
            [float(v) for v in self._self_pred_hist],
            self_pred_rect,
            (245, 92, 92),
            "Selfish predators",
        )
        self._draw_series_chart(
            [float(v) for v in self._prey_hist],
            prey_rect,
            (250, 207, 76),
            "Prey",
        )
        self._draw_series_chart(
            self._alt_freq_hist,
            freq_rect,
            (116, 255, 201),
            "Altruist frequency",
            percent=True,
        )

        pygame.display.flip()
        self._clock.tick(LIVE_GRID_FPS)
        return True

    def close(self) -> None:
        if pygame is None:
            return
        pygame.display.quit()
        pygame.quit()

    def hold(self, seconds: float) -> bool:
        """Keep the last frame visible for a short time."""
        if pygame is None or seconds <= 0.0:
            return True
        end_ms = pygame.time.get_ticks() + int(seconds * 1000.0)
        while pygame.time.get_ticks() < end_ms:
            if not self._poll_events():
                return False
            self._clock.tick(max(1, LIVE_GRID_FPS))
        return True


# ============================================================
# HELPERS
# ============================================================

MOVE_CHOICES = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def wrap(v: int, length: int) -> int:
    return v % length


def make_offsets(radius: int, include_origin: bool = True) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not include_origin and dx == 0 and dy == 0:
                continue
            if max(abs(dx), abs(dy)) <= radius:
                offsets.append((dx, dy))
    return offsets


ALTRUIST_HUNT_OFFSETS = make_offsets(ALTRUIST_HUNT_RADIUS, include_origin=True)
SELFISH_HUNT_OFFSETS = make_offsets(SELFISH_HUNT_RADIUS, include_origin=True)
TRANSFER_OFFSETS = make_offsets(TRANSFER_RADIUS, include_origin=True)


def build_cell_index_pred(predators: Dict[int, Predator]) -> Dict[Tuple[int, int], List[int]]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for pid, pred in predators.items():
        if not pred.alive:
            continue
        out.setdefault((pred.x, pred.y), []).append(pid)
    return out


def build_cell_index_prey(prey: Dict[int, Prey]) -> Dict[Tuple[int, int], List[int]]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for aid, animal in prey.items():
        if not animal.alive:
            continue
        out.setdefault((animal.x, animal.y), []).append(aid)
    return out


def local_ids(
    x: int,
    y: int,
    offsets: List[Tuple[int, int]],
    index: Dict[Tuple[int, int], List[int]],
) -> List[int]:
    out: List[int] = []
    for dx, dy in offsets:
        out.extend(index.get((wrap(x + dx, W), wrap(y + dy, H)), []))
    return out


def mortality_hazard(age: int, base: float, onset: int, slope: float) -> float:
    age_term = max(0, age - onset)
    return float(np.clip(base + (age_term * slope), 0.0, 0.99))


def _valid_transfer_recipient_mode() -> bool:
    return TRANSFER_RECIPIENT_MODE in {"altruist_only", "any_predator"}


def predator_counts(predators: Dict[int, Predator]) -> Tuple[int, int]:
    alt = sum(1 for pred in predators.values() if pred.alive and pred.altruistic)
    total = len(predators)
    return alt, (total - alt)


def initialize_run(
    rng: np.random.Generator,
) -> Tuple[Dict[int, Predator], Dict[int, Prey], np.ndarray, int, int]:
    predators: Dict[int, Predator] = {}
    prey: Dict[int, Prey] = {}
    grass = np.full((H, W), GRASS_INIT, dtype=float)

    next_pred_id = 0
    for _ in range(PRED_INIT):
        is_altruist = bool(rng.random() < ALTRUIST_INIT_FRACTION)
        predators[next_pred_id] = Predator(
            aid=next_pred_id,
            x=int(rng.integers(0, W)),
            y=int(rng.integers(0, H)),
            age=int(rng.integers(1, PRED_REPRO_MAX_AGE)),
            energy=float(np.clip(rng.normal(PRED_ENERGY_INIT_MEAN, PRED_ENERGY_INIT_SD), 0.45, PRED_ENERGY_CAP)),
            altruistic=is_altruist,
        )
        next_pred_id += 1

    next_prey_id = 0
    for _ in range(PREY_INIT):
        prey[next_prey_id] = Prey(
            aid=next_prey_id,
            x=int(rng.integers(0, W)),
            y=int(rng.integers(0, H)),
            age=int(rng.integers(0, PREY_REPRO_MAX_AGE)),
            energy=float(np.clip(rng.normal(PREY_ENERGY_INIT_MEAN, PREY_ENERGY_INIT_SD), 0.4, PREY_ENERGY_CAP)),
        )
        next_prey_id += 1

    return predators, prey, grass, next_pred_id, next_prey_id


def prey_phase(prey: Dict[int, Prey], grass: np.ndarray, rng: np.random.Generator) -> None:
    for animal in prey.values():
        if not animal.alive:
            continue

        animal.age += 1
        if rng.random() < PREY_MOVE_PROB:
            dx, dy = MOVE_CHOICES[int(rng.integers(0, len(MOVE_CHOICES)))]
            animal.x = wrap(animal.x + dx, W)
            animal.y = wrap(animal.y + dy, H)
            animal.energy -= PREY_MOVE_COST

        bite = min(PREY_BITE, grass[animal.y, animal.x])
        grass[animal.y, animal.x] -= bite
        animal.energy = min(PREY_ENERGY_CAP, animal.energy + bite - PREY_METAB)


def predator_move_and_cost(predators: Dict[int, Predator], rng: np.random.Generator) -> None:
    for pred in predators.values():
        if not pred.alive:
            continue

        pred.age += 1
        if rng.random() < PRED_MOVE_PROB:
            dx, dy = MOVE_CHOICES[int(rng.integers(0, len(MOVE_CHOICES)))]
            pred.x = wrap(pred.x + dx, W)
            pred.y = wrap(pred.y + dy, H)
            pred.energy -= PRED_MOVE_COST

        pred.energy -= PRED_METAB
        if pred.altruistic:
            pred.energy += ALTRUIST_FORAGE_GAIN
        else:
            pred.energy += SELFISH_FORAGE_GAIN
        pred.energy = min(pred.energy, PRED_ENERGY_CAP)


def hunting_phase(
    predators: Dict[int, Predator],
    prey: Dict[int, Prey],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    prey_index = build_cell_index_prey(prey)

    hunts = 0
    cooperative_hunts = 0

    predator_ids = [pid for pid, pred in predators.items() if pred.alive]
    rng.shuffle(predator_ids)

    for pid in predator_ids:
        pred = predators[pid]
        if not pred.alive:
            continue

        hunt_offsets = ALTRUIST_HUNT_OFFSETS if pred.altruistic else SELFISH_HUNT_OFFSETS
        candidate_prey = [
            qid
            for qid in local_ids(pred.x, pred.y, hunt_offsets, prey_index)
            if prey[qid].alive
        ]
        if not candidate_prey:
            continue

        target_id = int(rng.choice(candidate_prey))
        if not prey[target_id].alive:
            continue

        if pred.altruistic:
            success_prob = ALTRUIST_HUNT_BASE_SUCCESS
        else:
            success_prob = SELFISH_HUNT_SUCCESS

        pred.energy -= HUNT_ATTEMPT_COST
        success_prob = float(np.clip(success_prob, 0.0, HUNT_MAX_SUCCESS))

        if rng.random() < success_prob:
            # Tie predator gain to the prey's current energy (no fixed reward).
            # Prey may occasionally be at/under zero before mortality cleanup, so clip at zero.
            captured_prey_energy = max(0.0, float(prey[target_id].energy))
            prey[target_id].alive = False
            gained = captured_prey_energy * HUNT_ASSIMILATION_EFFICIENCY

            pred.energy = min(PRED_ENERGY_CAP, pred.energy + gained)
            hunts += 1

    return hunts, cooperative_hunts


def transfer_phase(predators: Dict[int, Predator], rng: np.random.Generator) -> Tuple[int, float]:
    pred_index = build_cell_index_pred(predators)
    if not _valid_transfer_recipient_mode():
        raise ValueError(f"Unknown TRANSFER_RECIPIENT_MODE: {TRANSFER_RECIPIENT_MODE}")
    donor_ids = [
        pid
        for pid, pred in predators.items()
        if pred.alive and pred.altruistic and pred.energy > TRANSFER_RESERVE
    ]
    rng.shuffle(donor_ids)

    transfers = 0
    transferred_energy = 0.0

    for donor_id in donor_ids:
        donor = predators[donor_id]
        if not donor.alive:
            continue

        for _ in range(MAX_TRANSFERS_PER_DONOR):
            available = donor.energy - TRANSFER_RESERVE
            if available <= 1e-9:
                break

            recipients = []
            for rid in local_ids(donor.x, donor.y, TRANSFER_OFFSETS, pred_index):
                if rid == donor_id:
                    continue
                recipient = predators[rid]
                if not recipient.alive:
                    continue
                if TRANSFER_RECIPIENT_MODE == "altruist_only":
                    if not recipient.altruistic:
                        continue
                elif TRANSFER_RECIPIENT_MODE != "any_predator":
                    raise ValueError(f"Unknown TRANSFER_RECIPIENT_MODE: {TRANSFER_RECIPIENT_MODE}")
                recipients.append(rid)

            if not recipients:
                break

            recipient_id = int(rng.choice(recipients))
            recipient = predators[recipient_id]

            amount = min(TRANSFER_CHUNK, available, TRANSFER_RECIPIENT_TARGET_ENERGY - recipient.energy)
            if amount <= 1e-9:
                break

            donor.energy -= amount
            recipient.energy += amount
            transfers += 1
            transferred_energy += amount

    return transfers, transferred_energy


def mortality_phase(predators: Dict[int, Predator], prey: Dict[int, Prey], rng: np.random.Generator) -> None:
    for animal in prey.values():
        if not animal.alive:
            continue
        if animal.energy <= 0.0 or animal.age > PREY_MAX_AGE:
            animal.alive = False
            continue
        hazard = mortality_hazard(animal.age, PREY_BASE_MORTALITY, PREY_AGE_MORT_ONSET, PREY_AGE_MORT_RATE)
        if rng.random() < hazard:
            animal.alive = False

    for pred in predators.values():
        if not pred.alive:
            continue
        if pred.energy <= 0.0 or pred.age > PRED_MAX_AGE:
            pred.alive = False
            continue
        hazard = mortality_hazard(pred.age, PRED_BASE_MORTALITY, PRED_AGE_MORT_ONSET, PRED_AGE_MORT_RATE)
        if pred.altruistic:
            hazard *= ALTRUIST_MORTALITY_MULT
        else:
            hazard *= SELFISH_MORTALITY_MULT
        hazard = float(np.clip(hazard, 0.0, 0.99))
        if rng.random() < hazard:
            pred.alive = False


def prey_reproduction(
    prey: Dict[int, Prey],
    next_prey_id: int,
    rng: np.random.Generator,
) -> int:
    alive_prey_ids = [aid for aid, animal in prey.items() if animal.alive]
    if not alive_prey_ids:
        return next_prey_id

    crowd_factor = max(0.02, 1.0 - (len(alive_prey_ids) / PREY_MAX_POP))
    rng.shuffle(alive_prey_ids)

    for aid in alive_prey_ids:
        animal = prey[aid]
        if not animal.alive:
            continue
        if not (PREY_REPRO_MIN_AGE <= animal.age <= PREY_REPRO_MAX_AGE):
            continue
        if animal.energy < PREY_REPRO_THRESH:
            continue
        if rng.random() >= PREY_REPRO_PROB * crowd_factor:
            continue

        child_energy = animal.energy * PREY_CHILD_SHARE
        if child_energy <= 0.05:
            continue

        animal.energy -= child_energy
        prey[next_prey_id] = Prey(
            aid=next_prey_id,
            x=animal.x,
            y=animal.y,
            energy=float(child_energy),
            age=0,
        )
        next_prey_id += 1

    return next_prey_id


def predator_reproduction(
    predators: Dict[int, Predator],
    next_pred_id: int,
    rng: np.random.Generator,
) -> int:
    alive_pred_ids = [aid for aid, pred in predators.items() if pred.alive]
    if not alive_pred_ids:
        return next_pred_id

    crowd_factor = max(0.02, 1.0 - (len(alive_pred_ids) / PRED_MAX_POP))
    rng.shuffle(alive_pred_ids)

    for aid in alive_pred_ids:
        pred = predators[aid]
        if not pred.alive:
            continue
        if not (PRED_REPRO_MIN_AGE <= pred.age <= PRED_REPRO_MAX_AGE):
            continue
        if pred.energy < PRED_REPRO_THRESH:
            continue

        repro_prob = PRED_REPRO_PROB * crowd_factor
        if pred.altruistic:
            repro_prob *= ALTRUIST_REPRO_MULT
        else:
            repro_prob *= SELFISH_REPRO_MULT
        if rng.random() >= repro_prob:
            continue

        child_energy = pred.energy * PRED_CHILD_SHARE
        if child_energy <= 0.1:
            continue
        pred.energy -= child_energy

        child_strategy = pred.altruistic
        if rng.random() < STRATEGY_MUTATION_PROB:
            child_strategy = not child_strategy

        predators[next_pred_id] = Predator(
            aid=next_pred_id,
            x=pred.x,
            y=pred.y,
            energy=float(child_energy),
            age=0,
            altruistic=child_strategy,
        )
        next_pred_id += 1

    return next_pred_id


def cleanup_agents(predators: Dict[int, Predator], prey: Dict[int, Prey]) -> None:
    dead_prey = [aid for aid, animal in prey.items() if not animal.alive]
    for aid in dead_prey:
        del prey[aid]

    dead_pred = [aid for aid, pred in predators.items() if not pred.alive]
    for aid in dead_pred:
        del predators[aid]


def simulate_tick(
    predators: Dict[int, Predator],
    prey: Dict[int, Prey],
    grass: np.ndarray,
    next_pred_id: int,
    next_prey_id: int,
    rng: np.random.Generator,
) -> Tuple[int, int, TickStats]:
    grass += GRASS_REGROWTH
    np.clip(grass, 0.0, GRASS_MAX, out=grass)

    prey_phase(prey, grass, rng)
    predator_move_and_cost(predators, rng)
    hunts, cooperative_hunts = hunting_phase(predators, prey, rng)
    transfers, transferred_energy = transfer_phase(predators, rng)
    mortality_phase(predators, prey, rng)
    cleanup_agents(predators, prey)

    next_prey_id = prey_reproduction(prey, next_prey_id, rng)
    next_pred_id = predator_reproduction(predators, next_pred_id, rng)

    altruists, selfish = predator_counts(predators)
    stats = TickStats(
        altruists=altruists,
        selfish=selfish,
        prey=len(prey),
        mean_grass=float(np.mean(grass)),
        hunts=hunts,
        cooperative_hunts=cooperative_hunts,
        transfers=transfers,
        transferred_energy=transferred_energy,
    )
    return next_pred_id, next_prey_id, stats


def run_single(seed: int, live_renderer: LiveGridRenderer | None = None) -> RunResult:
    rng = np.random.default_rng(seed)
    predators, prey, grass, next_pred_id, next_prey_id = initialize_run(rng)

    altruist_trace = np.zeros(STEPS, dtype=float)
    selfish_trace = np.zeros(STEPS, dtype=float)
    prey_trace = np.zeros(STEPS, dtype=float)
    grass_trace = np.zeros(STEPS, dtype=float)
    freq_trace = np.zeros(STEPS, dtype=float)
    transfer_trace = np.zeros(STEPS, dtype=float)

    selfish_extinction_step: int | None = None
    simulated_steps = 0

    for step in range(STEPS):
        next_pred_id, next_prey_id, stats = simulate_tick(
            predators,
            prey,
            grass,
            next_pred_id,
            next_prey_id,
            rng,
        )

        altruists = stats.altruists
        selfish = stats.selfish
        predators_total = altruists + selfish

        altruist_trace[step] = altruists
        selfish_trace[step] = selfish
        prey_trace[step] = stats.prey
        grass_trace[step] = stats.mean_grass
        transfer_trace[step] = stats.transfers
        freq_trace[step] = (altruists / predators_total) if predators_total > 0 else 0.0
        simulated_steps = step + 1

        if live_renderer is not None and (step % max(1, LIVE_GRID_EVERY) == 0):
            keep_running = live_renderer.draw(grass=grass, predators=predators, prey=prey, step=step)
            if not keep_running:
                break

        if selfish == 0 and selfish_extinction_step is None:
            selfish_extinction_step = step + 1

        if predators_total == 0 or stats.prey == 0:
            if live_renderer is not None:
                live_renderer.hold(LIVE_GRID_TERMINAL_HOLD_SEC)
            break

    if simulated_steps < STEPS:
        altruist_trace[simulated_steps:] = altruist_trace[simulated_steps - 1]
        selfish_trace[simulated_steps:] = selfish_trace[simulated_steps - 1]
        prey_trace[simulated_steps:] = prey_trace[simulated_steps - 1]
        grass_trace[simulated_steps:] = grass_trace[simulated_steps - 1]
        freq_trace[simulated_steps:] = freq_trace[simulated_steps - 1]
        transfer_trace[simulated_steps:] = transfer_trace[simulated_steps - 1]

    final_altruists = int(altruist_trace[simulated_steps - 1])
    final_selfish = int(selfish_trace[simulated_steps - 1])
    final_prey = int(prey_trace[simulated_steps - 1])
    selfish_absorbing_extinction_step: int | None = None
    if final_selfish == 0:
        selfish_positive_ticks = np.flatnonzero(selfish_trace > 0)
        if selfish_positive_ticks.size == 0:
            selfish_absorbing_extinction_step = 1
        else:
            # 1-based tick index of the first zero after the last strictly positive selfish count.
            selfish_absorbing_extinction_step = int(selfish_positive_ticks[-1] + 2)

    window = min(TAIL_WINDOW, STEPS)
    tail_alt = altruist_trace[STEPS - window: STEPS]
    tail_self = selfish_trace[STEPS - window: STEPS]
    tail_prey = prey_trace[STEPS - window: STEPS]
    tail_freq = freq_trace[STEPS - window: STEPS]

    selfish_near_zero_tail = float(np.mean(tail_self <= SELFISH_NEAR_ZERO_THRESHOLD))
    coexistence_tail = float(np.mean((tail_alt > 0) & (tail_prey > 0)))
    predator_prey_coexistence_tail = float(np.mean(((tail_alt + tail_self) > 0) & (tail_prey > 0)))

    return RunResult(
        altruist_trace=altruist_trace,
        selfish_trace=selfish_trace,
        prey_trace=prey_trace,
        grass_trace=grass_trace,
        freq_trace=freq_trace,
        transfer_trace=transfer_trace,
        final_altruists=final_altruists,
        final_selfish=final_selfish,
        final_prey=final_prey,
        selfish_extinct=(final_selfish == 0),
        selfish_extinction_step=selfish_extinction_step,
        selfish_absorbing_extinction_step=selfish_absorbing_extinction_step,
        selfish_near_zero_tail_fraction=selfish_near_zero_tail,
        coexistence_tail_fraction=coexistence_tail,
        predator_prey_coexistence_tail_fraction=predator_prey_coexistence_tail,
        mean_tail_altruists=float(np.mean(tail_alt)),
        mean_tail_selfish=float(np.mean(tail_self)),
        mean_tail_prey=float(np.mean(tail_prey)),
        mean_tail_altruist_frequency=float(np.mean(tail_freq)),
    )


def aggregate_runs(runs: List[RunResult]) -> Dict[str, float | np.ndarray]:
    altruists = np.vstack([r.altruist_trace for r in runs])
    selfish = np.vstack([r.selfish_trace for r in runs])
    prey = np.vstack([r.prey_trace for r in runs])
    grass = np.vstack([r.grass_trace for r in runs])
    freq = np.vstack([r.freq_trace for r in runs])
    transfer = np.vstack([r.transfer_trace for r in runs])

    selfish_extinction_steps = [r.selfish_extinction_step for r in runs if r.selfish_extinction_step is not None]
    mean_selfish_ext_step = float(np.mean(selfish_extinction_steps)) if selfish_extinction_steps else float("nan")
    selfish_absorbing_extinction_steps = [
        r.selfish_absorbing_extinction_step for r in runs if r.selfish_absorbing_extinction_step is not None
    ]
    mean_selfish_absorbing_ext_step = (
        float(np.mean(selfish_absorbing_extinction_steps)) if selfish_absorbing_extinction_steps else float("nan")
    )

    return {
        "runs": len(runs),
        "altruist_mean_traj": np.mean(altruists, axis=0),
        "selfish_mean_traj": np.mean(selfish, axis=0),
        "prey_mean_traj": np.mean(prey, axis=0),
        "grass_mean_traj": np.mean(grass, axis=0),
        "freq_mean_traj": np.mean(freq, axis=0),
        "transfer_mean_traj": np.mean(transfer, axis=0),
        "final_altruist_mean": float(np.mean([r.final_altruists for r in runs])),
        "final_selfish_mean": float(np.mean([r.final_selfish for r in runs])),
        "final_prey_mean": float(np.mean([r.final_prey for r in runs])),
        "selfish_extinction_rate": float(np.mean([1.0 if r.selfish_extinct else 0.0 for r in runs])),
        "selfish_extinction_step_mean": mean_selfish_ext_step,
        "selfish_absorbing_extinction_step_mean": mean_selfish_absorbing_ext_step,
        "selfish_near_zero_tail_mean": float(np.mean([r.selfish_near_zero_tail_fraction for r in runs])),
        "coexistence_tail_mean": float(np.mean([r.coexistence_tail_fraction for r in runs])),
        "predator_prey_coexistence_tail_mean": float(
            np.mean([r.predator_prey_coexistence_tail_fraction for r in runs])
        ),
        "tail_altruist_mean": float(np.mean([r.mean_tail_altruists for r in runs])),
        "tail_selfish_mean": float(np.mean([r.mean_tail_selfish for r in runs])),
        "tail_prey_mean": float(np.mean([r.mean_tail_prey for r in runs])),
        "tail_altruist_frequency_mean": float(np.mean([r.mean_tail_altruist_frequency for r in runs])),
        "mean_transfers_per_tick": float(np.mean([np.mean(r.transfer_trace) for r in runs])),
    }


def print_summary(summary: Dict[str, float | np.ndarray]) -> None:
    print("\n=== Predator-Prey-Grass: Transfer-Only Altruism vs Selfish ===")
    print(f"Replicates: {int(summary['runs'])}")
    print(f"transfer_recipient_mode={TRANSFER_RECIPIENT_MODE}")
    print()
    print(
        f"final_altruists={float(summary['final_altruist_mean']):.1f}, "
        f"final_selfish={float(summary['final_selfish_mean']):.1f}, "
        f"final_prey={float(summary['final_prey_mean']):.1f}"
    )
    print(
        f"selfish_extinction_rate={float(summary['selfish_extinction_rate']):.3f}, "
        f"selfish_near_zero_tail={float(summary['selfish_near_zero_tail_mean']):.3f}, "
        f"coexistence_tail={float(summary['coexistence_tail_mean']):.3f}, "
        f"tail_altruist_frequency={100.0 * float(summary['tail_altruist_frequency_mean']):.1f}%"
    )
    print(
        f"selfish_extinction_step_mean(first_zero)={float(summary['selfish_extinction_step_mean']):.1f}, "
        f"selfish_absorbing_extinction_step_mean={float(summary['selfish_absorbing_extinction_step_mean']):.1f}"
    )
    print(
        f"propagation_tail(any predator + prey)={float(summary['predator_prey_coexistence_tail_mean']):.3f}"
    )
    print(
        f"tail_means -> altruists={float(summary['tail_altruist_mean']):.1f}, "
        f"selfish={float(summary['tail_selfish_mean']):.1f}, "
        f"prey={float(summary['tail_prey_mean']):.1f}, "
        f"mean_transfers/tick={float(summary['mean_transfers_per_tick']):.3f}"
    )
    print()

    propagation_ok = float(summary["predator_prey_coexistence_tail_mean"]) >= 0.75
    selfish_suppression_ok = (
        float(summary["selfish_near_zero_tail_mean"]) >= 0.75
        and float(summary["coexistence_tail_mean"]) >= 0.75
    )
    if propagation_ok and selfish_suppression_ok:
        print("Result direction: propagation is stable and selfish predators are suppressed.")
    elif propagation_ok:
        print("Result direction: propagation is stable, but selfish suppression is weaker / slower.")
    elif selfish_suppression_ok:
        print("Result direction: selfish suppression occurs, but overall predator-prey propagation is not stable.")
    else:
        print("Result direction: retuning may be needed for propagation and/or selfish suppression.")


def print_tuning_lock_banner(*, comparison_seed_base: int | None = None) -> None:
    """One-line banner showing the exact-split tuning target and active shared defaults."""
    seed_base = SEED if comparison_seed_base is None else comparison_seed_base
    print(
        "[tuning-lock] target: altruist_only->selfish extinct 10/10, "
        "any_predator->altruists extinct 10/10"
        f" | shared defaults: PRED_REPRO_THRESH={PRED_REPRO_THRESH:.2f}, "
        f"PRED_REPRO_PROB={PRED_REPRO_PROB:.2f}, PRED_BASE_MORTALITY={PRED_BASE_MORTALITY:.4f}, "
        f"PRED_METAB={PRED_METAB:.3f}"
        f" | transfer: reserve={TRANSFER_RESERVE:.2f}, chunk={TRANSFER_CHUNK:.2f}, "
        f"target={TRANSFER_RECIPIENT_TARGET_ENERGY:.2f}, needy_filter=removed, "
        f"max_tx={MAX_TRANSFERS_PER_DONOR}"
        f" | comparison_seed_base={seed_base}"
        " | archived_examples=122913,20260224,20270401"
    )


def run_two_mode_comparison() -> Dict[str, Dict[str, float | np.ndarray]]:
    """Run a small paired comparison between altruist-only vs indiscriminate transfer recipients."""
    modes = ("altruist_only", "any_predator")
    seed_base = SEED if COMPARISON_SEED_BASE is None else COMPARISON_SEED_BASE
    replicate_seeds = [seed_base + (7919 * i) for i in range(REPLICATES)]

    original_mode = TRANSFER_RECIPIENT_MODE
    original_live_grid = LIVE_GRID
    original_save_csv = SAVE_SUMMARY_CSV
    original_save_plots = SAVE_PLOTS
    original_show_plots = SHOW_PLOTS

    results_by_mode: Dict[str, List[RunResult]] = {}
    summaries_by_mode: Dict[str, Dict[str, float | np.ndarray]] = {}

    try:
        globals()["LIVE_GRID"] = False
        globals()["SAVE_SUMMARY_CSV"] = False
        globals()["SAVE_PLOTS"] = False
        globals()["SHOW_PLOTS"] = False

        print_tuning_lock_banner(comparison_seed_base=seed_base)
        print("\n=== Transfer Recipient Mode Comparison (paired seeds) ===")
        print(f"Modes: {modes[0]}, {modes[1]}")
        print(f"Replicates: {REPLICATES}")
        print(f"Seed base: {seed_base}")
        print(f"Seeds: {replicate_seeds}")
        print()

        for mode in modes:
            globals()["TRANSFER_RECIPIENT_MODE"] = mode
            runs = [run_single(seed=rep_seed, live_renderer=None) for rep_seed in replicate_seeds]
            summary = aggregate_runs(runs)
            results_by_mode[mode] = runs
            summaries_by_mode[mode] = summary

            print(f"[mode={mode}]")
            print(
                f"final_altruists={float(summary['final_altruist_mean']):.1f}, "
                f"final_selfish={float(summary['final_selfish_mean']):.1f}, "
                f"final_prey={float(summary['final_prey_mean']):.1f}"
            )
            print(
                f"selfish_ext_rate={float(summary['selfish_extinction_rate']):.3f}, "
                f"coex_tail={float(summary['coexistence_tail_mean']):.3f}, "
                f"prop_tail={float(summary['predator_prey_coexistence_tail_mean']):.3f}, "
                f"tail_alt_freq={100.0 * float(summary['tail_altruist_frequency_mean']):.1f}%"
            )
            print(
                f"selfish_abs_ext_step_mean={float(summary['selfish_absorbing_extinction_step_mean']):.1f}, "
                f"mean_transfers/tick={float(summary['mean_transfers_per_tick']):.3f}"
            )
            print()

        left_runs = results_by_mode[modes[0]]
        right_runs = results_by_mode[modes[1]]
        print(f"Paired per-seed finals ({modes[0]} vs {modes[1]}):")
        print("seed | left(A,S) | right(A,S)")
        for rep_seed, left, right in zip(replicate_seeds, left_runs, right_runs):
            print(
                f"{rep_seed} | ({left.final_altruists},{left.final_selfish}) | "
                f"({right.final_altruists},{right.final_selfish})"
            )

        if SAVE_TWO_MODE_COMPARISON_ARTIFACTS:
            save_two_mode_comparison_artifacts(
                modes=modes,
                seed_base=seed_base,
                replicate_seeds=replicate_seeds,
                results_by_mode=results_by_mode,
                summaries_by_mode=summaries_by_mode,
            )

    finally:
        globals()["TRANSFER_RECIPIENT_MODE"] = original_mode
        globals()["LIVE_GRID"] = original_live_grid
        globals()["SAVE_SUMMARY_CSV"] = original_save_csv
        globals()["SAVE_PLOTS"] = original_save_plots
        globals()["SHOW_PLOTS"] = original_show_plots

    return summaries_by_mode


def save_two_mode_comparison_artifacts(
    *,
    modes: Tuple[str, str],
    seed_base: int,
    replicate_seeds: Sequence[int],
    results_by_mode: Dict[str, List[RunResult]],
    summaries_by_mode: Dict[str, Dict[str, float | np.ndarray]],
) -> None:
    """Write a simple markdown + CSV summary for the paired two-mode comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_two_mode_comparison_csv(seed_base=seed_base, modes=modes, summaries_by_mode=summaries_by_mode)
    _save_two_mode_comparison_md(
        modes=modes,
        seed_base=seed_base,
        replicate_seeds=replicate_seeds,
        results_by_mode=results_by_mode,
        summaries_by_mode=summaries_by_mode,
    )


def _seed_suffixed_comparison_paths(seed_base: int) -> Tuple[Path, Path]:
    """Return markdown/csv artifact paths, suffixing non-default seed-base reports."""
    if seed_base == SEED:
        return TWO_MODE_COMPARISON_MD_PATH, OUTPUT_DIR / TWO_MODE_COMPARISON_CSV_FILENAME

    md_path = TWO_MODE_COMPARISON_MD_PATH
    md_suffixed = md_path.with_name(f"{md_path.stem}_seed_{seed_base}{md_path.suffix}")

    csv_name = Path(TWO_MODE_COMPARISON_CSV_FILENAME)
    csv_suffix = csv_name.suffix or ".csv"
    csv_suffixed = OUTPUT_DIR / f"{csv_name.stem}_seed_{seed_base}{csv_suffix}"
    return md_suffixed, csv_suffixed


def _save_two_mode_comparison_csv(
    *,
    seed_base: int,
    modes: Tuple[str, str],
    summaries_by_mode: Dict[str, Dict[str, float | np.ndarray]],
) -> None:
    _, out_path = _seed_suffixed_comparison_paths(seed_base)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "runs",
                "final_altruist_mean",
                "final_selfish_mean",
                "final_prey_mean",
                "selfish_extinction_rate",
                "selfish_extinction_step_mean",
                "selfish_absorbing_extinction_step_mean",
                "selfish_near_zero_tail_mean",
                "coexistence_tail_mean",
                "predator_prey_coexistence_tail_mean",
                "tail_altruist_mean",
                "tail_selfish_mean",
                "tail_prey_mean",
                "tail_altruist_frequency_mean",
                "mean_transfers_per_tick",
            ]
        )
        for mode in modes:
            summary = summaries_by_mode[mode]
            writer.writerow(
                [
                    mode,
                    int(summary["runs"]),
                    float(summary["final_altruist_mean"]),
                    float(summary["final_selfish_mean"]),
                    float(summary["final_prey_mean"]),
                    float(summary["selfish_extinction_rate"]),
                    float(summary["selfish_extinction_step_mean"]),
                    float(summary["selfish_absorbing_extinction_step_mean"]),
                    float(summary["selfish_near_zero_tail_mean"]),
                    float(summary["coexistence_tail_mean"]),
                    float(summary["predator_prey_coexistence_tail_mean"]),
                    float(summary["tail_altruist_mean"]),
                    float(summary["tail_selfish_mean"]),
                    float(summary["tail_prey_mean"]),
                    float(summary["tail_altruist_frequency_mean"]),
                    float(summary["mean_transfers_per_tick"]),
                ]
            )


def _save_two_mode_comparison_md(
    *,
    modes: Tuple[str, str],
    seed_base: int,
    replicate_seeds: Sequence[int],
    results_by_mode: Dict[str, List[RunResult]],
    summaries_by_mode: Dict[str, Dict[str, float | np.ndarray]],
) -> None:
    left_mode, right_mode = modes
    left_runs = results_by_mode[left_mode]
    right_runs = results_by_mode[right_mode]
    md_path, _ = _seed_suffixed_comparison_paths(seed_base)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    left_selfish_extinct = sum(1 for run in left_runs if run.final_selfish == 0)
    right_selfish_extinct = sum(1 for run in right_runs if run.final_selfish == 0)
    left_altruist_extinct = sum(1 for run in left_runs if run.final_altruists == 0)
    right_altruist_extinct = sum(1 for run in right_runs if run.final_altruists == 0)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Transfer-Only Two-Mode Comparison\n\n")
        handle.write("- Script: `predpreygrass_altruism/predpreygrass_transfer_only_altruism_vs_selfish.py`\n")
        handle.write(f"- Modes: `{left_mode}` vs `{right_mode}`\n")
        handle.write(f"- Replicates: `{len(replicate_seeds)}`\n")
        handle.write(f"- Seed base: `{seed_base}`\n")
        handle.write(f"- Seeds: `{list(replicate_seeds)}`\n\n")

        for mode in modes:
            summary = summaries_by_mode[mode]
            handle.write(f"## `{mode}`\n\n")
            handle.write(
                f"- final means: altruists={float(summary['final_altruist_mean']):.1f}, "
                f"selfish={float(summary['final_selfish_mean']):.1f}, "
                f"prey={float(summary['final_prey_mean']):.1f}\n"
            )
            handle.write(
                f"- selfish_ext_rate={float(summary['selfish_extinction_rate']):.3f}, "
                f"coex_tail={float(summary['coexistence_tail_mean']):.3f}, "
                f"prop_tail={float(summary['predator_prey_coexistence_tail_mean']):.3f}, "
                f"tail_alt_freq={100.0 * float(summary['tail_altruist_frequency_mean']):.1f}%\n"
            )
            handle.write(
                f"- selfish_abs_ext_step_mean={float(summary['selfish_absorbing_extinction_step_mean']):.1f}, "
                f"mean_transfers/tick={float(summary['mean_transfers_per_tick']):.3f}\n\n"
            )

        handle.write("## Paired Outcome Counts\n\n")
        handle.write(f"- `{left_mode}` selfish extinction count: `{left_selfish_extinct}/{len(left_runs)}`\n")
        handle.write(f"- `{right_mode}` selfish extinction count: `{right_selfish_extinct}/{len(right_runs)}`\n")
        handle.write(f"- `{left_mode}` altruist extinction count: `{left_altruist_extinct}/{len(left_runs)}`\n")
        handle.write(f"- `{right_mode}` altruist extinction count: `{right_altruist_extinct}/{len(right_runs)}`\n\n")

        handle.write("## Paired Per-Seed Finals (A,S)\n\n")
        handle.write(f"| seed | {left_mode} | {right_mode} |\n")
        handle.write("|---:|---:|---:|\n")
        for rep_seed, left_run, right_run in zip(replicate_seeds, left_runs, right_runs):
            handle.write(
                f"| {rep_seed} | ({left_run.final_altruists},{left_run.final_selfish}) | "
                f"({right_run.final_altruists},{right_run.final_selfish}) |\n"
            )


def save_summary_csv(summary: Dict[str, float | np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "summary_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "runs",
                "final_altruist_mean",
                "final_selfish_mean",
                "final_prey_mean",
                "selfish_extinction_rate",
                "selfish_extinction_step_mean",
                "selfish_absorbing_extinction_step_mean",
                "selfish_near_zero_tail_mean",
                "coexistence_tail_mean",
                "predator_prey_coexistence_tail_mean",
                "tail_altruist_mean",
                "tail_selfish_mean",
                "tail_prey_mean",
                "tail_altruist_frequency_mean",
                "mean_transfers_per_tick",
            ]
        )
        writer.writerow(
            [
                int(summary["runs"]),
                float(summary["final_altruist_mean"]),
                float(summary["final_selfish_mean"]),
                float(summary["final_prey_mean"]),
                float(summary["selfish_extinction_rate"]),
                float(summary["selfish_extinction_step_mean"]),
                float(summary["selfish_absorbing_extinction_step_mean"]),
                float(summary["selfish_near_zero_tail_mean"]),
                float(summary["coexistence_tail_mean"]),
                float(summary["predator_prey_coexistence_tail_mean"]),
                float(summary["tail_altruist_mean"]),
                float(summary["tail_selfish_mean"]),
                float(summary["tail_prey_mean"]),
                float(summary["tail_altruist_frequency_mean"]),
                float(summary["mean_transfers_per_tick"]),
            ]
        )


def plot_summary(summary: Dict[str, float | np.ndarray]) -> None:
    ticks = np.arange(1, STEPS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(ticks, summary["altruist_mean_traj"], lw=2.2, color="#2a9d8f", label="Altruistic predators")
    ax.plot(ticks, summary["selfish_mean_traj"], lw=2.2, color="#e76f51", label="Selfish predators")
    ax.set_title("Predator Strategies in One Shared World")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ticks, summary["freq_mean_traj"], lw=2.3, color="#264653")
    ax.set_title("Altruist Frequency")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(ticks, summary["prey_mean_traj"], lw=2.0, color="#6d597a", label="Prey")
    ax.plot(ticks, summary["grass_mean_traj"], lw=1.8, color="#588157", label="Mean grass")
    ax.set_title("Prey and Grass")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Level")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    labels = [
        "Selfish extinction",
        "Selfish near-zero tail",
        "Altruist-prey coexist tail",
        "Any predator+prey propagation tail",
    ]
    vals = [
        float(summary["selfish_extinction_rate"]),
        float(summary["selfish_near_zero_tail_mean"]),
        float(summary["coexistence_tail_mean"]),
        float(summary["predator_prey_coexistence_tail_mean"]),
    ]
    colors = ["#d62828", "#f77f00", "#1d3557", "#2a9d8f"]
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, vals, color=colors)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0.0, 1.02)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            min(1.0, float(val)) + 0.02,
            f"{float(val):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#e8e8e8",
        )
    ax.axhline(0.75, color="#bdbdbd", lw=1.1, ls="--", alpha=0.8)
    ax.text(
        0.01,
        0.78,
        "0.75 threshold",
        transform=ax.transAxes,
        fontsize=9,
        color="#d9d9d9",
        va="bottom",
    )
    ax.set_title("Outcome Metrics (Suppression vs Propagation)")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Predator-Prey-Grass with Heritable Altruist/Selfish Predators", y=0.995)
    fig.tight_layout()

    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "mixed_altruist_selfish_dynamics.png"
        fig.savefig(out_path, dpi=PLOT_DPI)
        print(f"Saved plot: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def run_experiment() -> Dict[str, float | np.ndarray]:
    print_tuning_lock_banner()
    results: List[RunResult] = []
    replicate_seeds = [SEED + (7919 * i) for i in range(REPLICATES)]

    for rep_idx, rep_seed in enumerate(replicate_seeds, start=1):
        render_this_run = LIVE_GRID and (rep_idx == LIVE_GRID_REPLICATE_INDEX)

        live_renderer: LiveGridRenderer | None = None
        if render_this_run:
            if pygame is None:
                raise RuntimeError("pygame is not available; set LIVE_GRID=False or install pygame")
            live_renderer = LiveGridRenderer(
                title=f"PredPreyGrass Transfer-Only (rep {rep_idx}/{REPLICATES})"
            )

        try:
            result = run_single(seed=rep_seed, live_renderer=live_renderer)
        finally:
            if live_renderer is not None:
                live_renderer.close()

        results.append(result)

        if rep_idx % REPORT_EVERY == 0:
            print(f"Completed {rep_idx}/{REPLICATES} replicates")

    summary = aggregate_runs(results)
    print_summary(summary)

    if SAVE_SUMMARY_CSV:
        save_summary_csv(summary)
        print(f"Saved summary CSV: {OUTPUT_DIR / 'summary_metrics.csv'}")

    plot_summary(summary)
    return summary


if __name__ == "__main__":
    if RUN_TWO_MODE_COMPARISON:
        run_two_mode_comparison()
    else:
        run_experiment()
