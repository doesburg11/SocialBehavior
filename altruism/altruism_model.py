"""
Altruism (NetLogo -> Python/NumPy) — Patch-only model

This is a faithful port of the NetLogo code you provided (HTML dump)
into a vectorized NumPy simulation. It mirrors the original:
- patches have three states: black (empty), green (selfish), pink (altruist)
- per-tick: altruism benefit, fitness checks, neighbor fitness recording,
  lottery weights, and next-generation updates (using 4-neighborhood)
- parameters match the NetLogo names

Usage
-----
$ python altruism_model.py               # runs a small demo
$ python altruism_model.py --steps 200 --width 101 --height 101 --seed 42
$ python altruism_model.py --plot        # shows a live matplotlib view

You can import and use the `AltruismModel` class in your own scripts.

Mapping notes (NetLogo -> Python)
---------------------------------
- pcolor: 2=pink (altruist), 1=green (selfish), 0=black (empty)
- neighbors4: implemented via array rolls (up, down, left, right)
- random-float 1.0 -> np.random.random(...)
- "clear-patch" resets the patch to black with zeros in all state vars

This file aims to match the logic found in your uploaded NetLogo code.
"""

from __future__ import annotations
import argparse
import numpy as np
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt  # optional
except Exception:
    plt = None


# Color/state mapping to mirror NetLogo pcolor categories
BLACK = 0  # empty
GREEN = 1  # selfish
PINK = 2  # altruist


@dataclass
class Params:
    width: int = 51
    height: int = 51
    torus: bool = True  # NetLogo patches wrap by default; keep True unless you want borders
    # NetLogo sliders / globals (choose your own defaults if needed)
    altruistic_probability: float = 0.39
    selfish_probability: float = 0.39
    benefit_from_altruism: float = 0.468
    cost_of_altruism: float = 0.156
    disease: float = 0.213
    harshness: float = 0.96
    seed: int | None = None


class AltruismModel:
    def __init__(self, params: Params):
        self.p = params
        if self.p.seed is not None:
            np.random.seed(self.p.seed)

        H, W = self.p.height, self.p.width

        # State arrays per patch
        self.pcolor = np.zeros((H, W), dtype=np.int8)
        self.benefit_out = np.zeros((H, W), dtype=np.float32)  # 1 for altruists, 0 for selfish
        self.altruism_benefit = np.zeros((H, W), dtype=np.float32)
        self.fitness = np.zeros((H, W), dtype=np.float32)

        self.self_weight = np.zeros((H, W), dtype=np.float32)
        self.self_fitness = np.zeros((H, W), dtype=np.float32)
        self.alt_weight = np.zeros((H, W), dtype=np.float32)
        self.alt_fitness = np.zeros((H, W), dtype=np.float32)
        self.harsh_weight = np.zeros((H, W), dtype=np.float32)
        self.harsh_fitness = np.zeros((H, W), dtype=np.float32)

        self.ticks = 0
        self.setup()

    # ---- NetLogo procedures ----

    def setup(self):
        """Equivalent to NetLogo `setup` -> initialize patches, reset ticks."""
        self.clear_all()
        self.initialize()
        self.ticks = 0

    def clear_all(self):
        """clear-all -> zero all arrays"""
        for arr in [
            self.pcolor, self.benefit_out, self.altruism_benefit, self.fitness,
            self.self_weight, self.self_fitness, self.alt_weight, self.alt_fitness,
            self.harsh_weight, self.harsh_fitness
        ]:
            arr[...] = 0

    def initialize(self):
        """NetLogo `initialize` (patch procedure)."""
        H, W = self.p.height, self.p.width
        r = np.random.random((H, W))
        # ifelse structure from NetLogo:
        # pink if r < altruistic_probability
        pink_mask = r < self.p.altruistic_probability
        # green if r < altruistic_probability + selfish_probability (and not pink)
        green_mask = (~pink_mask) & (r < (self.p.altruistic_probability + self.p.selfish_probability))
        # else black (already default)

        self.pcolor[pink_mask] = PINK
        self.benefit_out[pink_mask] = 1.0

        self.pcolor[green_mask] = GREEN
        self.benefit_out[green_mask] = 0.0

    def go(self):
        """
        NetLogo `go`:
        - stop if all patches are neither pink nor green (i.e., all black)
        - set altruism-benefit
        - perform-fitness-check
        - lottery (record-neighbor-fitness, find-lottery-weights, next-generation)
        - tick
        """
        # Stop condition: all patches != pink and != green -> all black
        if np.all((self.pcolor != PINK) & (self.pcolor != GREEN)):
            return False  # signal stop

        self._compute_altruism_benefit()
        self._perform_fitness_check()
        self._lottery()
        self.ticks += 1
        return True

    # ---- Helper methods mirroring NetLogo patch procedures ----

    def _neighbors4_sum(self, arr: np.ndarray) -> np.ndarray:
        """Sum of 4-neighbors of arr. Uses torus wrapping if enabled."""
        if self.p.torus:
            up = np.roll(arr, -1, axis=0)
            down = np.roll(arr, 1, axis=0)
            left = np.roll(arr, -1, axis=1)
            right = np.roll(arr, 1, axis=1)
            return up + down + left + right
        else:
            # Non-wrapping borders: pad with zeros and slice
            H, W = arr.shape
            out = np.zeros_like(arr)
            out[:-1, :] += arr[1:, :]   # up neighbors for all but last row
            out[1:, :] += arr[:-1, :]  # down neighbors for all but first row
            out[:, :-1] += arr[:, 1:]   # left neighbors for all but last col
            out[:, 1:] += arr[:, :-1]  # right neighbors for all but first col
            return out

    def _compute_altruism_benefit(self):
        # altruism-benefit = benefit-from-altruism * (benefit-out + sum of neighbors4 benefit-out) / 5
        neighbor_sum = self._neighbors4_sum(self.benefit_out)
        self.altruism_benefit = self.p.benefit_from_altruism * (self.benefit_out + neighbor_sum) / 5.0

    def _perform_fitness_check(self):
        # For green:  fitness = 1 + altruism-benefit
        # For pink:   fitness = (1 - cost-of-altruism) + altruism-benefit
        # For black:  fitness = harshness
        self.fitness.fill(0.0)

        green_mask = self.pcolor == GREEN
        pink_mask = self.pcolor == PINK
        black_mask = self.pcolor == BLACK

        self.fitness[green_mask] = 1.0 + self.altruism_benefit[green_mask]
        self.fitness[pink_mask] = (1.0 - self.p.cost_of_altruism) + self.altruism_benefit[pink_mask]
        self.fitness[black_mask] = self.p.harshness

    def _record_neighbor_fitness(self):
        # reset
        self.alt_fitness.fill(0.0)
        self.self_fitness.fill(0.0)
        self.harsh_fitness.fill(0.0)

        pink_mask = (self.pcolor == PINK)
        green_mask = (self.pcolor == GREEN)
        black_mask = (self.pcolor == BLACK)

        # self contribution
        self.alt_fitness[pink_mask] += self.fitness[pink_mask]
        self.self_fitness[green_mask] += self.fitness[green_mask]
        self.harsh_fitness[black_mask] += self.fitness[black_mask]

        # neighbor contributions: roll fitness masked by neighbor color
        def add_neighbor_contrib(target_arr, color_mask):
            # fitness from neighbors with given color
            src = self.fitness * color_mask
            target_arr += self._neighbors4_sum(src)

        # Call the helper for each color
        add_neighbor_contrib(self.alt_fitness, pink_mask)
        add_neighbor_contrib(self.self_fitness, green_mask)
        add_neighbor_contrib(self.harsh_fitness, black_mask)

    def _find_lottery_weights(self):
        """
        NetLogo `find-lottery-weights`:
        fitness-sum = alt_fitness + self_fitness + harsh_fitness + disease
        weights are normalized; if sum==0 -> all weights 0
        """
        fitness_sum = self.alt_fitness + self.self_fitness + self.harsh_fitness + self.p.disease
        # avoid divide-by-zero
        nz = fitness_sum > 0
        self.alt_weight.fill(0.0)
        self.self_weight.fill(0.0)
        self.harsh_weight.fill(0.0)

        self.alt_weight[nz] = self.alt_fitness[nz] / fitness_sum[nz]
        self.self_weight[nz] = self.self_fitness[nz] / fitness_sum[nz]
        self.harsh_weight[nz] = (self.harsh_fitness[nz] + self.p.disease) / fitness_sum[nz]

    def _next_generation(self):
        """
        NetLogo `next-generation`:
        - draw uniform random
        - if r < alt_weight -> pink (benefit_out=1)
        - elif r < alt_weight + self_weight -> green (benefit_out=0)
        - else clear-patch
        """
        H, W = self.p.height, self.p.width
        r = np.random.random((H, W))
        cut1 = self.alt_weight
        cut2 = self.alt_weight + self.self_weight

        to_pink = r < cut1
        to_green = (~to_pink) & (r < cut2)
        to_black = ~(to_pink | to_green)

        # pink
        self.pcolor[to_pink] = PINK
        self.benefit_out[to_pink] = 1.0

        # green
        self.pcolor[to_green] = GREEN
        self.benefit_out[to_green] = 0.0

        # black -> clear-patch
        if np.any(to_black):
            self._clear_patch_mask(to_black)

    def _clear_patch_mask(self, mask: np.ndarray):
        """NetLogo `clear-patch` for a boolean mask of patches."""
        self.pcolor[mask] = BLACK
        self.altruism_benefit[mask] = 0.0
        self.fitness[mask] = 0.0
        self.alt_weight[mask] = 0.0
        self.self_weight[mask] = 0.0
        self.harsh_weight[mask] = 0.0
        self.alt_fitness[mask] = 0.0
        self.self_fitness[mask] = 0.0
        self.harsh_fitness[mask] = 0.0
        self.benefit_out[mask] = 0.0

    def _lottery(self):
        """NetLogo `lottery` -> record-neighbor-fitness; find-lottery-weights; next-generation"""
        self._record_neighbor_fitness()
        self._find_lottery_weights()
        self._next_generation()

    # ---- Convenience ----

    def counts(self):
        """Return counts of (pink, green, black)."""
        pink = int(np.sum(self.pcolor == PINK))
        green = int(np.sum(self.pcolor == GREEN))
        black = int(np.sum(self.pcolor == BLACK))
        return pink, green, black

    def run(self, steps: int = 100, stop_when_empty: bool = True):
        history = []
        for _ in range(steps):
            cont = self.go()
            history.append(self.counts())
            if stop_when_empty and not cont:
                break
        return history

    def as_rgb(self):
        """Return an (H,W,3) array for visualization: black, green, pink."""
        H, W = self.p.height, self.p.width
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[self.pcolor == BLACK] = (0, 0, 0)
        rgb[self.pcolor == GREEN] = (0, 0.8, 0)
        rgb[self.pcolor == PINK] = (1, 0.4, 0.7)
        return rgb


def demo_plot(model: AltruismModel, steps: int = 200, interval: float = 0.01):
    if plt is None:
        print("matplotlib not available. Install it to use --plot.")
        return
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(model.as_rgb(), interpolation="nearest")
    ax.set_title("Altruism model — t=0")
    ax.set_axis_off()
    fig.tight_layout()
    for t in range(steps):
        alive = model.go()
        im.set_data(model.as_rgb())
        ax.set_title(f"Altruism model — t={model.ticks}")
        plt.pause(interval)
        if not alive:
            break
    plt.ioff()
    plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=51)
    ap.add_argument("--height", type=int, default=51)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--torus", action="store_true", default=True)
    ap.add_argument("--no-torus", dest="torus", action="store_false")
    ap.add_argument("--altruistic-probability", type=float, default=0.33)
    ap.add_argument("--selfish-probability", type=float, default=0.33)
    ap.add_argument("--benefit-from-altruism", type=float, default=0.5)
    ap.add_argument("--cost-of-altruism", type=float, default=0.2)
    ap.add_argument("--disease", type=float, default=0.0)
    ap.add_argument("--harshness", type=float, default=0.0)
    ap.add_argument("--plot", action="store_true", default=True, help="show live visualization (requires matplotlib)")
    args = ap.parse_args()

    params = Params(
        width=args.width,
        height=args.height,
        torus=args.torus,
        altruistic_probability=args.altruistic_probability,
        selfish_probability=args.selfish_probability,
        benefit_from_altruism=args.benefit_from_altruism,
        cost_of_altruism=args.cost_of_altruism,
        disease=args.disease,
        harshness=args.harshness,
        seed=args.seed,
    )
    model = AltruismModel(params)

    if args.plot:
        demo_plot(model, steps=args.steps)
    else:
        hist = model.run(steps=args.steps)
        for t, (pink, green, black) in enumerate(hist, start=1):
            print(f"t={t:04d}  pink={pink:5d}  green={green:5d}  black={black:5d}")


if __name__ == "__main__":
    main()
