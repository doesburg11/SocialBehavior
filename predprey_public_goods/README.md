# Predator--Prey Cooperation Model Results

## With Formal Evolutionary Interpretation

This document summarizes the current predator--prey cooperation results
and provides a theoretical interpretation using:

- Hamilton's Rule (kin assortment framing)
- Multilevel Selection
- Price Equation
- Public Goods Game Structure
- Spatial Assortment

## Contents

1. Ecological Dynamics
2. Evolutionary Dynamics
3. Hamilton's Rule Interpretation
4. Multilevel Selection Perspective
5. Price Equation Formulation
6. Spatial Assortment
7. Public Goods Game Structure (Current Implementation)
8. Trait Reference View (Selected Chart)
9. Adaptive Parameter Sweep (`COOP_COST` x `P0`)
10. Interpretation of the Full System
11. Visualization Notes
12. Reproduction of Results
13. Key Parameter Settings
14. Next Directions
15. Mathematical Derivation (Current Reward Rule)
16. Simulation Logic (Code-Level)
17. One-Tick Worked Example (Visual)
18. Comparison vs MARL Stag-Hunt (Updated)

------------------------------------------------------------------------

# 1. Ecological Dynamics

## Population Oscillations

<p align="center">
  <img src="../assets/predprey_public_goods/01_population_oscillations.png" alt="Population Oscillations" width="400">
</p>

Observed in the current chart:

- Predator and prey counts oscillate over the full run.
- Cycles are irregular in period and amplitude.
- Peaks are lagged (prey rises are typically followed by predator rises).

This is consistent with spatially perturbed Lotka--Volterra-like coupling.

## Phase Plot (Predators vs Prey)

<p align="center">
  <img src="../assets/predprey_public_goods/02_phase_plot.png" alt="Variance Heatmap" width="400">
</p>

Observed in the current chart:

- The phase path forms nested and crossing loops (not a single closed orbit).
- Dynamics remain bounded but noisy.
- The system shows persistent oscillatory coupling rather than immediate collapse.

------------------------------------------------------------------------

# 2. Evolutionary Dynamics

## Trait Evolution: Mean Cooperation

![Mean Cooperation](../assets/predprey_public_goods/03_trait_mean.png)

Observed in the current chart:

- Mean cooperation starts near 0.5.
- It rises transiently (around 0.7), then declines.
- Long-run behavior stays in an intermediate band (roughly 0.43-0.52).

This run does not show fixation at 1.0.

## Trait Variance

<p align="center">
  <img src="../assets/predprey_public_goods/04_trait_variance.png" alt="Variance Heatmap" width="400">
</p>

Observed in the current chart:

- Variance drops quickly early in the run.
- It remains low with occasional transient spikes.

This indicates concentration around an intermediate trait region, not full
collapse to a pure 1.0-cooperation state.

------------------------------------------------------------------------

# 3. Hamilton's Rule Interpretation

Let cooperation level be trait `c`.

Hamilton's inequality in reduced form is:

`r b > c`

where:

- `r` is local assortment/relatedness-like structure,
- `b` is the marginal group benefit from additional contribution,
- `c` is the individual cost of contribution.

In this model:

- Local birth and movement structure can keep positive assortment.
- Benefit saturation is controlled by the hunt function via `P0`.
- Per-tick cooperation cost (`COOP_COST * coop`) provides direct individual cost.

Because `b` is state-dependent (group composition and `P0`), the net selection
gradient is not globally positive. That matches the observed intermediate regime.

------------------------------------------------------------------------

# 4. Multilevel Selection Perspective

A standard decomposition is:

`Delta z = Cov_group(W_g, z_g)/Wbar + E_g[Cov_ind(W_i, z_i)]/Wbar`

Interpretation for this system:

- Between-group component: more cooperative local groups can convert prey
  encounters into energy more reliably.
- Within-group component: each individual pays its own cooperation cost while
  reward is shared at group level.

This naturally allows mixed outcomes where cooperation is maintained but does
not necessarily fix at 1.0.

------------------------------------------------------------------------

# 5. Price Equation Formulation

At population level:

`Delta z = Cov(W, z)/Wbar + E(W Delta z_transmission)/Wbar`

With mutation, spatial turnover, and ecological fluctuations, the covariance term
can be positive in some states and weak/negative in others.

Empirically, the current trajectory is consistent with:

- sustained nonzero covariance favoring cooperation early,
- followed by an interior regime where costs and saturated benefits balance.

------------------------------------------------------------------------

# 6. Spatial Assortment

## Local Clustering Heatmap

<p align="center">
  <img src="../assets/predprey_public_goods/05_clustering_heatmap.png" alt="Clustering Heatmap" width="400">
</p>

Observed in the current chart:

- Predators occupy clustered patches rather than a uniform field.
- Many dark regions are predator-empty neighborhoods.
- Occupied patches show intermediate-to-high local cooperation values.

## Live Grid Snapshot

<p align="center">
  <img src="../assets/predprey_public_goods/06_live_grid.png" alt="Live Grid" width="400">
</p>

Observed in the current snapshot:

- Predator trait colors are mostly mid-range.
- Prey density and predator occupancy are spatially heterogeneous.
- Spatial structure and trait structure are visibly coupled.

------------------------------------------------------------------------

# 7. Public Goods Game Structure (Current Implementation)

The implemented hunt rule is public sharing among local hunters:

- Kill probability increases with summed local contribution.
- If a kill occurs, energy is split equally among hunters in that local group.
- Each predator still pays its own cooperation cost every tick.

This creates a social-dilemma-like tension:

- Group performance improves with higher total contribution.
- Individual marginal incentive can weaken as group contribution grows.

That mechanism is compatible with stable intermediate cooperation.

------------------------------------------------------------------------

# 8. Trait Reference View (Selected Chart)

<p align="center">
  <img src="../assets/predprey_public_goods/03_trait_mean.png" alt="Trait mean" width="400">
</p>

This selected reference chart shows:

- an early transient increase,
- then long-run intermediate cooperation,
- with no terminal drop-to-zero event in this figure.

------------------------------------------------------------------------

# 9. Adaptive Parameter Sweep (`COOP_COST` x `P0`)

Sweep outputs currently used here are in `predprey_public_goods/images/`.
Metric per cell: mean cooperation over tail window, averaged across successful runs.

## Round 1 (broad scan)

<p align="center">
  <img src="../assets/predprey_public_goods/coop_cost_p0_heatmap_r1.png" alt="[Sweep Round 1" width="400">
</p>

Observed pattern:

- Co-existence of predators and prey emerges at various mean cooperation
- Highest mean cooperation appears at low `COOP_COST`, low `P0`.
- Cooperation generally decreases as either `COOP_COST` or `P0` increases.
- Gray regions indicate cells without enough successful runs.

## Round 2 (high-cost/high-`P0` refinement)

<p align="center">
  <img src="../assets/predprey_public_goods/coop_cost_p0_heatmap_r2.png" alt="[Sweep Round 12" width="400">
</p>

Observed pattern:

- Cooperation is mostly low-to-moderate in this region.
- Local stochastic pockets exist, but no broad high-cooperation band appears.

## Round 3 (moderate-`P0`, lower-cost refinement)

<p align="center">
  <img src="../assets/predprey_public_goods/coop_cost_p0_heatmap_r3.png" alt="[Sweep Round 3" width="540">
</p>

Observed pattern:

- Cooperation is generally higher than in Round 2.
- A broad intermediate band (roughly 0.35-0.55) is visible, with local peaks.

Important limit of interpretation:

- These heatmaps do not identify a single minimum cooperation threshold needed
  for coexistence.
- They summarize cooperation levels in successful finite-horizon runs; they are
  not direct equilibrium-threshold maps.

------------------------------------------------------------------------

# 10. Interpretation of the Full System

Current combined evidence supports:

- persistent predator--prey oscillations,
- non-fixating intermediate cooperation in the baseline trait trajectory,
- spatial clustering that shapes both ecology and selection,
- parameter-dependent cooperation regimes in sweep analysis.

The system is best interpreted as state-dependent selection under ecological
feedbacks, rather than a globally monotonic drive to full cooperation.

------------------------------------------------------------------------

# 11. Visualization Notes

Core ecology/trait figures are generated from:

- `predprey_public_goods/emerging_cooperation.py`

Sweep figures are generated from:

- `predprey_public_goods/sweep_p0_prey_repro.py`

Animation layers:

- Base: local cooperation field.
- Overlay: prey density (log-scaled, zeros masked).
- Predators: open-circle markers with edge color = cooperation trait.

------------------------------------------------------------------------

# 12. Reproduction of Results

From repo root:

```bash
./.conda/bin/python predprey_public_goods/emerging_cooperation.py
./.conda/bin/python predprey_public_goods/sweep_p0_prey_repro.py
```

Notes:

- Sweep images are saved under `predprey_public_goods/images/`.
- Baseline plots are shown interactively unless you add explicit save logic.
- For deterministic baselines, set `SEED` in
  `predprey_public_goods/emerging_cooperation.py`.

------------------------------------------------------------------------

# 13. Key Parameter Settings

Defaults in `predprey_public_goods/emerging_cooperation.py`:

- Grid: `W=60`, `H=60`
- Initial populations: `PRED_INIT=100`, `PREY_INIT=1400`
- Predator initial energy: `PRED_ENERGY_INIT=1.7`
- Steps: `STEPS=2500`
- Predator costs: `METAB_PRED=0.052`, `MOVE_COST=0.008`, `COOP_COST=0.09`
- Predator reproduction: `BIRTH_THRESH_PRED=4.2`, `PRED_REPRO_PROB=0.10`,
  `PRED_MAX=800`, `LOCAL_BIRTH_R=1`
- Mutation: `MUT_RATE=0.03`, `MUT_SIGMA=0.08`
- Hunt: `HUNT_RULE="energy_threshold_gate"`, `HUNT_R=1`,
  `HUNTER_POOL_R=1`, `P0=0.24`, `KILL_ENERGY=3.8`, `COOP_POWER_FLOOR=0.35`
- Prey: `PREY_MOVE_PROB=0.25`, `PREY_REPRO_PROB=0.058`, `PREY_MAX=3200`,
  `PREY_METAB=0.05`, `PREY_MOVE_COST=0.01`, `PREY_BIRTH_THRESH=2.0`,
  `PREY_BIRTH_SPLIT=0.36`, `PREY_BITE_SIZE=0.24`
- Grass: `GRASS_INIT=0.8`, `GRASS_MAX=3.0`, `GRASS_REGROWTH=0.055`
- Clustering radius: `CLUST_R=2`

Defaults in `predprey_public_goods/sweep_p0_prey_repro.py`:

- `COOP_COST` range: `0.00-1.00` (step `0.01`)
- `P0` range: `0.00-1.00` (step `0.01`)
- `successes=10`, `max_attempts=100`, `tail_window=200`
- Adaptive refinement: `rounds=3`, `top_k=5`, `refine_step_factor=0.5`

------------------------------------------------------------------------

# 14. Next Directions

- Add an explicit coexistence probability map (`Pr[survival to T]`) alongside
  mean cooperation maps.
- Track and report extinction boundary curves in (`COOP_COST`, `P0`) space.
- Estimate effective assortment `r(t)` directly from local trait correlation.
- Compare single-seed trajectories against multi-seed confidence intervals.
- Add optional deterministic export pipeline for baseline figures.

------------------------------------------------------------------------

# 15. Mathematical Derivation (Current Reward Rule)

This section summarizes the implemented hard-gate reward logic.

For a local hunter set `g` around a candidate prey:

- Trait sum: `S_g = sum_{j in g} c_j`
- Cooperative power:
  `P_g = sum_{j in g} e_j * [alpha + (1 - alpha) c_j]`,
  with `alpha = COOP_POWER_FLOOR`
- Prey energy threshold: `E_prey`

Gate 1 (hard constraint):

`P_g >= E_prey`

Gate 2 (probabilistic success in `energy_threshold_gate` mode):

`p_kill(S_g) = 1 - (1 - p0)^(S_g)`

If both gates pass, kill energy is shared equally:

`G_i = KILL_ENERGY / n_g`

Per-tick predator cost:

`C_i = METAB_PRED + MOVE_COST + COOP_COST * c_i`

A local fitness proxy under gate mode is therefore:

`W_i ~ I(P_g >= E_prey) * p_kill(S_g) * (KILL_ENERGY / n_g) - C_i`

Because benefits are both thresholded and saturating while costs are linear in
`c_i`, interior cooperation regimes remain expected.

------------------------------------------------------------------------

# 16. Simulation Logic (Code-Level)

This section documents the exact update order used in
`predprey_public_goods/emerging_cooperation.py`.

## State Variables

- Predator agent: `(x, y, energy, coop)` where `coop in [0,1]`.
- Prey agent: `(x, y, energy)`.
- Grass field: per-cell energy `grass[y, x]`.
- Space is a wrapped torus (`wrap`), so movement beyond an edge re-enters on
  the opposite side.

## Per-Tick Update Order

1. Grass regrowth (`GRASS_REGROWTH`, capped by `GRASS_MAX`).
2. Prey movement, energy costs, grass consumption, prey reproduction.
3. Build spatial indexes for prey and predators.
4. Predator hunting and energy gain.
5. Remove killed prey.
6. Predator costs, movement, reproduction, mutation, death.

## Prey Dynamics

- Each prey moves with probability `PREY_MOVE_PROB` by a local step in
  `{ -1, 0, 1 }` for x and y.
- Each prey pays `PREY_METAB` and (if moved) `PREY_MOVE_COST`.
- Each prey consumes grass at its cell up to `PREY_BITE_SIZE`.
- Prey with `energy <= 0` are removed.
- Reproduction is density-limited by:
  `repro_scale = max(0, 1 - prey_count / PREY_MAX)`.
- Birth is energy-gated (`energy >= PREY_BIRTH_THRESH`) and stochastic:
  `PREY_REPRO_PROB * repro_scale`.
- On birth, child gets `PREY_BIRTH_SPLIT * parent_energy` and the parent loses
  that energy.

## Hunting Logic

- Predators are grouped by current cell.
- For each predator-occupied cell `(cx, cy)`, candidate prey are collected from
  all cells in a square neighborhood radius `HUNT_R` (Chebyshev radius).
- Hunters are pooled around each victim using `HUNTER_POOL_R`.
- Hard gate: cooperative weighted power must exceed prey energy.
- In `energy_threshold_gate` mode, an additional probabilistic gate is applied:
  `p_kill = 1 - (1 - P0)^S` with `S = sum(coop_i)`.
- If a kill occurs, one prey candidate is removed.
- Kill reward is shared equally among hunters:
  `share = KILL_ENERGY / n_hunters`.

## Predator Energy, Reproduction, Mutation

- Each predator pays per tick:
  `METAB_PRED + MOVE_COST + COOP_COST * coop`.
- Predators then move by a local wrapped step.
- Reproduction is thresholded and probabilistic:
  `energy >= BIRTH_THRESH_PRED` and
  `random < PRED_REPRO_PROB * pred_repro_scale`.
- `pred_repro_scale` includes predator crowding (`PRED_MAX`) and prey
  availability (`len(preys) / PREY_INIT`).
- On reproduction, parent energy is halved; child inherits parent trait and
  local position.
- Child mutates with probability `MUT_RATE`:
  `coop_child = clamp01(coop_parent + Normal(0, MUT_SIGMA))`.
- Predators with `energy <= 0` are removed.

## Run Termination and Outputs

- A run stops early if either predators or prey go extinct (`pred_n == 0` or
  `prey_n == 0`); this is an extinction run.
- A run is marked successful only if no extinction occurs before `STEPS`.
- With `RESTART_ON_EXTINCTION=True`, `main()` retries up to `MAX_RESTARTS`.
- Recorded outputs include:
  predator count history, prey count history, mean/variance cooperation history,
  optional animation snapshots, final predator list, `success` flag, and
  `extinction_step`.

------------------------------------------------------------------------

# 17. One-Tick Worked Example (Visual)

This diagram visualizes one concrete tick using the same numeric example used
to explain the update logic.

![One Tick Worked Example](../assets/predprey_public_goods/tick_logic_example.svg)

## Gridworld View of the Same Tick

This version shows the same numerical example in a concrete local grid:

- Predators `A,B,C` occupy one cell.
- The highlighted blue square is the `HUNT_R=1` neighborhood used to collect
  prey candidates.
- Left panel: before hunt (all candidate prey present).
- Right panel: after hunt, where one candidate prey is removed because
  `draw < p_kill`.

![One Tick Gridworld](../assets/predprey_public_goods/tick_logic_gridworld.svg)

To regenerate:

```bash
./.conda/bin/python predprey_public_goods/visualize_tick_logic.py
```

------------------------------------------------------------------------

# 18. Comparison vs MARL Stag-Hunt (Updated)

This project intentionally keeps one core difference from
`predpreygrass/rllib/stag_hunt_forward_view`:

- Nature-focused cooperation here: cooperation is a heritable trait (`coop`).
- Nurture-focused cooperation there: cooperation is an action decision
  (`join_hunt`) each step.

What is now aligned more closely with the MARL ecology:

- Prey have explicit energy household and can starve.
- Grass is explicit, regrows each tick, and is consumed by prey.
- Predator reproduction is energy-driven with additional regulation.
- Cooperative hunt uses local pooling plus energy-threshold gating.

What still differs (beyond the intended trait-vs-action distinction):

- No explicit `join_cost` / scavenger free-rider payoff split.
- No RL action/observation API or per-agent termination/truncation outputs.
- No bounded-grid wall/LOS movement constraints.
- Single-species predator + scalar trait evolution, rather than typed MARL agent
  populations.
