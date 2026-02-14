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

- `predprey_public_goods/sweep_coop_cost_p0.py`

Animation layers:

- Base: local cooperation field.
- Overlay: prey density (log-scaled, zeros masked).
- Predators: open-circle markers with edge color = cooperation trait.

------------------------------------------------------------------------

# 12. Reproduction of Results

From repo root:

```bash
python predprey_public_goods/emerging_cooperation.py
python predprey_public_goods/sweep_coop_cost_p0.py
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
- Initial populations: `PRED_INIT=250`, `PREY_INIT=600`
- Steps: `STEPS=2500`
- Predator costs: `METAB_PRED=0.06`, `MOVE_COST=0.008`, `COOP_COST=0.20`
- Predator reproduction: `BIRTH_THRESH_PRED=3.0`, `LOCAL_BIRTH_R=1`
- Mutation: `MUT_RATE=0.03`, `MUT_SIGMA=0.08`
- Hunt: `HUNT_R=1`, `P0=0.18`, `KILL_ENERGY=4.0`
- Prey: `PREY_MOVE_PROB=0.25`, `PREY_REPRO_PROB=0.04`, `PREY_MAX=1600`
- Clustering radius: `CLUST_R=2`

Defaults in `predprey_public_goods/sweep_coop_cost_p0.py`:

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

Let `c_i in [0,1]` be individual cooperation in local hunting group `g`,
with:

`S_g = sum_{j in g} c_j`

Kill probability:

`p_kill(S_g) = 1 - (1 - p0)^(S_g)`

Current reward rule (equal sharing among local hunters):

`G_i = p_kill(S_g) * E / n_g`

Cost per tick:

`C_i = kappa c_i`

Fitness proxy:

`W_i = (E / n_g) * [1 - (1 - p0)^(S_g)] - kappa c_i`

Selection gradient:

`dW_i/dc_i = (E / n_g) * (1 - p0)^(S_g) * ln(1 / (1 - p0)) - kappa`

Condition for local selection favoring more cooperation:

`(E / n_g) * (1 - p0)^(S_g) * ln(1 / (1 - p0)) > kappa`

Because the benefit term decreases with `S_g` (saturation) while cost remains
linear, interior cooperation regimes are expected in broad parameter regions.
