# Predator--Prey Cooperation Model Results

## With Formal Evolutionary Interpretation

This document summarizes the results of the spatial predator--prey model
with evolving continuous cooperation traits, and provides a formal
evolutionary interpretation using:

-   Hamilton's Rule (Kin Selection)
-   Multilevel Selection
-   Price Equation
-   Public Goods Game Structure
-   Spatial Assortment Theory

## Contents

1. Ecological Dynamics
2. Evolutionary Dynamics
3. Hamilton's Rule Interpretation
4. Multilevel Selection Perspective
5. Price Equation Formulation
6. Spatial Assortment
7. Public Goods Game Structure (Contributors-Only Sharing)
8. Public-Goods Sharing (Social-Dilemma Regime)
9. Adaptive Parameter Sweep (`COOP_COST` x `P0`)
10. Interpretation of the Full System
11. Visualization Notes
12. Reproduction of Results
13. Key Parameter Settings
14. Next Directions
15. Mathematical Derivation of the Selection Gradient

------------------------------------------------------------------------

# 1. Ecological Dynamics

## Population Oscillations

![Population Oscillations](../assets/predprey_public_goods/01_population_oscillations.png)

Classic predator--prey oscillations are observed:

-   Prey peaks precede predator peaks
-   Predator crashes allow prey recovery
-   System remains dynamically stable

This confirms ecological coupling resembling Lotka--Volterra dynamics:

dPrey/dt = rP - a P C

dPred/dt = b a P C - m C

Where: - P = prey density - C = predator density - a = encounter rate -
b = conversion efficiency - m = predator mortality

Spatial structure modifies these equations locally but preserves
cyclical structure.

## Phase Plot (Predators vs Prey)

![Phase Plot](../assets/predprey_public_goods/02_phase_plot.png)

The phase trajectory forms a stable loop rather than spiraling to
extinction or exploding without bound. This supports the interpretation
that the system settles into a resilient oscillatory regime rather than
chaotic collapse.

------------------------------------------------------------------------

# 2. Evolutionary Dynamics

## Trait Evolution: Mean Cooperation

![Mean Cooperation](../assets/predprey_public_goods/03_trait_mean.png)

Mean cooperation rapidly increases toward fixation (≈1.0).

This indicates strong directional selection:

∂W/∂c \> 0 for all c ∈ \[0,1\]

where W is fitness and c is cooperation level.

------------------------------------------------------------------------

## Trait Variance Collapse

![Variance](../assets/predprey_public_goods/04_trait_variance.png)

Variance approaches zero → near monomorphic cooperative population.

No balancing selection or polymorphism is maintained under current
parameters.

------------------------------------------------------------------------

# 3. Hamilton's Rule Interpretation

Cooperation increases kill probability:

p_kill = 1 − (1 − p0)\^(Σ c_i)

Energy is shared among contributors only.

The fitness benefit of cooperation depends on:

-   b = marginal increase in survival/reproduction due to improved
    hunting success
-   c = energetic cost of cooperation
-   r = spatial assortment (probability that neighbors share similar
    trait)

Hamilton's Rule:

r b \> c

In this simulation:

-   Local reproduction creates high spatial assortment r
-   Benefits b are strong due to nonlinear kill function
-   Costs c are moderate

Thus:

r b − c \> 0

for most of trait space → cooperation is strictly advantageous.

This explains rapid fixation.

------------------------------------------------------------------------

# 4. Multilevel Selection Perspective

We can decompose selection into:

Δz = Cov_group(W_g, z_g)/W̄ + E_g\[Cov_ind(W_i, z_i)\]/W̄

Where:

-   z = cooperation trait
-   W_g = group fitness (local predator cluster success)
-   W_i = individual fitness

In this model:

-   Groups with higher mean cooperation have higher hunting success.
-   Within-group selection is weak because defectors do not receive
    shared rewards.

Therefore, between-group selection dominates.

This is a multilevel cooperation regime.

------------------------------------------------------------------------

# 5. Price Equation Formulation

The evolutionary change in cooperation can be written as:

Δz = Cov(W, z) / W̄

Since high-cooperation individuals consistently receive more energy and
reproduce more, the covariance term remains positive across time.

Thus cooperation monotonically increases.

Variance collapses because directional selection pushes toward boundary
condition c = 1.

------------------------------------------------------------------------

# 6. Spatial Assortment

Local clustering heatmap:

![Clustering Heatmap](../assets/predprey_public_goods/05_clustering_heatmap.png)

Spatial reproduction ensures that offspring remain near parents.

Assortment coefficient α \> 0 emerges endogenously.

Effective relatedness:

r ≈ Corr(z_i, z_neighbors)

High r enables Hamilton's condition.

## Live Grid Snapshot (Final Frame)

![Live Grid](../assets/predprey_public_goods/06_live_grid.png)

The final-frame visualization layers multiple signals for interpretation:

-   Background encodes local cooperation field.
-   Overlay shows prey density (log-scaled to prevent dense patches from washing out the map).
-   Predators are open circles with edge color mapped to cooperation level.

This makes it easier to see how spatial structure and trait values co-vary.

------------------------------------------------------------------------

# 7. Public Goods Game Structure

This system resembles a spatial public goods game with exclusion:

-   Contributions increase group success
-   Only contributors receive reward
-   Defectors receive no benefit

This is not a Prisoner's Dilemma.

It is closer to a coordination game where cooperation strictly
dominates.

Thus fixation is expected.

------------------------------------------------------------------------

# 8. Public-Goods Sharing (Social-Dilemma Regime)

Here we switch to *pure public sharing*: all hunters share the kill
equally, regardless of contribution. With this change and a suitable
cost/benefit balance, cooperation no longer fixates.

![Public-Goods Mean Cooperation](../assets/predprey_public_goods/07_trait_mean_public_goods.png)

Mean cooperation stabilizes at an intermediate level (~0.5-0.65) rather
than fixing at 1.0. This is a stable social-dilemma regime: increasing
cooperation improves group outcomes, but individual costs prevent full
cooperation. The resulting equilibrium reflects persistent coexistence
of higher and lower contributors.

Example parameters for the figure above: `P0 = 0.14`, `COOP_COST = 0.15`
(public sharing; no exclusion).

------------------------------------------------------------------------

# 9. Adaptive Parameter Sweep (`COOP_COST` x `P0`)

To map where intermediate cooperation is stable, we sweep `COOP_COST`
and `P0` and average tail cooperation over successful runs only.

## Round 1 (coarse grid)

![Sweep Round 1](../assets/predprey_public_goods/coop_cost_p0_heatmap_r1.png)

## Round 2 (refined window)

![Sweep Round 2](../assets/predprey_public_goods/coop_cost_p0_heatmap_r2.png)

## Round 3 (refined window)

![Sweep Round 3](../assets/predprey_public_goods/coop_cost_p0_heatmap_r3.png)

## Final exported map

![Sweep Final](../assets/predprey_public_goods/coop_cost_p0_heatmap.png)

Observed patterns from the updated charts:

-   The coarse scan shows a strong gradient in `P0`: lower `P0` keeps
    higher mean cooperation, while higher `P0` pushes the system toward
    lower-intermediate cooperation in this regime.
-   Adaptive refinement concentrates on a robust intermediate-cooperation
    basin around `P0 ≈ 0.21-0.25` and `COOP_COST ≈ 0.165-0.20`.
-   Within that basin, most cells remain in the mid range
    (`mean_coop ≈ 0.35-0.50`) with local stochastic pockets above and
    below.
-   Gray cells indicate parameter points where the target number of
    successful runs was not reached.

------------------------------------------------------------------------

# 10. Interpretation of the Full System

Across sharing rules, the model demonstrates:

-   Stable ecological oscillations (predator-prey coupling).
-   Under contributors-only sharing: strong directional selection and
    near-fixation of cooperation.
-   Under public sharing with tuned costs/benefits: stable intermediate
    cooperation instead of fixation.
-   Spatial clustering that reinforces assortment and shapes trait
    dynamics.

------------------------------------------------------------------------

# 11. Visualization Notes

Core dynamics plots and the live grid snapshot are generated by
`predprey_public_goods/emerging_cooperation.py`.

Adaptive sweep heatmaps are generated by
`predprey_public_goods/sweep_coop_cost_p0.py`.

The animation layers are:

-   Base layer: local cooperation heatmap (clustering metric).
-   Overlay: prey density heatmap with log scaling (zeros masked).
-   Predators: open circles with edge color mapped to cooperation level.

The animation is capped at `ANIM_STEPS` for clarity, while summary plots
use the full simulation horizon.

------------------------------------------------------------------------

# 12. Reproduction of Results

To regenerate the dynamics plots and snapshot figures:

```bash
python predprey_public_goods/emerging_cooperation.py
```

To regenerate the adaptive sweep heatmaps:

```bash
python predprey_public_goods/sweep_coop_cost_p0.py
```

If you want deterministic plots, set `SEED` in the script to a fixed
integer before running.

------------------------------------------------------------------------

# 13. Key Parameter Settings (Simulation Run)

These settings define the ecological and evolutionary pressures used to
generate the baseline figures. Full definitions live in
`predprey_public_goods/emerging_cooperation.py`.

-   Grid: `W=60`, `H=60`
-   Initial populations: `PRED_INIT=250`, `PREY_INIT=600`
-   Steps: `STEPS=2500`
-   Predator energetics: `METAB_PRED=0.06`, `MOVE_COST=0.008`, `COOP_COST=0.20`
-   Predator reproduction: `BIRTH_THRESH_PRED=3.0`, `LOCAL_BIRTH_R=1`
-   Mutation: `MUT_RATE=0.03`, `MUT_SIGMA=0.08`
-   Hunt mechanics: `HUNT_R=1`, `P0=0.18`, `KILL_ENERGY=4.0`
-   Prey dynamics: `PREY_MOVE_PROB=0.25`, `PREY_REPRO_PROB=0.04`, `PREY_MAX=1600`
-   Clustering radius: `CLUST_R=2`

Adaptive sweep defaults (`predprey_public_goods/sweep_coop_cost_p0.py`):

-   `COOP_COST` range: `0.15-0.20` (`step=0.01`)
-   `P0` range: `0.05-0.35` (`step=0.01`)
-   `successes=10`, `max_attempts=100`, `tail_window=200`
-   Adaptive rounds: `rounds=3`, `top_k=5`, `refine_step_factor=0.5`

------------------------------------------------------------------------

# 14. Next Directions (If Desired)

To generate richer evolutionary dynamics:

-   Allow defectors partial benefit (introduce stronger temptation).
-   Add diminishing returns to cooperation.
-   Add explicit risk cost for high cooperation.
-   Introduce competing predator morphs with distinct hunt strategies.
-   Measure and report an explicit assortment coefficient over time.

------------------------------------------------------------------------

# 15. Mathematical Derivation of the Selection Gradient


Let \(c_i \in [0,1]\) be individual cooperation, and for the local hunting group \(g\):

$$
S_g = \sum_{j \in g} c_j = c_i + S_{-i}
$$

Kill probability:

$$
p_{\text{kill}}(S_g) = 1 - (1 - p_0)^{S_g}
$$

Expected energy gain (contributors-only proportional sharing):

$$
G_i = p_{\text{kill}}(S_g)\,E\,\frac{c_i}{S_g}
$$

Cost per tick:

$$
C_i = \kappa c_i
$$

Fitness proxy:

$$
W_i = E\frac{c_i}{S_g}\Bigl[1 - (1 - p_0)^{S_g}\Bigr] - \kappa c_i
$$

Define:

$$
f(S) = 1 - (1 - p_0)^S
$$

Then \(W_i = E\frac{c_i}{S_g}f(S_g) - \kappa c_i\).

Selection gradient:

$$
\frac{\partial W_i}{\partial c_i}
=
E\left[
\frac{\partial}{\partial c_i}\left(\frac{c_i}{S_g}\right)f(S_g)
+
\frac{c_i}{S_g}f'(S_g)\frac{\partial S_g}{\partial c_i}
\right]
-\kappa
$$

Use:

$$
\frac{\partial S_g}{\partial c_i}=1,\qquad
\frac{\partial}{\partial c_i}\left(\frac{c_i}{S_g}\right)=\frac{S_g-c_i}{S_g^2}=\frac{S_{-i}}{S_g^2}
$$

and:

$$
f'(S_g) = (1-p_0)^{S_g}\ln\left(\frac{1}{1-p_0}\right)
$$

Final result:

$$
\frac{\partial W_i}{\partial c_i}
=
E\left[
\frac{S_{-i}}{S_g^2}\Bigl(1-(1-p_0)^{S_g}\Bigr)
+
\frac{c_i}{S_g}(1-p_0)^{S_g}\ln\left(\frac{1}{1-p_0}\right)
\right]
-\kappa
$$

Cooperation is selected for when:

$$
\frac{\partial W_i}{\partial c_i} > 0
$$
