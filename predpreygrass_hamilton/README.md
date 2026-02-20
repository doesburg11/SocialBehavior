# PredPreyGrass Hamilton Simulation (Minimal Inclusive Fitness)

Main file:

- `predpreygrass_hamilton.py`

## Model Goal

Keep cooperation minimal and explicit:

- cooperation is only energy transfer (`donor -> recipient`)
- no cooperative hunting or other cooperation channels
- transfer is selected by expected own-gene propagation (inclusive fitness)

## Core Decision Rule

For each candidate transfer `dE`, donor computes:

`score = g_kin * (r_hat * B) + g_spouse * (0.5 * delta_child_survival * n_shared_dependents * B) - C`

Transfer is accepted iff:

- `score > 0`
- donor remains above reserve energy

Where:

- `g_kin in [0,1]` is the donor's kin-transfer gene
- `g_spouse in [0,1]` is the donor's spouse-care gene
- `r_hat` is cue-based estimated relatedness
- `B` is recipient survival/fitness gain from `dE`
- `C` is donor fitness cost from `dE`
- `0.5` is parent-child relatedness
- `delta_child_survival` captures how much child survival improves if co-parent support succeeds

This keeps Hamilton logic in place (`r_hat * B - C`) but allows an explicit
indirect pathway for helping an unrelated co-parent when it protects shared
offspring.

## Evolving Genes

Each agent carries:

- `g_kin` (kin-transfer tendency)
- `g_spouse` (co-parent support tendency)

Both are in `[0,1]`.

Inheritance:

- child gets parental average plus Gaussian mutation:
  - `g_child = clip((g_mother + g_father)/2 + N(0, GENE_MUT_SD), 0, 1)`

Founder initialization:

- sampled around `GENE_INIT_MEAN` with `GENE_INIT_SD`, clipped to `[0,1]`

## Kin Recognition (`r_hat`) vs Pedigree (`r_true`)

Decision-side relatedness uses cues only:

`r_hat = clip(KIN_CONSERVATIVE_BIAS + KIN_W_PARENT_OFFSPRING * parent_offspring_signal + KIN_W_CORESIDENCE * coresidence_signal, 0, 1)`

Cue channels:

- parent-offspring cue with noisy detection (`PARENT_OFFSPRING_CUE_TPR/FPR`)
- co-residence memory (decay + local increment)

Pedigree `r_true` is retained only for diagnostics/calibration:

- confusion metrics (FP/FN/etc.)
- `r_hat` error tracking
- kin vs non-kin transfer leakage

No transfer decision uses `r_true` directly.

## Predator Juvenile Constraints (Human-like)

- Dependency limit: `PRED_DEPENDENT_AGE_MAX = 16`
- Juvenile hunting ramps with age to full by 16:
  - `hunt_capacity(age) = clip(PRED_JUV_HUNT_MIN_FACTOR + (1 - PRED_JUV_HUNT_MIN_FACTOR) * age / 16, 0, 1)`
- No extra probabilistic maturation gate is applied; survival is driven by normal energy/age dynamics.

## Spouse-Care Pathway

Spouse-care term is active for predators via:

- `PRED_CO_PARENT_CHILD_SURVIVAL_DELTA = 0.0` (disabled by default)

Prey currently has:

- `PREY_CO_PARENT_CHILD_SURVIVAL_DELTA = 0.0` (disabled for simplicity)

## Tick Order

1. Grass regrowth
2. Prey phase
3. Predator phase
4. Solo predation
5. Cleanup + pedigree pruning
6. Sexual reproduction
7. Cue-memory decay/update
8. Transfer phase (inclusive score rule)
9. Final cleanup + pruning

## Diagnostics Exported in `kin_hist`

Includes existing transfer diagnostics plus:

- gene trajectories:
  - `pred_mean_g_kin`, `pred_mean_g_spouse`
  - `prey_mean_g_kin`, `prey_mean_g_spouse`
- decision decomposition:
  - `*_mean_lhs` (raw `r_hat * B`)
  - `*_mean_gene_lhs` (`g_kin * r_hat * B`)
  - `*_mean_spouse_term`
  - `*_mean_margin` (full inclusive score)
- spouse-help behavior:
  - `*_spouse_help_rate`
  - `*_spouse_help_energy`

Plots now include:

- original kin diagnostics
- gene/spouse diagnostics (`plot_gene_diagnostics`)

## Tuned Defaults For 1000-Step Runs

The current defaults were tuned for long-run persistence (1000 steps) with
both species present in tested seeds. Original baseline values are shown in
brackets.

Key tuned predator values:

- `PRED_INIT = 100` (`original: 500`) # initial predator population
- `PRED_METAB = 0.008` (`original: 0.04`) # baseline energy burn per tick
- `PRED_MOVE_COST = 0.001` (`original: 0.010`) # extra energy cost when moving
- `PRED_HUNT_R = 1` (`original: 0`) # hunt search radius in grid cells
- `PRED_SOLO_HUNT_SCALE = 0.18` (`original: 0.05`) # hunt success steepness vs predator energy
- `PRED_HUNT_YIELD = 1.0` (`original: 0.85`) # predator energy gained from killed prey
- `PRED_REPRO_PROB = 0.80` (`original: 0.20`) # female birth attempt probability
- `PRED_CHILD_SHARE = 0.72` (`original: 0.25`) # fraction of maternal energy given to newborn
- `PRED_BIRTH_THRESH = 0.75` (`original: 1.5`) # minimum female energy required for reproduction
- `PRED_FEMALE_FERTILE_MAX = 70` (`original: 40`) # upper fertility age for females
- `PRED_MALE_MATURE_AGE = 12` (`original: 16`) # age at which males can mate
- `PRED_JUV_HUNT_MIN_FACTOR = 0.35` (`original: 0.0`) # hunting capacity at age 0 (before ramp)
- `PRED_KIN_TRANSFER_CHUNK = 0.02` (`original: 0.04`) # energy amount per transfer action
- `PRED_KIN_MAX_CHUNKS_PER_DONOR = 2` (`original: 4`) # max transfer actions per donor per tick
- `PRED_KIN_RESERVE = 1.2` (`original: 0.35`) # minimum donor energy kept after transfers

Key tuned prey support values:

- `PREY_INIT = 2200` (`original: 1000`) # initial prey population
- `PREY_REPRO_PROB = 0.998` (`original: 0.95`) # prey female birth attempt probability
- `PREY_MAX = 12000` (`original: 5000`) # prey carrying cap used by crowd scaling

## Live Grid

You can enable a real-time spatial grid view of the world state:

- `LIVE_GRID = True`
- `LIVE_GRID_EVERY = 1` (update every N ticks)
- `LIVE_GRID_PAUSE = 0.001` (UI pause per refresh)
- `LIVE_GRID_SHOW_CELL_LINES = True`

Color coding:

- green = grass biomass
- blue = prey density
- red = predator density

The live grid is separate from the end-of-run summary plots.

## Sharing Events Saved To Disk

Accepted energy-transfer events are written to CSV for audit/verification.

- path: `predpreygrass_hamilton/output/sharing_events.csv`
- enabled by `SAVE_SHARING_EVENTS = True`
- overwrite/append behavior via `SHARING_EVENTS_OVERWRITE`

Each row stores:

- sender/receiver IDs and species (`donor_id`, `recipient_id`, `donor_species`, `recipient_species`)
- relation fields (`relation_hint`, `r_hat`, `r_true`)
- ages (`donor_age`, `recipient_age`)
- energies before/after transfer (`donor_energy_before/after`, `recipient_energy_before/after`)
- transfer amount and decision terms (`transfer_energy`, `benefit_B`, `cost_C`, `inclusive_score`)
- tick index (`step`) and channel (`pred_kin` or `prey_kin`)

## Run

From repo root:

```bash
./.conda/bin/python predpreygrass_hamilton/predpreygrass_hamilton.py
```

Programmatic smoke check:

```python
import predpreygrass_hamilton.predpreygrass_hamilton as sim

sim.STEPS = 120
sim.PLOT_RESULTS = False
out = sim.run_sim(seed_override=123)
```
