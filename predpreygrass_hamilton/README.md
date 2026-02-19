# PredPreyGrass Hamilton Simulation (Cue-Based Kin Recognition)

Main file:

- `predpreygrass_hamilton.py`

## Summary

This simulation uses:

- solo predator hunting (no cooperative hunting trait)
- sexual demography for predators and prey
- Hamilton-rule energy transfer as the only cooperation channel
- cue-based kin recognition (`r_hat`) for decisions
- pedigree relatedness (`r_true`) only for hidden evaluation/calibration

Decision rule remains:

`r * B > C`

but with `r = r_hat` (estimated from cues), not oracle genealogy.

## Locked Recognition Design

- Cue model: `maternal_residence`
- Conservative bias against non-kin helping
- Pedigree `r_true` is never used to choose actions
- Pedigree `r_true` is used only for diagnostics

## Demographic Constraints

### Predators

- Life expectancy: `75` ticks
- Female fertility: `16..40`
- Male maturity: `16`
- Dependency: `<=16`

### Prey

- Life expectancy: `5` ticks
- Female fertility: `2..5`
- Male maturity: `2`
- Dependency: `<=1`

## Cue-Based Relatedness (`r_hat`)

For donor `d` and recipient `r`:

`r_hat = clip(KIN_CONSERVATIVE_BIAS + KIN_W_MATERNAL * maternal_signal + KIN_W_CORESIDENCE * coresidence_signal, 0, 1)`

### Maternal cue

- True condition: shared known mother ID
- Observed signal is noisy:
  - if true shared mother: Bernoulli(`MATERNAL_CUE_TPR`)
  - else: Bernoulli(`MATERNAL_CUE_FPR`)

### Co-residence cue

- Co-residence memory is maintained per species (pairwise familiarity)
- Per tick:
  - memory decays by `CORESIDENCE_DECAY`
  - nearby pairs are incremented by `CORESIDENCE_INCREMENT`
- Signal:

`coresidence_signal = min(1, score / CORESIDENCE_SATURATION)`

## Hamilton Transfer Mechanics

Transfer candidate `dE` is accepted when:

`r_hat * B(dE) > C(dE)`

Where:

- `B(dE) = S(E_recip + dE) - S(E_recip)`
- `C(dE) = F(E_donor) - F(E_donor - dE)`

`S` and `F` are logistic proxies (species-specific setpoint/scale parameters).

Transfers are:

- local (`*_KIN_TRANSFER_R`)
- chunked (`*_KIN_TRANSFER_CHUNK`)
- capped per donor per tick (`*_KIN_MAX_CHUNKS_PER_DONOR`)
- constrained by donor reserve (`*_KIN_RESERVE`)
- dependency-prioritized when enabled (`DEPENDENT_PRIORITY`)

## Pedigree (`r_true`) Usage

Pedigree and coefficient-of-relationship are still tracked for evaluation:

- child relation propagation uses two-parent approximation
- `r_true = coefficient_of_relationship(donor, recipient)`

`r_true` is logged, compared to `r_hat`, and used for confusion metrics only.

## Co-Residence Memory Lifecycle

Per species memory state:

- initialize with `init_cue_memory(...)`
- at each step:
  1. `decay_cue_memory(...)`
  2. `update_co_residence_memory(...)`
  3. kin transfers use the updated memory
  4. `prune_cue_memory(...)` after final cleanup

## Founder Relatedness Defaults

Defaults are now unrelated founders:

- `PRED_FOUNDER_CLAN_SIZE = 1`
- `PREY_FOUNDER_CLAN_SIZE = 1`

If you want seeded kin structure at t=0, set clan sizes >1 and use `FOUNDER_CLAN_R`.

## Tick Order

1. Grass regrowth
2. Prey phase: age/move/metabolism/feed
3. Predator phase: age/move/metabolism
4. Solo predation
5. Cleanup (starvation/age) + prune pedigree
6. Sexual reproduction (prey, then predators)
7. Cue-memory decay/update (pred + prey)
8. Kin transfer phase (pred + prey) using `r_hat`
9. Final cleanup + prune pedigree + prune cue memory

## Diagnostics

### Existing keys kept (semantic update)

- `pred_mean_r`, `prey_mean_r` now mean **estimated** `r_hat`
- Existing transfer/benefit/cost/margin keys remain

### New calibration keys

Per species:

- `*_true_r_mean`
- `*_r_abs_err_mean`
- `*_false_pos_rate`
- `*_false_neg_rate`
- `*_help_nonkin_energy`
- `*_help_kin_energy`

Confusion logic:

- Estimated positive if `r_hat >= EST_KIN_THRESHOLD`
- True positive if `r_true >= TRUE_KIN_THRESHOLD`

## Key Config Block (Cue Model)

- `KIN_CUE_MODEL = "maternal_residence"`
- `KIN_CONSERVATIVE_BIAS = -0.15`
- `KIN_W_MATERNAL = 0.75`
- `KIN_W_CORESIDENCE = 0.35`
- `MATERNAL_CUE_TPR = 0.85`
- `MATERNAL_CUE_FPR = 0.005`
- `CORESIDENCE_RADIUS_PRED = 2`
- `CORESIDENCE_RADIUS_PREY = 1`
- `CORESIDENCE_DECAY = 0.95`
- `CORESIDENCE_INCREMENT = 1.0`
- `CORESIDENCE_SATURATION = 8.0`
- `TRUE_KIN_THRESHOLD = 0.25`
- `EST_KIN_THRESHOLD = 0.25`

## Run

From repo root:

```bash
./.conda/bin/python predpreygrass_hamilton/predpreygrass_hamilton.py
```

## Short Programmatic Check

```python
import predpreygrass_hamilton.predpreygrass_hamilton as sim

sim.STEPS = 120
sim.PLOT_RESULTS = False
out = sim.run_sim(seed_override=123)
```
