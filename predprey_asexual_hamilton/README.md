# PredPrey Asexual Hamilton Simulation

Main file:

- `predprey_asexual_hamilton.py`

## Goal

Keep the Hamilton-transfer mechanism while removing high-complexity subsystems.

This variant is intentionally simplified to focus on:

- predator-prey-grass ecology
- inheritance and mutation of one cooperation gene (`g_kin`)
- kin-directed energy transfer using inclusive fitness

## Stepwise Simplification Update (2026-02-20)

1. Forked the model into a new directory: `predprey_asexual_hamilton/`.
2. Removed sexual-demography requirements from reproduction (no mate search, no sex roles).
3. Replaced two-parent inheritance with clonal inheritance + mutation:
   - `g_child = clip(g_parent + N(0, GENE_MUT_SD), 0, 1)`
4. Removed spouse-care pathway entirely:
   - deleted `g_spouse` from agents
   - removed spouse-term contribution from transfer score
   - removed spouse diagnostics/plots
5. Removed cue-memory recognition subsystem:
   - no co-residence memory update/decay/prune
   - no cue-weighted `r_hat` model
6. Replaced recognition model with deterministic pedigree estimate:
   - `r_hat = coefficient_of_relationship(donor, recipient)`
7. Kept Hamilton transfer core and ecological pipeline intact (movement, metabolism, solo predation, grass regrowth).
8. Updated output paths and diagnostics to the new directory.

## Core Decision Rule

For each transfer candidate `dE`, donor computes:

`score = g_kin * (r_hat * B) - C`

Transfer is accepted iff:

- `score > 0`
- donor remains above reserve energy

Where:

- `g_kin in [0,1]` is the donor's kin-transfer gene
- `r_hat` is estimated relatedness used for decision
- `B` is recipient survival/fitness gain from `dE`
- `C` is donor fitness cost from `dE`

In this simplified model, `r_hat` is set directly from pedigree relatedness.

## How Multi-Chunk Transfers Work

Transfers are chunked, not one-shot.

For each donor in the transfer phase:

1. Initialize a per-tick counter `chunks_done = 0`.
2. While `chunks_done < max_chunks_per_donor`:
3. Compute `available = donor.energy - reserve_energy`.
4. Set `transfer = min(transfer_chunk, available)`.
5. Re-scan nearby candidates within transfer radius.
6. Evaluate inclusive score for each candidate using current energies:
   - `score = g_kin * (r_hat * B) - C`
7. Pick the best candidate by score.
8. Accept transfer only if score is positive and reserve is still respected after transfer.
9. If accepted, move energy, increment `chunks_done`, and repeat from step 3.
10. Stop early if any gate fails (no energy above reserve, no candidates, non-positive score, reserve violation).

Implications:

- A single donor may help multiple recipients in one tick.
- A single donor may also help the same recipient repeatedly if that recipient remains best.
- Decision terms are recomputed after each accepted chunk, so behavior adapts to updated energies.

## Asexual Reproduction

Both species reproduce clonally:

- parent must be within species-specific reproductive age window
- parent must exceed species-specific energy threshold
- reproduction succeeds with species-specific probability and crowd scaling
- newborn receives a share of parent energy
- newborn `g_kin` is parent `g_kin` plus Gaussian mutation
- pedigree is updated with clonal relatedness propagation

## Kin Relatedness

- A full relatedness matrix is maintained and pruned as agents die.
- Founder clans can still seed non-zero initial relatedness.
- Transfer diagnostics retain `r_hat` and `r_true`; with this simplification they match.

## Tick Order

1. Grass regrowth
2. Prey phase
3. Predator phase
4. Solo predation
5. Cleanup + pedigree pruning
6. Asexual reproduction
7. Transfer phase (Hamilton score rule)
8. Final cleanup + pedigree pruning

## Diagnostics (`kin_hist`)

Includes:

- transfer rates and total transferred energy
- `r_hat`, `r_true`, absolute error, false positive/false negative rates
- decision terms (`B`, `C`, `lhs`, `gene_lhs`, margin)
- kin vs non-kin helped energy
- evolving `g_kin` means for predators and prey
- ecological counters (`solo_kills`, `prey_to_pred_energy`)

## Sharing Events CSV

Accepted transfers are written to:

- `predprey_asexual_hamilton/output/sharing_events.csv`

Each row includes donor/recipient IDs/species, relation hints, energies before/after,
transfer size, Hamilton terms, and step index.

## Run

From repo root:

```bash
./.conda/bin/python predprey_asexual_hamilton/predprey_asexual_hamilton.py
```

Programmatic smoke check:

```python
import predprey_asexual_hamilton.predprey_asexual_hamilton as sim

sim.STEPS = 120
sim.PLOT_RESULTS = False
sim.LIVE_GRID = False
sim.SAVE_SHARING_EVENTS = False
out = sim.run_sim(seed_override=123)
```
