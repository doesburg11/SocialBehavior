# Predator-Prey-Grass with Mixed Heritable Strategies

Main file:

- `predprey_grass_altruism.py`

Alternative variant in the same folder:

- `predpreygrass_selfish_v_altruistic_predators.py`
  - stricter fairness-controlled comparison version (no strategy-specific exogenous energy, no built-in mortality/reproduction multipliers, equal baseline hunt radius/success)

## Goal

Create one shared predator-prey-grass world (not separate scenarios) where:

1. altruistic and selfish predators coexist in the same run,
2. strategy is heritable (with mutation),
3. selfish predators are driven to near-zero,
4. altruists and prey continue to coexist.

## Independence

This module is standalone:

- no imports from other simulation modules in this repository,
- own world dynamics, strategy inheritance, visualization, and analysis pipeline,
- all parameters defined in-file (no CLI parameter interface).

## Model Structure

- Environment:
  - toroidal 2D grid (`W x H`) with regrowing grass.
- Prey:
  - move, graze grass, metabolize energy, age, die, reproduce.
- Predators (same world, mixed strategies):
  - move, metabolize, hunt prey, age, die, reproduce.
  - each predator carries:
    - `altruistic` (bool strategy gene),
    - `kin_id` (heritable kin cue).

## Heritable Strategy Mechanism

Predator offspring inherit parent strategy with mutation:

- `child_strategy = parent_strategy` unless mutation occurs.
- mutation probability is `STRATEGY_MUTATION_PROB`.

Predator offspring inherit `kin_id` with optional kin mutation:

- `child_kin = parent_kin` unless mutation occurs.
- mutation probability is `KIN_ID_MUTATION_PROB`.

## Kin-Directed Altruism Mechanism

Altruists cooperate only with kin-recognized allies.

### Hunting success (altruist)

`p_success = clip(ALTRUIST_HUNT_BASE_SUCCESS + ALTRUIST_HUNT_SUPPORT_BONUS * n_kin_allies, 0, HUNT_MAX_SUCCESS)`

Variables:

- `n_kin_allies`: number of nearby alive altruists recognized as kin,
- `ALTRUIST_HUNT_BASE_SUCCESS`: baseline altruist hunting success,
- `ALTRUIST_HUNT_SUPPORT_BONUS`: incremental support bonus per kin ally,
- `HUNT_MAX_SUCCESS`: hard cap for hunting success probability.

### Hunting success (selfish)

`p_success = SELFISH_HUNT_SUCCESS`

Selfish predators also pay local conflict cost from nearby selfish neighbors.

### Energy transfer

Altruists with surplus energy donate to needy altruist kin in local radius:

- donor threshold: `TRANSFER_RESERVE`,
- needy threshold: `TRANSFER_NEEDY_THRESHOLD`,
- transfer size per event: `TRANSFER_CHUNK` (up to `TRANSFER_TARGET`).

## Live Pygame Grid

Enable in `predprey_grass_altruism.py`:

- `LIVE_GRID = True`

Live view now uses a right-side panel (next to the grid) for status, legend, and charts.

The right-side panel shows four rolling charts:

1. altruistic predator count,
2. selfish predator count,
3. prey count,
4. altruist frequency.

Distinction in live grid:

- altruistic predators: blue dots,
- selfish predators: red dots,
- prey: yellow dots.
- a boxed in-world legend overlay is shown by default (top-left of the grid).
- grass is rendered in grayscale (darker = low biomass, lighter = high biomass).

Playback is slowed by default:

- `LIVE_GRID_FPS = 4`
- terminal state hold: `LIVE_GRID_TERMINAL_HOLD_SEC = 3.0`

## Outputs

Written to `predprey_grass_altruism/output/`:

- `summary_metrics.csv`
- `mixed_altruist_selfish_dynamics.png`

## Validated Default Snapshot (2026-02-21)

Default deterministic demo parameters:

- `SEED = 122913`
- `REPLICATES = 1`
- `STEPS = 520`

Observed:

- final altruists: `128`
- final selfish: `1`
- final prey: `2039`
- selfish near-zero tail fraction: `1.000`
- altruist-prey coexistence tail fraction: `1.000`
- tail altruist frequency: `99.3%`
- selfish first reaches near-zero threshold (`<= 3`) at tick `32`
- selfish extinction occurs at tick `39`

This satisfies the requested direction in one mixed-population run:

1. selfish strategy collapses to near-zero,
2. altruists persist,
3. prey persist.

## Run

From repo root:

```bash
./.conda/bin/python predprey_grass_altruism/predprey_grass_altruism.py
```

## Programmatic Use

```python
import predprey_grass_altruism.predprey_grass_altruism as sim

sim.LIVE_GRID = False
sim.REPLICATES = 6
summary = sim.run_experiment()
```

## Stepwise Implementation Update (2026-02-21)

1. Replaced scenario-comparison architecture with a single mixed world containing altruistic and selfish predators simultaneously.
2. Added heritable strategy inheritance in predator reproduction.
3. Added optional strategy mutation (`STRATEGY_MUTATION_PROB`) to allow frequency evolution rather than fixed strategy composition.
4. Added heritable kin cue (`kin_id`) and optional kin mutation (`KIN_ID_MUTATION_PROB`).
5. Refactored cooperative hunting so altruistic support depends on recognized kin allies in local radius.
6. Refactored transfer phase so energy sharing occurs only from altruistic donors to needy altruistic kin.
7. Added mixed-population time series collection: altruist count, selfish count, prey, grass, transfer count, altruist frequency.
8. Added mixed-population summary metrics including selfish near-zero tail fraction and altruist-prey coexistence tail fraction.
9. Reworked live renderer to show true in-world mixed strategies (no paired-reference scenario workaround).
10. Added fourth live chart for altruist frequency in addition to altruist/selfish/prey charts.
11. Tuned strategy dynamics so selfish is strongly selected against in the default deterministic run while altruists and prey persist.
12. Slowed live playback (`LIVE_GRID_FPS = 4`) to make selfish decline and transition easier to observe.
13. Updated output plot to mixed-strategy dynamics (`mixed_altruist_selfish_dynamics.png`).
14. Updated this README with mechanism details, variable definitions, validated snapshot, and reproducible defaults.
15. Added a larger in-world legend overlay and reduced grass color intensity to improve live-grid readability.
16. Switched the live grass background to grayscale so only agents use strong colors, improving visual distinction.
17. Moved live population charts from below the grid to a right-side panel and grouped status + legend there for readability.

## Notes

- The default setup is a deterministic demonstration configuration.
- For robustness testing across stochastic runs, increase `REPLICATES` and set `LIVE_GRID = False`.
- Under broad seed sweeps, ecological stochasticity can still produce occasional full predator collapse depending on parameter sensitivity.

## Strict Variant Update (2026-02-22)

For `predprey_grass_altruism/predpreygrass_selfish_v_altruistic_predators.py`:

1. Removed strategy-specific exogenous energy injection by setting altruist fallback foraging gain to `0.0` (selfish was already `0.0`).
2. Equalized strategy mortality multipliers (`ALTRUIST_MORTALITY_MULT = SELFISH_MORTALITY_MULT = 1.0`).
3. Equalized strategy reproduction multipliers (`ALTRUIST_REPRO_MULT = SELFISH_REPRO_MULT = 1.0`).
4. Equalized baseline hunting reach (`ALTRUIST_HUNT_RADIUS = SELFISH_HUNT_RADIUS = 2`).
5. Equalized baseline hunt success (`ALTRUIST_HUNT_BASE_SUCCESS = SELFISH_HUNT_SUCCESS`), while keeping altruist support bonus as the behavioral cooperation mechanism.
6. Updated the file docstring to state that this variant is a stricter comparison configuration.
7. Changed hunting gain to use prey energy at capture (`captured_prey_energy * HUNT_ASSIMILATION_EFFICIENCY`) instead of a fixed predator reward, improving energy accounting in the strict variant.
8. Removed selfish-specific hunting conflict penalty (`SELFISH_CONFLICT_COST = 0.0`) to make the strict variant more symmetric.
9. Renamed `ALTRUIST_ASSIST_SHARE` to `ALTRUIST_ASSIST_SHARE_MAX_ENERGY` to clarify that it is an absolute energy amount (not a percentage).
10. Tuned only shared (strategy-neutral) ecological parameters in the strict variant so predator populations can propagate under the stricter energy accounting.
11. Shared propagation-tune changes include higher prey productivity (`PREY_INIT`, `PREY_BITE`, `PREY_REPRO_PROB`, `GRASS_REGROWTH`) and lower shared predator maintenance/attempt costs (`PRED_MOVE_PROB`, `PRED_MOVE_COST`, `PRED_METAB`, `HUNT_ATTEMPT_COST`).
12. Shared propagation-tune changes also lower predator reproduction threshold and mortality while raising predator reproduction rate / lifespan (`PRED_REPRO_THRESH`, `PRED_REPRO_PROB`, `PRED_BASE_MORTALITY`, `PRED_MAX_AGE`), plus a higher equal baseline hunt success for both strategies (`ALTRUIST_HUNT_BASE_SUCCESS = SELFISH_HUNT_SUCCESS`).
13. Added a separate strict-variant summary metric for propagation (`predator_prey_coexistence_tail_mean` = any predator + prey coexistence in the tail) so propagation success is reported independently from selfish-suppression strength.
14. Updated the strict-variant outcome plot panel to include a fourth bar for `predator_prey_coexistence_tail_mean` and a dashed `0.75` threshold line, making suppression-vs-propagation visually distinct.
15. Added numeric value labels above the strict-variant outcome bars so the plotted metrics can be read directly without estimating from the y-axis.
16. Renamed `KIN_MUTATION_PROB` to `KIN_ID_MUTATION_PROB` for clarity (it mutates the kin label, not strategy).
17. Set `ALTRUIST_HUNT_SUPPORT_BONUS = 0.0` in the strict variant to remove the altruist-only kill-probability bonus, isolating other behavioral differences (e.g., sharing/kin targeting).
18. Fixed hunt-time assist sharing overflow at `PRED_ENERGY_CAP` so the altruist hunter now subtracts only the energy actually delivered to the helper (no hidden energy loss from capped helper gain in that step).
19. Updated the strict-variant `OUTPUT_DIR` to `predpreygrass_altruism/output` after the folder rename, so CSV/plot outputs are written to the canonical renamed directory instead of the legacy `predprey_grass_altruism/output` path.
20. Retuned the strict variant again (shared parameters only, with `ALTRUIST_HUNT_SUPPORT_BONUS = 0.0` preserved) to restore predator propagation: increased equal baseline hunt success for both strategies and mildly increased prey productivity while lowering shared predator maintenance/attempt costs and easing shared predator lifespan/reproduction thresholds.
21. Added a strict-variant `HELP_TARGETING_MODE` toggle with two instant-switch modes: `kin_targeted` (default) and `random_any_predator`, applied consistently to both hunt-time sharing and `transfer_phase` recipient targeting so recipient discrimination can be neutralized without changing transfer amounts/caps.
22. Added `selfish_absorbing_extinction_step_mean` (first selfish-zero tick that stays zero through the rest of the run horizon) to the strict-variant reporting/CSV, alongside the existing `selfish_extinction_step_mean` (first zero, even if selfish later reappears), so temporary-vs-permanent extinction timing can be distinguished.
23. Tuned the strict-variant transfer parameters for a mode-dependent outcome test (`HELP_TARGETING_MODE` switch): `TRANSFER_RESERVE = 1.0`, `TRANSFER_TARGET = 2.4`, `TRANSFER_NEEDY_THRESHOLD = 1.4`, `MAX_TRANSFERS_PER_DONOR = 16`. In the tested seed set, this preserves the strong `kin_targeted -> selfish extinction` pattern while making `random_any_predator` frequently produce altruist extinction (demonstrating the importance of recipient discrimination).
24. Added a built-in paired `HELP_TARGETING_MODE` comparison runner (`run_help_targeting_mode_comparison`) with config toggles to run `kin_targeted` and `random_any_predator` back-to-back on the same replicate seeds, print compact per-mode summaries, and print paired per-seed final outcomes without requiring manual mode switching.
25. Performed a focused local refinement around the transfer split point (near `TRANSFER_RESERVE=1.0`, `TRANSFER_TARGET=2.4`, `TRANSFER_NEEDY_THRESHOLD=1.4`, `MAX_TRANSFERS_PER_DONOR=16`) and kept the current defaults because nearby variants (`reserve=0.8` and/or `max_transfers=20`) weakened the desired `random_any_predator -> altruist extinction` behavior in the tested seed subset.
26. Added `random_altruist_only` to `HELP_TARGETING_MODE` (random nearby altruist recipients without kin filtering) so kin-targeting can be compared against an altruist-only non-kin targeting control in addition to `random_any_predator`.
27. Added `COMPARISON_SUMMARY_CSV` export for the built-in mode comparison runner, writing one row per mode plus a paired-difference row (`help_targeting_mode_comparison_summary.csv`) with paired outcome-count statistics.
28. Added `predpreygrass_altruism/COMPARISON.md` with a paired 4-seed comparison of `kin_targeted` vs `random_any_predator`, including matched-seed final outcomes and summary-metric deltas.
29. Added explicit `ENABLE_HUNT_ASSIST_SHARING` switch (default `False` in the current experiment setup) so hunt-time helper sharing can be disabled cleanly without overloading `ALTRUIST_ASSIST_SHARE_MAX_ENERGY = 0.0`; this isolates `transfer_phase` targeting effects when comparing `HELP_TARGETING_MODE`.
30. Regenerated `predpreygrass_altruism/COMPARISON.md` as a three-mode report (`kin_targeted`, `random_altruist_only`, `random_any_predator`) under `ENABLE_HUNT_ASSIST_SHARING = False`, so the comparison now includes the altruist-only non-kin targeting control while helper sharing is disabled.
31. Retuned the transfer-only split test (`ENABLE_HUNT_ASSIST_SHARING = False`) by increasing `TRANSFER_TARGET` to `3.2` and `TRANSFER_NEEDY_THRESHOLD` to `2.2` (keeping `TRANSFER_RESERVE = 1.0`, `MAX_TRANSFERS_PER_DONOR = 16`) to strengthen the mode-dependent outcome: in the tested 4 paired seeds, `kin_targeted` still produced selfish extinction in all 4 while `random_any_predator` produced altruist extinction in all 4.
32. Regenerated `predpreygrass_altruism/COMPARISON.md` again as a focused two-mode transfer-only tuned-split report (`kin_targeted` vs `random_any_predator`) under `ENABLE_HUNT_ASSIST_SHARING = False`, matching the retuned transfer parameters and documenting the 4/4 paired-seed split outcome.
33. Extended the built-in comparison runner so it auto-generates `predpreygrass_altruism/COMPARISON.md` on every `run_help_targeting_mode_comparison()` execution (in addition to the comparison CSV), using the actual paired run results and writing summary metrics, per-seed final outcomes, and pairwise outcome-count sections directly from the runner.
34. Updated the default comparison-mode tuple to three modes (`kin_targeted`, `random_altruist_only`, `random_any_predator`) so the built-in comparison runner now tests both recipient-discrimination controls (kin-targeted and altruist-only random) against fully non-discriminatory random recipient targeting in one run.
35. Ran a 10-seed paired transfer-only robustness comparison (`ENABLE_HUNT_ASSIST_SHARING = False`, `STEPS = 500`) with the tuned split parameters (`TRANSFER_RESERVE = 1.0`, `TRANSFER_TARGET = 3.2`, `TRANSFER_NEEDY_THRESHOLD = 2.2`, `MAX_TRANSFERS_PER_DONOR = 16`) and regenerated the auto-written `COMPARISON.md`: both `kin_targeted` and `random_altruist_only` produced selfish extinction in 10/10 seeds, while `random_any_predator` produced selfish extinction in only 1/10 seeds and altruist extinction in 8/10 seeds, preserving the mode-dependent selection-direction split at the 10-seed level.
36. Added a separate simplified script copy `predpreygrass_altruism/predpreygrass_transfer_only_altruism_vs_selfish.py` for transfer-only mechanism testing (selfish vs altruistic giving only): removed kin-recognition targeting, removed post-hunt helper sharing, removed the built-in comparison-runner/reporting block, kept the same ecological dynamics/live grid/summary outputs, and replaced the recipient-targeting choice with a single two-value switch `TRANSFER_RECIPIENT_MODE = \"altruist_only\" | \"any_predator\"` so discriminatory vs non-discriminatory transfer can be compared with less code optionality.
37. Renamed transfer-threshold variables in the simplified script for clarity: `TRANSFER_TARGET -> TRANSFER_RECIPIENT_TARGET_ENERGY` and `TRANSFER_NEEDY_THRESHOLD -> TRANSFER_RECIPIENT_NEEDY_ENERGY_THRESHOLD`, so the recipient-role semantics are explicit at the config level.
38. Added a minimal built-in paired comparison helper to the simplified script (`run_two_mode_comparison`) controlled by a single switch `RUN_TWO_MODE_COMPARISON`, which runs `altruist_only` vs `any_predator` on the same seeds, disables live rendering/file outputs during the comparison, and prints compact paired per-seed final outcomes without reintroducing the larger comparison framework.
39. Validated/tuned the simplified transfer-only copy against the carried-over transfer settings (`TRANSFER_RESERVE = 1.0`, `TRANSFER_RECIPIENT_TARGET_ENERGY = 3.2`, `TRANSFER_RECIPIENT_NEEDY_ENERGY_THRESHOLD = 2.2`, `MAX_TRANSFERS_PER_DONOR = 16`) using a 10-seed paired comparison (`STEPS = 500`): `altruist_only` produced selfish extinction in 10/10 seeds, while `any_predator` produced selfish extinction in 0/10 seeds and left altruists extinct in 8/10 seeds (strong split retained, so defaults were kept).
40. Changed the default `HELP_TARGETING_MODE` in the full strict script to `random_altruist_only` so altruistic predators give to nearby altruists without kin filtering by default (instead of the prior `random_any_predator` default), while preserving the mode switch for later comparisons.
41. Reverted the accidental default-mode change in the full strict script (`HELP_TARGETING_MODE`) back to `random_any_predator` after clarifying that the request to remove kin targeting was intended for the separate simplified copy, not for the original comparison script.
42. Retuned the simplified transfer-only copy toward a stronger paired mode split by increasing transfer intensity/eligibility (`TRANSFER_CHUNK = 0.30`, `TRANSFER_RECIPIENT_TARGET_ENERGY = 3.4`, `TRANSFER_RECIPIENT_NEEDY_ENERGY_THRESHOLD = 2.4`, with `TRANSFER_RESERVE = 1.0`, `MAX_TRANSFERS_PER_DONOR = 16` unchanged). In a 10-seed paired comparison (`STEPS = 500`), this preserved `altruist_only -> selfish extinction` in 10/10 seeds and improved `any_predator -> altruist extinction` from 8/10 to 9/10 seeds (not yet 10/10).
43. Completed a constrained exact (early-exit) search using `PRED_REPRO_PROB` as the third shared-ecology lever (with `PRED_METAB` held at `0.050`) and updated the simplified transfer-only defaults to `PRED_REPRO_THRESH = 1.34`, `PRED_REPRO_PROB = 0.29`, and `PRED_BASE_MORTALITY = 0.0023`. Under the same 10-seed paired comparison setup (`STEPS = 500`), this reaches the exact target split: `TRANSFER_RECIPIENT_MODE = "altruist_only"` yields selfish extinction in 10/10 seeds, while `TRANSFER_RECIPIENT_MODE = "any_predator"` yields altruist extinction in 10/10 seeds.
44. Added built-in comparison artifact writing to the simplified script’s `run_two_mode_comparison()` (Markdown report + CSV summary) and a config-level “tuning lock” comment block documenting that the current shared predator defaults are tuned for the paired 10-seed exact-split objective. Regenerated a fresh simplified comparison report (`predpreygrass_altruism/COMPARISON_TRANSFER_ONLY.md`) and summary CSV (`predpreygrass_altruism/output_transfer_only_simple/two_mode_comparison_summary.csv`) at the exact-split defaults, and verified the same exact split on a second independent 10-seed sequence (`seed_base = 20260224`, stride `7919`): `altruist_only -> selfish extinct 10/10`, `any_predator -> altruists extinct 10/10`.
45. Added `COMPARISON_SEED_BASE` to the simplified script so `run_two_mode_comparison()` can be rerun against a different seed base without changing `SEED`, plus a one-line runtime tuning banner that prints the exact-split target and the active shared predator/transfer defaults. Comparison artifact writing is now seed-aware: default-seed runs keep writing to `predpreygrass_altruism/COMPARISON_TRANSFER_ONLY.md` and `predpreygrass_altruism/output_transfer_only_simple/two_mode_comparison_summary.csv`, while non-default `COMPARISON_SEED_BASE` runs also write seed-suffixed side-by-side archives (for example `predpreygrass_altruism/COMPARISON_TRANSFER_ONLY_seed_20260224.md` and `predpreygrass_altruism/output_transfer_only_simple/two_mode_comparison_summary_seed_20260224.csv`).
46. Set the simplified script default `RUN_TWO_MODE_COMPARISON = False` so running `predpreygrass_transfer_only_altruism_vs_selfish.py` now executes the normal single experiment path by default (live grid / standard summary outputs) instead of automatically launching the paired two-mode comparison helper. The comparison helper remains available by explicitly switching `RUN_TWO_MODE_COMPARISON = True`.
47. Added `USE_NEEDY_THRESHOLD` to the simplified transfer-only script as a clean ablation switch for recipient need-gating in `transfer_phase` (when `False`, the recipient energy threshold filter is bypassed while all other transfer rules remain unchanged). A 4-condition exact matrix on the tuned defaults (`TRANSFER_RECIPIENT_MODE ∈ {altruist_only, any_predator}` × `USE_NEEDY_THRESHOLD ∈ {True, False}`; 10 paired seeds, `STEPS = 500`) showed the same headline selection-direction split in all four cells (`10/10` each), indicating that recipient discrimination (`altruist_only` vs `any_predator`) is sufficient for the tuned split outcome in this regime, while the needy-threshold is not required for the exact directional result on the tested seeds.
48. Removed the recipient needy-threshold mechanism entirely from the simplified transfer-only script after the ablation showed it was not required for the exact directional split on the tested seeds. The simplified transfer phase now filters recipients only by local proximity and `TRANSFER_RECIPIENT_MODE` (plus donor reserve/transfer caps), and the tuning banner reports `needy_filter=removed` to make this explicit in runtime output.
49. Retuned the no-needy simplified transfer-only defaults after the removal caused a small robustness drop on the alternate 10-seed sequence (`seed_base = 20260224`): increasing the shared predator reproduction probability from `PRED_REPRO_PROB = 0.29` to `0.30` restored the exact paired split on both tested seed bases while keeping the mechanism unchanged (`altruist_only -> selfish extinct 10/10`, `any_predator -> altruists extinct 10/10` for `seed_base = 122913` and `20260224`, `STEPS = 500`).
50. Regenerated the simplified transfer-only two-mode comparison artifacts after the no-needy retune so the archived reports/CSVs now match the patched defaults (`PRED_REPRO_PROB = 0.30`), and added a third seed-suffixed robustness archive using `COMPARISON_SEED_BASE = 20270401`. All three archived 10-seed bases (`122913`, `20260224`, `20270401`) show the exact directional split at `STEPS = 500`: `TRANSFER_RECIPIENT_MODE = "altruist_only"` -> selfish extinct `10/10`; `TRANSFER_RECIPIENT_MODE = "any_predator"` -> altruists extinct `10/10`.
51. Added explicit archived-seed examples for the simplified two-mode comparison helper directly in `predpreygrass_transfer_only_altruism_vs_selfish.py` (comment near `COMPARISON_SEED_BASE` and one-line tuning banner suffix), so the three archived robustness runs (`122913`, `20260224`, `20270401`) are visible from the script itself when rerunning `run_two_mode_comparison()`.

Interpretation:

- This makes the comparison more appropriate for mechanism testing, but it is still a simulation model (not empirical evidence).
- Outcomes may differ strongly from the tuned demo and usually require retuning if you want stable coexistence plus slow selfish decline.
