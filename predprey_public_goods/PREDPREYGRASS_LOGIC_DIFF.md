# PredPreyGrass Stag-Hunt Logic Diff

This document compares:

- `predprey_public_goods/emerging_cooperation.py`
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view/predpreygrass_rllib_env.py`
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view/config/config_env_stag_hunt_forward_view.py`

## Scope

This is the direct comparison against `rllib/stag_hunt_forward_view` (not `ecology/minimal_engine.py`).

## One-Screen Summary

- `emerging_cooperation.py` is now a grass-coupled ecology with explicit prey energy household and heritable cooperation trait (`coop`).
- `stag_hunt_forward_view` is an RLlib multi-agent environment with typed agents, action-based `join_hunt`, and per-agent RL outputs.
- Intended conceptual difference:
  - `emerging_cooperation.py`: cooperation is nature-like (inherited trait).
  - `stag_hunt_forward_view`: cooperation is nurture-like (step-wise action choice).

## Side-by-Side Mechanics

| Area | `stag_hunt_forward_view` | `emerging_cooperation.py` |
|---|---|---|
| Agent model | Typed IDs (`type_1_predator`, `type_1_prey`, `type_2_prey`) with per-agent state (`predpreygrass_rllib_env.py:25`, `predpreygrass_rllib_env.py:237`) | Two dataclasses: `Predator(x,y,energy,coop)`, `Prey(x,y,energy)` (`emerging_cooperation.py:114`, `emerging_cooperation.py:122`) |
| Core loop order | Decay/age -> grass regen -> movement -> prey engagements -> removals -> reproduction -> RLlib outputs (`predpreygrass_rllib_env.py:401`, `predpreygrass_rllib_env.py:459`, `predpreygrass_rllib_env.py:496`) | Grass regrowth -> prey energy household/repro -> hunt -> predator energy/repro/death (`emerging_cooperation.py:156`, `emerging_cooperation.py:159`, `emerging_cooperation.py:207`, `emerging_cooperation.py:286`) |
| Movement | Action-driven; bounded grid with occupancy/walls/optional LOS checks (`predpreygrass_rllib_env.py:748`, `predpreygrass_rllib_env.py:809`, `predpreygrass_rllib_env.py:822`) | Random local movement with toroidal wrap (`emerging_cooperation.py:168`, `emerging_cooperation.py:298`) |
| Resource layer | Grass regrows and is consumed by prey (`predpreygrass_rllib_env.py:708`, `predpreygrass_rllib_env.py:1249`) | Grass regrows and is consumed by prey (`emerging_cooperation.py:157`, `emerging_cooperation.py:179`) |
| Capture trigger | Evaluated per prey engagement (`predpreygrass_rllib_env.py:419`, `predpreygrass_rllib_env.py:1245`) | Evaluated per predator-occupied cell (`emerging_cooperation.py:211`) |
| Capture neighborhood | Predators within Moore-1 around prey (`predpreygrass_rllib_env.py:1016`, `predpreygrass_rllib_env.py:1031`) | Victim candidates by `HUNT_R`; hunter pool by `HUNTER_POOL_R` (`emerging_cooperation.py:213`, `emerging_cooperation.py:243`) |
| Cooperation mechanism | Explicit `join_hunt` action (joiners vs free-riders) (`predpreygrass_rllib_env.py:718`, `predpreygrass_rllib_env.py:1038`) | Trait-based contribution via `coop`; no action-level join/defect (`emerging_cooperation.py:256`, `emerging_cooperation.py:265`) |
| Capture success rule | `sum(joiner_energy) > prey_energy + margin` (`predpreygrass_rllib_env.py:1044`, `predpreygrass_rllib_env.py:1049`) | `energy_threshold_gate`: hard power gate + probabilistic gate from summed `coop` (`emerging_cooperation.py:238`, `emerging_cooperation.py:260`, `emerging_cooperation.py:266`) |
| Failed hunt cost | Joiners pay `team_capture_join_cost`; immediate starvation possible (`predpreygrass_rllib_env.py:1052`, `predpreygrass_rllib_env.py:1082`) | No explicit failed-hunt penalty beyond baseline predator costs (`emerging_cooperation.py:271`, `emerging_cooperation.py:294`) |
| Success payoff split | Joiners share prey energy (equal/proportional) + optional scavenger share (`predpreygrass_rllib_env.py:1112`, `predpreygrass_rllib_env.py:1120`, `predpreygrass_rllib_env.py:1166`) | Fixed `KILL_ENERGY` split equally among hunters (`emerging_cooperation.py:276`, `emerging_cooperation.py:278`) |
| Prey feeding | Grass feeding if not captured (`predpreygrass_rllib_env.py:1249`, `predpreygrass_rllib_env.py:1269`) | Grass feeding before hunt each tick (`emerging_cooperation.py:179`, `emerging_cooperation.py:181`) |
| Predator reproduction | Thresholded, child gets configured fixed energy, parent pays that amount; capacity-limited IDs (`predpreygrass_rllib_env.py:1307`, `predpreygrass_rllib_env.py:1360`, `predpreygrass_rllib_env.py:1311`) | Thresholded + probabilistic + crowding/prey-availability scaling; parent energy halves (`emerging_cooperation.py:288`, `emerging_cooperation.py:301`, `emerging_cooperation.py:305`) |
| Prey reproduction | Thresholded by prey type; child gets type-specific configured energy (`predpreygrass_rllib_env.py:1399`, `predpreygrass_rllib_env.py:1452`) | Energy-thresholded and stochastic with crowding; child receives split of parent energy (`emerging_cooperation.py:163`, `emerging_cooperation.py:187`, `emerging_cooperation.py:188`) |
| Mutation/evolution | No trait mutation in env reproduction | Predator trait mutation present (`emerging_cooperation.py:311`) |
| Death model | Starvation (all agents), predation, time-limit truncation (`predpreygrass_rllib_env.py:414`, `predpreygrass_rllib_env.py:993`, `predpreygrass_rllib_env.py:606`) | Starvation for prey and predators, plus predation (`emerging_cooperation.py:176`, `emerging_cooperation.py:316`, `emerging_cooperation.py:283`) |
| Episode termination | Per-agent RL `terminations`/`truncations` + `__all__` (`predpreygrass_rllib_env.py:541`, `predpreygrass_rllib_env.py:606`) | Single-loop extinction break + optional restart in `main()` (`emerging_cooperation.py:431`, `emerging_cooperation.py:629`) |

## Capture Logic: Direct Contrast

### `stag_hunt_forward_view` (`predpreygrass_rllib_env.py:1027`)

1. For each prey, gather nearby predators.
2. Split into `joiners` and `free_riders` by `join_hunt` action.
3. If joiner energy is insufficient, attempt fails and joiners pay join cost.
4. If successful, prey energy is distributed to joiners (and optionally scavengers), then prey is terminated.

### `emerging_cooperation.py` (`emerging_cooperation.py:207`)

1. For each predator-occupied cell, gather nearby prey candidates.
2. Choose one victim and assemble a local hunter pool.
3. Apply hard cooperative-power threshold against prey energy.
4. In gate mode, apply additional probabilistic success from summed trait contributions.
5. On success, distribute fixed kill reward and remove prey.

## Reproduction Logic: Direct Contrast

### `stag_hunt_forward_view`

- Predators and prey reproduce when crossing energy thresholds (`predpreygrass_rllib_env.py:453`, `predpreygrass_rllib_env.py:457`).
- Parent pays child start energy directly (`predpreygrass_rllib_env.py:1360`, `predpreygrass_rllib_env.py:1452`).
- Spawn can be blocked by type-specific ID pool limits (`predpreygrass_rllib_env.py:1311`, `predpreygrass_rllib_env.py:1403`).

### `emerging_cooperation.py`

- Prey reproduction is energy-thresholded + stochastic and crowding-limited (`emerging_cooperation.py:163`, `emerging_cooperation.py:187`).
- Predator reproduction is thresholded + stochastic with crowding and prey-availability scaling (`emerging_cooperation.py:288`, `emerging_cooperation.py:301`).
- Parent energy halves on predator reproduction; predator trait mutation remains active (`emerging_cooperation.py:305`, `emerging_cooperation.py:311`).

## Config-Level Differences (Current Defaults)

| Concept | `stag_hunt_forward_view` defaults | `emerging_cooperation.py` defaults |
|---|---|---|
| Max steps | `max_steps = 1000` (`config_env_stag_hunt_forward_view.py:4`) | `STEPS = 2500` (`emerging_cooperation.py:51`) |
| Initial predators/prey | `10` predators, `20` prey total (`config_env_stag_hunt_forward_view.py:59`, `config_env_stag_hunt_forward_view.py:61`, `config_env_stag_hunt_forward_view.py:62`) | `100` predators, `1400` prey (`emerging_cooperation.py:47`, `emerging_cooperation.py:48`) |
| Predator baseline costs | `energy_loss_per_step_predator = 0.08` (`config_env_stag_hunt_forward_view.py:27`) | `METAB_PRED + MOVE_COST + COOP_COST * coop` (`emerging_cooperation.py:54`, `emerging_cooperation.py:55`, `emerging_cooperation.py:56`) |
| Capture knobs | `team_capture_margin`, `team_capture_join_cost`, scavenger share (`config_env_stag_hunt_forward_view.py:47`, `config_env_stag_hunt_forward_view.py:50`, `config_env_stag_hunt_forward_view.py:51`) | `HUNT_RULE`, `P0`, `HUNTER_POOL_R`, `COOP_POWER_FLOOR` (`emerging_cooperation.py:66`, `emerging_cooperation.py:67`, `emerging_cooperation.py:69`, `emerging_cooperation.py:70`) |
| Prey economy | Typed prey bite sizes + grass regen (`config_env_stag_hunt_forward_view.py:42`, `config_env_stag_hunt_forward_view.py:64`, `config_env_stag_hunt_forward_view.py:66`) | Single prey type with explicit metabolism/move cost/bite size + grass regen (`emerging_cooperation.py:79`, `emerging_cooperation.py:80`, `emerging_cooperation.py:83`, `emerging_cooperation.py:88`) |

## Why Direct Tuning Transfer Is Still Hard

- Cooperation semantics differ by design (trait inheritance vs action decision).
- Team-capture game details differ (`join_cost`, free-rider scavenging, explicit join intent).
- State/action interfaces differ (RLlib multi-agent API vs single simulation loop).
- Movement constraints differ (bounded + walls/LOS vs wrapped random walk).

So the systems are now ecologically closer, but parameter equivalence is still not one-to-one.
