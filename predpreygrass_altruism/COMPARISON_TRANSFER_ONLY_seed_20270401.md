# Transfer-Only Two-Mode Comparison

- Script: `predpreygrass_altruism/predpreygrass_transfer_only_altruism_vs_selfish.py`
- Modes: `altruist_only` vs `any_predator`
- Replicates: `10`
- Seed base: `20270401`
- Seeds: `[20270401, 20278320, 20286239, 20294158, 20302077, 20309996, 20317915, 20325834, 20333753, 20341672]`

## `altruist_only`

- final means: altruists=424.2, selfish=0.0, prey=5018.7
- selfish_ext_rate=1.000, coex_tail=1.000, prop_tail=1.000, tail_alt_freq=100.0%
- selfish_abs_ext_step_mean=172.5, mean_transfers/tick=718.489

## `any_predator`

- final means: altruists=0.0, selfish=247.0, prey=5263.9
- selfish_ext_rate=0.000, coex_tail=0.031, prop_tail=1.000, tail_alt_freq=0.1%
- selfish_abs_ext_step_mean=nan, mean_transfers/tick=23.810

## Paired Outcome Counts

- `altruist_only` selfish extinction count: `10/10`
- `any_predator` selfish extinction count: `0/10`
- `altruist_only` altruist extinction count: `0/10`
- `any_predator` altruist extinction count: `10/10`

## Paired Per-Seed Finals (A,S)

| seed | altruist_only | any_predator |
|---:|---:|---:|
| 20270401 | (462,0) | (0,270) |
| 20278320 | (451,0) | (0,267) |
| 20286239 | (381,0) | (0,271) |
| 20294158 | (393,0) | (0,197) |
| 20302077 | (415,0) | (0,244) |
| 20309996 | (389,0) | (0,261) |
| 20317915 | (440,0) | (0,269) |
| 20325834 | (450,0) | (0,183) |
| 20333753 | (445,0) | (0,207) |
| 20341672 | (416,0) | (0,301) |
