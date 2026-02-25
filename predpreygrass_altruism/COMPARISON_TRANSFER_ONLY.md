# Transfer-Only Two-Mode Comparison

- Script: `predpreygrass_altruism/predpreygrass_transfer_only_altruism_vs_selfish.py`
- Modes: `altruist_only` vs `any_predator`
- Replicates: `10`
- Seed base: `122913`
- Seeds: `[122913, 130832, 138751, 146670, 154589, 162508, 170427, 178346, 186265, 194184]`

## `altruist_only`

- final means: altruists=422.2, selfish=0.0, prey=4780.6
- selfish_ext_rate=1.000, coex_tail=1.000, prop_tail=1.000, tail_alt_freq=100.0%
- selfish_abs_ext_step_mean=171.0, mean_transfers/tick=713.159

## `any_predator`

- final means: altruists=0.0, selfish=279.4, prey=5180.8
- selfish_ext_rate=0.000, coex_tail=0.021, prop_tail=1.000, tail_alt_freq=0.1%
- selfish_abs_ext_step_mean=nan, mean_transfers/tick=24.595

## Paired Outcome Counts

- `altruist_only` selfish extinction count: `10/10`
- `any_predator` selfish extinction count: `0/10`
- `altruist_only` altruist extinction count: `0/10`
- `any_predator` altruist extinction count: `10/10`

## Paired Per-Seed Finals (A,S)

| seed | altruist_only | any_predator |
|---:|---:|---:|
| 122913 | (402,0) | (0,270) |
| 130832 | (459,0) | (0,309) |
| 138751 | (417,0) | (0,351) |
| 146670 | (431,0) | (0,243) |
| 154589 | (417,0) | (0,287) |
| 162508 | (413,0) | (0,280) |
| 170427 | (434,0) | (0,263) |
| 178346 | (425,0) | (0,291) |
| 186265 | (394,0) | (0,258) |
| 194184 | (430,0) | (0,242) |
