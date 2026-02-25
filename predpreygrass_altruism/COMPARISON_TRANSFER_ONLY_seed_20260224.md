# Transfer-Only Two-Mode Comparison

- Script: `predpreygrass_altruism/predpreygrass_transfer_only_altruism_vs_selfish.py`
- Modes: `altruist_only` vs `any_predator`
- Replicates: `10`
- Seed base: `20260224`
- Seeds: `[20260224, 20268143, 20276062, 20283981, 20291900, 20299819, 20307738, 20315657, 20323576, 20331495]`

## `altruist_only`

- final means: altruists=423.9, selfish=0.0, prey=4848.2
- selfish_ext_rate=1.000, coex_tail=1.000, prop_tail=1.000, tail_alt_freq=100.0%
- selfish_abs_ext_step_mean=177.4, mean_transfers/tick=715.572

## `any_predator`

- final means: altruists=0.0, selfish=257.4, prey=5470.2
- selfish_ext_rate=0.000, coex_tail=0.065, prop_tail=1.000, tail_alt_freq=0.3%
- selfish_abs_ext_step_mean=nan, mean_transfers/tick=56.182

## Paired Outcome Counts

- `altruist_only` selfish extinction count: `10/10`
- `any_predator` selfish extinction count: `0/10`
- `altruist_only` altruist extinction count: `0/10`
- `any_predator` altruist extinction count: `10/10`

## Paired Per-Seed Finals (A,S)

| seed | altruist_only | any_predator |
|---:|---:|---:|
| 20260224 | (461,0) | (0,228) |
| 20268143 | (396,0) | (0,295) |
| 20276062 | (429,0) | (0,244) |
| 20283981 | (394,0) | (0,200) |
| 20291900 | (411,0) | (0,227) |
| 20299819 | (442,0) | (0,224) |
| 20307738 | (417,0) | (0,301) |
| 20315657 | (400,0) | (0,284) |
| 20323576 | (438,0) | (0,295) |
| 20331495 | (451,0) | (0,276) |
