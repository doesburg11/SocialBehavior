# Cooperation Model

A Python/NumPy simulation of resource sharing with two strategies: cooperative vs greedy cows grazing on regrowing grass. Includes a Pygame UI for interactive exploration and a quick CLI runner for batch simulation. This folder also contains the original NetLogo model file for reference.

## Features
- Patch-based 2D world with regrowing grass
- Two strategies: cooperative (harvest conservatively) vs greedy (harvest whenever possible)
- Energy metabolism, movement, reproduction, and death
- Adjustable parameters via UI sliders
- Live plot of population sizes saved to `coop_plot.png`

## Files
- `cooperation_model.py` — Core simulation (importable class and CLI entry)
- `cooperation_pygame_ui.py` — Pygame-based interactive UI with sliders and plotting
- `Cooperation.nlogox` — Original NetLogo model (reference)

## Quick start
### Run the CLI simulation
```bash
python cooperation_model.py
```
Prints ticks and final counts of cooperative vs greedy cows.

### Run the interactive UI
```bash
python cooperation_pygame_ui.py
```
Controls (buttons on the right):
- Start/Stop — toggle simulation
- Reset — reset the world
- Plot: ON/OFF — toggle live plot (saved as `coop_plot.png`)
- Step — advance one tick when stopped

Keyboard:
- R — reset (same as Reset button)

## Parameters (tunable)
Defined at the top of `cooperation_model.py` and exposed in the UI:
- `grid_size` — world size (default 50)
- `initial_cows` — starting population
- `cooperative_probability` — chance a new cow is cooperative
- `reproduction_cost`, `reproduction_threshold`
- `stride_length`, `metabolism`
- `high_growth_chance`, `low_growth_chance` — grass regrowth probabilities (percent)
- `grass_energy`, `max_grass_height`, `low_high_threshold`

## Requirements
- Python 3.8+
- numpy
- pygame (for UI)
- matplotlib (for plotting)

Install with pip:
```bash
pip install numpy pygame matplotlib
```
If building Pygame on Linux requires a compiler, you may need:
```bash
conda install -y -c conda-forge gcc
```

## Notes
- The UI saves plots to `coop_plot.png` in this folder.
- To script experiments, import `CooperationModel` from `cooperation_model.py` and call `step()` or `run(steps=...)`.

## Reference
- NetLogo BEAGLE/EACH models (original inspiration)
