# SocialBehavior

A collection of agent-based models exploring human social behavior. Includes interactive UIs and grid search tools for parameter exploration.

## Environments
This repo uses a project-local Conda environment stored at `.conda/` so it travels with the workspace and VS Code can auto-select it.

- Interpreter path: `/home/doesburg/Projects/SocialBehavior/.conda/bin/python`
- VS Code setting: see `.vscode/settings.json` (we set `python.defaultInterpreterPath` and enable terminal activation)

Activate the environment in a terminal when running commands manually:
```bash
source ./.conda/bin/activate
# or run without activation using the interpreter directly:
./.conda/bin/python -m pip install -r altruism/requirements.txt
./.conda/bin/python altruism/altruism_model.py
```

If you see a “bad interpreter” error, regenerate entry scripts (pip, etc.) with:
```bash
./.conda/bin/python -m pip install --upgrade --force-reinstall pip setuptools wheel
```

## Models

### Altruism Model
- **Description:** Patch-based grid simulation of altruism vs selfishness, ported from NetLogo to Python/NumPy.
- **Features:**
	- Each cell can be empty (black), selfish (green), or altruist (pink)
	- Simulates benefit/cost of altruism, fitness, and generational updates
	- Fully vectorized NumPy implementation for fast simulation
	- Pygame UI for interactive exploration
	- Matplotlib plots for population dynamics
	- Grid search for parameter sweeps
- **Files:**
	- `altruism_model.py`: Core simulation logic (importable class, CLI demo, and plotting)
	- `altruism_pygame_ui.py`: Pygame-based interactive UI
	- `altruism_grid_search.py`: Grid search for coexistence probabilities
	- `plot_coexistence_surface.py`, `plot_heatmaps.py`: Visualization scripts
	- `grid_search_results.csv`: Results from grid search
- **Usage:**
	- Run CLI demo:
		```bash
		python altruism/altruism_model.py --steps 200 --width 101 --height 101 --seed 42
		```
	- Run Pygame UI:
		```bash
		python altruism/altruism_pygame_ui.py
		```
	- Run grid search:
		```bash
		python altruism/altruism_grid_search.py
		```
- **Requirements:**
	- Python 3.8+
	- numpy
	- pygame (for UI)
	- matplotlib (for plotting)
	- torch (for surface fitting)

### Cooperation Model
- **Description:** Evolutionary biology model of greedy vs cooperative agents (cows) competing for grass, ported from NetLogo.
- **Features:**
	- Agents move, eat, reproduce, and die based on energy and grass availability
	- Cooperative cows avoid eating low grass, greedy cows eat regardless
	- Grass regrows at different rates depending on height
	- Pygame UI for visualization
- **Files:**
	- `cooperation_model.py`: Core simulation logic
	- `cooperation_pygame_ui.py`: Pygame-based interactive UI
	- `Cooperation.nlogox`: Original NetLogo model
- **Usage:**
	- Run CLI demo:
		```bash
		python cooperation/cooperation_model.py
		```
	- Run Pygame UI:
		```bash
		python cooperation/cooperation_pygame_ui.py
		```
- **Requirements:**
	- Python 3.8+
	- numpy
	- pygame
	- matplotlib

## Installation
Install dependencies:
```bash
pip install numpy pygame matplotlib torch
```
For Pygame visualization, you may need:
```bash
conda install -y -c conda-forge gcc=14.2.0
```

## References
- Original NetLogo models from Uri Wilensky and the EACH unit (Evolution of Altruistic and Cooperative Habits)
- See `altruism/README.md` and `cooperation/Cooperation.nlogox` for more details

