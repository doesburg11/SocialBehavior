# Interpolated 3D body of % altruists as a function of b, c, and harshness (fixed disease) using Plotly
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Load results
df = pd.read_csv('altruism/grid_search_results_extended.csv')

# Filter for fixed disease
fixed_disease = 0.26
body_df = df[df['disease'] == fixed_disease]

# Prepare data
b = body_df['benefit_from_altruism'].values
c = body_df['cost_of_altruism'].values
h = body_df['harshness'].values
altruist_avg = body_df['altruist_avg'].values
selfish_avg = body_df['selfish_avg'].values

# Calculate % of altruists as fraction of total cells (2601)
altruist_pct = altruist_avg / 2601
selfish_pct = selfish_avg / 2601

# Debugging output for rows with 95% altruists or selfish
print((altruist_pct > 0.95).sum(), "rows with >95% altruists")
print((selfish_pct > 0.95).sum(), "rows with >95% selfish")

# Create a dense 3D grid
b_grid = np.linspace(b.min(), b.max(), 40)
c_grid = np.linspace(c.min(), c.max(), 40)
h_grid = np.linspace(h.min(), h.max(), 40)
B, C, H = np.meshgrid(b_grid, c_grid, h_grid, indexing='ij')

# Interpolate altruist_pct onto the grid (use 'nearest' for robustness)
points = np.column_stack((b, c, h))
values = altruist_pct
altruist_interp = griddata(points, values, (B, C, H), method='nearest', fill_value=0)

# Interpolate selfish_pct onto the grid (use 'nearest' for robustness)
selfish_interp = griddata(points, selfish_pct, (B, C, H), method='nearest', fill_value=0)

# Define button values for both altruist and selfish: 0.0, 0.1, ..., 0.8, 0.9, 0.99
altruist_options = [0.1 * i for i in range(10)] + [0.95, 0.0]
altruist_options = sorted(set(altruist_options))  # Ensure unique and sorted

selfish_options = [0.1 * i for i in range(10)] + [0.95, 0.0]
selfish_options = sorted(set(selfish_options))  # Ensure unique and sorted

# Create Plotly figure with sliders for both altruist and selfish
fig = go.Figure()

# Add altruist isosurfaces
for aval in altruist_options:
    fig.add_trace(go.Isosurface(
        x=B.flatten(),
        y=C.flatten(),
        z=H.flatten(),
        value=altruist_interp.flatten(),
        isomin=aval,
        isomax=aval + 0.01,
        surface_count=1,
        colorscale='Turbo',
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        visible=False,
        name=f'% altruist_avg={aval}'
    ))

# Add selfish isosurfaces
for sval in selfish_options:
    fig.add_trace(go.Isosurface(
        x=B.flatten(),
        y=C.flatten(),
        z=H.flatten(),
        value=selfish_interp.flatten(),
        isomin=sval,
        isomax=sval + 0.05,
        surface_count=1,
        colorscale='Viridis',
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        visible=False,
        name=f'% selfish_avg={sval}'
    ))

# Set first altruist isosurface visible by default
fig.data[0].visible = True

# Create sliders for altruist and selfish
steps_altruist = []
for i, aval in enumerate(altruist_options):
    step = dict(
        method="update",
        args=[{"visible": [j == i for j in range(len(altruist_options))] + [False]*len(selfish_options)}],
        label=f"{aval:.2f}"
    )
    steps_altruist.append(step)

steps_selfish = []
for i, sval in enumerate(selfish_options):
    step = dict(
        method="update",
        args=[{"visible": [False]*len(altruist_options) + [j == i for j in range(len(selfish_options))]}],
        label=f"{sval:.2f}"
    )
    steps_selfish.append(step)

sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "% altruist_avg = "},
        pad={"t": 50},
        steps=steps_altruist,
        x=0.02,
        y=0.95,
        len=0.35
    ),
    dict(
        active=0,
        currentvalue={"prefix": "% selfish_avg = "},
        pad={"t": 50},
        steps=steps_selfish,
        x=0.02,
        y=0.85,
        len=0.35
    )
]

fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis_title='Benefit from altruism (b)',
        yaxis_title='Cost of altruism (c)',
        zaxis_title='Harshness',
        zaxis=dict(range=[0, 1])
    ),
    title='Select % altruist_avg or % selfish_avg isosurface (disease = 0.26)'
)

fig.show()
