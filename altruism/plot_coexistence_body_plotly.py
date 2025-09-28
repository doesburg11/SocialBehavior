# Interpolated 3D body of coexistence probability = 1 as a function of b, c, and harshness (fixed disease) using Plotly
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Load results
df = pd.read_csv('altruism/grid_search_results.csv')

# Filter for fixed disease
fixed_disease = 0.26
body_df = df[df['disease'] == fixed_disease]

# Prepare data
b = body_df['benefit_from_altruism'].values
c = body_df['cost_of_altruism'].values
h = body_df['harshness'].values
coexist = body_df['coexist_prob'].values

# Create a dense 3D grid
b_grid = np.linspace(b.min(), b.max(), 40)
c_grid = np.linspace(c.min(), c.max(), 40)
h_grid = np.linspace(h.min(), h.max(), 40)
B, C, H = np.meshgrid(b_grid, c_grid, h_grid, indexing='ij')

# Interpolate coexist_prob onto the grid (use 'nearest' for robustness)
points = np.column_stack((b, c, h))
values = coexist
coexist_interp = griddata(points, values, (B, C, H), method='nearest', fill_value=0)

# Radio button values for coexist_prob isosurface
harshness_options = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

# Create Plotly figure with updatemenus (radio buttons)
fig = go.Figure()

for hval in harshness_options:
    fig.add_trace(go.Isosurface(
        x=B.flatten(),
        y=C.flatten(),
        z=H.flatten(),
        value=coexist_interp.flatten(),
        isomin=hval,
        isomax=hval + 0.01,
        surface_count=1,
        colorscale='Turbo',
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        visible=(hval == harshness_options[0]),  # Only first is visible by default
        name=f'coexist_prob={hval}'
    ))

# Add radio buttons to toggle isosurfaces
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=[
                dict(
                    label=f"coexist_prob = {hval}",
                    method="update",
                    args=[{"visible": [hval == opt for opt in harshness_options]}]
                ) for hval in harshness_options
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top"
        )
    ],
    scene=dict(
        xaxis_title='Benefit from altruism (b)',
        yaxis_title='Cost of altruism (c)',
        zaxis_title='Harshness',
        zaxis=dict(range=[0, 1])
    ),
    title='Select coexist_prob isosurface (disease = 0.26)'
)

fig.show()
