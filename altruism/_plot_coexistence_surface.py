# 3D surface plot of coexistence probability = 1 as a function of b, c, and harshness (fixed disease)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load results
df = pd.read_csv('altruism/grid_search_results.csv')

# Filter for fixed disease and coexistence probability = 1
fixed_disease = 0.26
surface_df = df[(df['disease'] == fixed_disease) & (df['coexist_prob'] == 1)]

# Prepare data
b = surface_df['benefit_from_altruism'].values
c = surface_df['cost_of_altruism'].values
h = surface_df['harshness'].values

X = np.column_stack((b, c)).astype(np.float32)
y = h.astype(np.float32)
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


# Define a simpler neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train neural network
for epoch in range(2000):
    optimizer.zero_grad()
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Create grid for smooth surface
b_grid = np.linspace(b.min(), b.max(), 50)
c_grid = np.linspace(c.min(), c.max(), 50)
B, C = np.meshgrid(b_grid, c_grid)
X_grid = np.column_stack((B.ravel(), C.ravel())).astype(np.float32)
X_grid_tensor = torch.from_numpy(X_grid)
with torch.no_grad():
    H_pred = net(X_grid_tensor).numpy().reshape(B.shape)
    H_pred = np.clip(H_pred, 0.0, 1.0)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the PyTorch neural network surface
surf = ax.plot_surface(B, C, H_pred, cmap='YlOrBr', alpha=0.8, edgecolor='none')
# Overlay the original scatter points for reference
ax.scatter(b, c, h, c='red', s=10, alpha=0.7, label='Data points (p=1)')

ax.set_xlabel('Benefit from altruism (b)')
ax.set_ylabel('Cost of altruism (c)')
ax.set_zlabel('Harshness')
ax.set_title('Coexistence Probability = 1 Surface (PyTorch NN)\n(disease = 0.26)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Harshness')
ax.legend()
ax.set_zlim(0.0, 1.0)
plt.tight_layout()
plt.show()
