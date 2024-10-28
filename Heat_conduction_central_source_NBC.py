import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers, neurons, activation=nn.Tanh()):
        super(PINN, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3, neurons))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(neurons, neurons))
        self.layers.append(nn.Linear(neurons, 1))

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        output = inputs
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        output = self.layers[-1](output) + 25  # Add 25°C to the output to start from room temperature
        return output

# Define the loss functions
def pde_loss(model, x, y, t, epsilon, f):
    u = model(x, y, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    residual = u_t - epsilon * (u_xx + u_yy) - f
    return torch.mean(residual ** 2)

def initial_loss(model, x, y, t, u0):
    u = model(x, y, t)
    return torch.mean((u - u0) ** 2)

def boundary_loss(model, x, y, t, length):
    u = model(x, y, t)
    
    x_boundary = (x <= 1e-6) | (x >= length - 1e-6)
    y_boundary = (y <= 1e-6) | (y >= length - 1e-6)
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    loss_x = torch.mean(u_x[x_boundary]**2)
    loss_y = torch.mean(u_y[y_boundary]**2)
    
    return loss_x + loss_y

def generate_data(n_points, length, total_time):
    x = torch.rand(n_points, 1, requires_grad=True) * length
    y = torch.rand(n_points, 1, requires_grad=True) * length
    t = torch.rand(n_points, 1, requires_grad=True) * total_time
    
    n_boundary = n_points // 10
    x_boundary = torch.cat([torch.zeros(n_boundary, 1), torch.full((n_boundary, 1), length)], dim=0)
    y_boundary = torch.cat([torch.zeros(n_boundary, 1), torch.full((n_boundary, 1), length)], dim=0)
    t_boundary = torch.rand(2 * n_boundary, 1, requires_grad=True) * total_time
    
    x = torch.cat([x, x_boundary, torch.rand(2 * n_boundary, 1) * length], dim=0)
    y = torch.cat([y, torch.rand(2 * n_boundary, 1) * length, y_boundary], dim=0)
    t = torch.cat([t, t_boundary, t_boundary], dim=0)
    
    return x.to(device), y.to(device), t.to(device)

def heat_source(x, y, center_x, center_y, radius, strength, t):
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    return torch.where(t > 0, strength * torch.exp(-distance**2 / (2 * radius**2)), torch.zeros_like(t))

# Simulation parameters
length = 1.0
total_time = 5.0
n_points = 1000
n_points_plot = 100
layers = 5
neurons = 50
epochs = 3000
learning_rate = 0.001
epsilon = 0.1

# Initialize model and optimizer
model = PINN(layers, neurons).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
weight_residual = 1.0
weight_initial = 1.0
weight_boundary = 1.0

# Heat source parameters
center_x, center_y = 0.5, 0.5
heat_radius = 0.1
heat_strength = 500.0  # Increased heat strength to reach higher temperatures

# Training loop
for epoch in range(epochs):
    x, y, t = generate_data(n_points, length, total_time)
    u0 = torch.full_like(x, 25)  # Initial condition (room temperature)
    f = heat_source(x, y, center_x, center_y, heat_radius, heat_strength, t)

    optimizer.zero_grad()
    loss_residual = pde_loss(model, x, y, t, epsilon, f)
    loss_initial = initial_loss(model, x, y, torch.zeros_like(t), u0)
    loss_boundary = boundary_loss(model, x, y, t, length)
    loss = weight_residual * loss_residual + weight_initial * loss_initial + weight_boundary * loss_boundary
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


cmap = plt.get_cmap('hot')

# Visualization
fig, ax = plt.subplots(figsize=(8, 7))
x_plot = np.linspace(0, length, n_points_plot)
y_plot = np.linspace(0, length, n_points_plot)
x_plot, y_plot = np.meshgrid(x_plot, y_plot)
x_plot = torch.tensor(x_plot.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
y_plot = torch.tensor(y_plot.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

# Create a static colorbar
t_plot = torch.zeros_like(x_plot).to(device)
with torch.no_grad():
    u_plot = model(x_plot, y_plot, t_plot).cpu().numpy().reshape(n_points_plot, n_points_plot)
im = ax.imshow(u_plot, extent=[0, length, 0, length], origin='lower', cmap=cmap, vmin=25, vmax=125, animated=True)
fig.colorbar(im, ax=ax, label='Temperature (°C)')

# Move the text object for displaying temperature below the plot
temp_text = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='top', fontsize=12)

def update(frame):
    t_plot = torch.full_like(x_plot, frame * 0.1).to(device)
    with torch.no_grad():
        u_plot = model(x_plot, y_plot, t_plot).cpu().numpy().reshape(n_points_plot, n_points_plot)
    
    im.set_array(u_plot)
    ax.set_title(f'2D Heat Conduction with Center Heat Source at t={frame * 0.1:.1f}s')
    
    center_temp = u_plot[n_points_plot//2, n_points_plot//2]
    edge_temp = (u_plot[0, 0] + u_plot[0, -1] + u_plot[-1, 0] + u_plot[-1, -1]) / 4
    temp_text.set_text(f'Center Temp: {center_temp:.2f}°C, Edge Temp: {edge_temp:.2f}°C')
    
    print(f"Time: {frame * 0.1:.1f}s, Center Temp: {center_temp:.2f}°C, Edge Temp: {edge_temp:.2f}°C")
    
    return [im, temp_text]

ax.set_xlabel('x')
ax.set_ylabel('y')

# Adjust the layout to make room for the text below
plt.subplots_adjust(bottom=0.2)

anim = FuncAnimation(fig, update, frames=51, interval=200, blit=True)
anim.save('heat_conduction_center_source_diffusion.gif', writer='pillow', fps=10)
plt.close(fig)

print("Simulation complete. GIF saved as 'heat_conduction_center_source_diffusion.gif'")
