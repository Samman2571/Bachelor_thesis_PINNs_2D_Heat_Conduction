import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-1], 1))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the PDE residual
def pde_residual(model, x, y, t):
    u = model(torch.cat([x, y, t], dim=1))
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    return u_t - alpha * (u_xx + u_yy)

# Hyperparameters and constants
layers = [32, 32, 32, 32]
alpha = 0.1  # Thermal diffusivity
learning_rate = 1e-4
num_epochs = 20000

# Instantiate the model
model = PINN(layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training data
num_points = 1000
x = torch.rand((num_points, 1), requires_grad=True) * 2 - 1  # x in [-1, 1]
y = torch.rand((num_points, 1), requires_grad=True) * 2 - 1  # y in [-1, 1]
t = torch.rand((num_points, 1), requires_grad=True) * 2  # t in [0, 2]

# Initial condition data (interior points only)
num_ic_points = 1000
x_ic = torch.rand((num_ic_points, 1), requires_grad=True) * 1.8 - 0.9  # x in [-0.9, 0.9]
y_ic = torch.rand((num_ic_points, 1), requires_grad=True) * 1.8 - 0.9  # y in [-0.9, 0.9]
t_ic = torch.zeros_like(x_ic).requires_grad_(True)

# Non-uniform initial condition with two hot quadrants and two cold quadrants
u_ic = 25 + 25 * torch.sign(x_ic * y_ic)

# Boundary condition data
num_bc_points = 200
x_bc_left = torch.ones((num_bc_points, 1), requires_grad=True) * -1  # x = -1
x_bc_right = torch.ones((num_bc_points, 1), requires_grad=True) * 1  # x = 1
y_bc_bottom = torch.ones((num_bc_points, 1), requires_grad=True) * -1  # y = -1
y_bc_top = torch.ones((num_bc_points, 1), requires_grad=True) * 1  # y = 1
y_bc = torch.linspace(-1, 1, num_bc_points).view(-1, 1).requires_grad_(True)
x_bc = torch.linspace(-1, 1, num_bc_points).view(-1, 1).requires_grad_(True)
t_bc = torch.rand((num_bc_points, 1)) * 2  # t in [0, 2]
t_bc.requires_grad_(True)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # PDE residual loss
    pde_loss = torch.mean(pde_residual(model, x, y, t) ** 2)
    
    # Initial condition loss
    u_pred_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
    ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)
    
    # Boundary condition loss (Neumann BC)
    u_left = model(torch.cat([x_bc_left, y_bc, t_bc], dim=1))
    u_right = model(torch.cat([x_bc_right, y_bc, t_bc], dim=1))
    u_bottom = model(torch.cat([x_bc, y_bc_bottom, t_bc], dim=1))
    u_top = model(torch.cat([x_bc, y_bc_top, t_bc], dim=1))
    
    u_x_left = torch.autograd.grad(u_left, x_bc_left, grad_outputs=torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_bc_right, grad_outputs=torch.ones_like(u_right), create_graph=True)[0]
    u_y_bottom = torch.autograd.grad(u_bottom, y_bc_bottom, grad_outputs=torch.ones_like(u_bottom), create_graph=True)[0]
    u_y_top = torch.autograd.grad(u_top, y_bc_top, grad_outputs=torch.ones_like(u_top), create_graph=True)[0]
    
    bc_loss = torch.mean(u_x_left**2 + u_x_right**2 + u_y_bottom**2 + u_y_top**2)
    
    # Total loss
    loss = 10 * pde_loss + ic_loss + 10 * bc_loss
    
    # Backward pass and optimization
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, PDE Loss: {pde_loss.item():.4f}, IC Loss: {ic_loss.item():.4f}, BC Loss: {bc_loss.item():.4f}")

# Saving the model
model_path = 'pinn_heat_redistribution_neumann_2sec.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}.")

# Parameters for GIF creation
num_points = 100
num_timesteps = 100  # Increased number of timesteps for smoother animation

# Generate grid points
x = torch.linspace(-1, 1, num_points)
y = torch.linspace(-1, 1, num_points)
x, y = torch.meshgrid(x, y, indexing='ij')
x, y = x.reshape(-1, 1), y.reshape(-1, 1)

# Path to save frames and GIF
save_path = r'C:\Users\samma\OneDrive\Desktop\Thesis_Samman_Aryal\V5\V6\FInals\Images\heat_redistribution_neumann_2sec'
os.makedirs(save_path, exist_ok=True)

# Generate frames
frames = []
for t in np.linspace(0, 2, num_timesteps):  # Changed to 2 seconds
    t_tensor = torch.ones_like(x) * t
    input_tensor = torch.cat([x, y, t_tensor], dim=1)
    with torch.no_grad():
        u_pred = model(input_tensor).reshape(num_points, num_points).numpy()
    
    # Print min and max values for debugging
    print(f"Time {t:.2f}: Min = {np.min(u_pred):.2f}, Max = {np.max(u_pred):.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(u_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', vmin=0, vmax=100)
    plt.colorbar(label='Temperature')
    plt.title(f'Time = {t:.2f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save frame
    frame_path = os.path.join(save_path, f'frame_{int(t*50):03d}.png')
    plt.savefig(frame_path)
    frames.append(imageio.v2.imread(frame_path))
    plt.close()

# Create GIF
gif_path = os.path.join(save_path, 'heat_redistribution_neumann_2sec.gif')
imageio.mimsave(gif_path, frames, fps=10)  # Increased fps for smoother animation

print(f"GIF created successfully at {gif_path}.")