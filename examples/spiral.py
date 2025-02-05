import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(n_points, noise_std=0.1):
    """Generate a 2D spiral dataset with optional noise."""
    t = np.linspace(0, 4 * np.pi, n_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    spiral = np.stack([x, y], axis=1)
    noise = np.random.normal(scale=noise_std, size=spiral.shape)
    return spiral + noise

# Generate and visualize the dataset
n_points = 1000
spiral_data = generate_spiral(n_points)
plt.scatter(spiral_data[:, 0], spiral_data[:, 1], s=5)
plt.title("Noisy Spiral Dataset")
plt.axis("equal")
plt.show()

class LinearSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def noise_schedule(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def forward_process(self, x0, t):
        """Perturb x0 with time-dependent Gaussian noise."""
        alpha_t = np.exp(-0.5 * self.noise_schedule(t) * t)
        noise = np.random.normal(size=x0.shape)
        xt = alpha_t * x0 + np.sqrt(1 - alpha_t**2) * noise
        return xt, noise

import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        return self.fc(t.unsqueeze(-1))

class UNet(nn.Module):
    def __init__(self, input_dim, base_channels, time_dim):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t).unsqueeze(-1)  # Time embedding
        x = x + t_emb
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

from torch.optim import Adam

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2
base_channels = 64
time_dim = 16
epochs = 500
batch_size = 64
lr = 1e-3

# Model, optimizer, and loss
model = UNet(input_dim, base_channels, time_dim).to(device)
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Dataset preparation
spiral_tensor = torch.tensor(spiral_data, dtype=torch.float32).to(device)
data_loader = torch.utils.data.DataLoader(spiral_tensor, batch_size=batch_size, shuffle=True)

# Training
sde = LinearSDE()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x0 in data_loader:
        x0 = x0.transpose(1, 2)  # Shape (batch_size, input_dim, 1)
        t = torch.rand(x0.size(0), device=device) * sde.T
        xt, noise = sde.forward_process(x0, t)

        optimizer.zero_grad()
        predicted_noise = model(xt, t)
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}")


# Generate noisy data and denoise it
model.eval()
x0_sample = spiral_tensor[:100].unsqueeze(-1)  # Take a few samples
t_sample = torch.tensor([sde.T], device=device)  # Maximum noise level
xt, _ = sde.forward_process(x0_sample, t_sample)

with torch.no_grad():
    denoised = model(xt, t_sample)

# Plot noisy vs. denoised
xt_np = xt.cpu().squeeze(-1).numpy()
denoised_np = denoised.cpu().squeeze(-1).numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(xt_np[:, 0], xt_np[:, 1], s=5, label="Noisy Data")
plt.title("Noisy Data")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.scatter(denoised_np[:, 0], denoised_np[:, 1], s=5, label="Denoised Data")
plt.title("Denoised Data")
plt.axis("equal")
plt.show()


