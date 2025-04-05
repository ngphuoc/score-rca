import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
epochs = 200
batch_size = 512
lr = 1e-3
sigma_min = 0.01
sigma_max = 50.0

# Simple 2D dataset (Gaussian mixture)
def sample_data(n):
    clusters = torch.tensor([[3, 3], [-3, -3], [3, -3], [-3, 3]], device=device)
    idx = torch.randint(0, 4, (n,), device=device)
    return clusters[idx] + 0.3 * torch.randn(n, 2, device=device)

# Simple U-Net (MLP-based)
class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        # Compute log_sigma = log(sigma_min * (sigma_max/sigma_min)^t)
        log_sigma = torch.log(sigma_min * (sigma_max / sigma_min)**t).unsqueeze(-1)
        # Expand to match the shape of x (batch, 1) and concatenate with x along the last dimension.
        inputs = torch.cat([x, log_sigma.expand_as(x[:, :1])], dim=-1)
        return self.net(inputs)

# VE-SDE definition
class VESDE(torchsde.SDEIto):
    def __init__(self, sigma_min, sigma_max):
        super().__init__(noise_type="diagonal")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def f(self, t, y):
        return torch.zeros_like(y)

    def g(self, t, y):
        sigma_t = self.sigma(t)
        # Convert constants to tensors before passing to torch.log
        sigma_min_tensor = torch.tensor(self.sigma_min, device=y.device)
        sigma_max_tensor = torch.tensor(self.sigma_max, device=y.device)

        constant = torch.sqrt(torch.tensor(2.0, device=y.device) *
                              (torch.log(sigma_max_tensor) - torch.log(sigma_min_tensor)))
        return sigma_t * constant

    # def g(self, t, y):
    #     sigma_t = self.sigma(t)
    #     # Compute constant sqrt(2 * (log(sigma_max) - log(sigma_min)))
    #     constant = torch.sqrt(torch.tensor(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)), device=y.device))
    #     return sigma_t * constant

# Model, optimizer, SDE setup
model = UNet2D().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
sde = VESDE(sigma_min, sigma_max)

# Training loop (Score Matching Objective)
for epoch in tqdm(range(epochs), desc="Training"):
    optimizer.zero_grad()

    x0 = sample_data(batch_size)
    t = torch.rand(batch_size, device=device)

    sigma_t = sde.sigma(t)
    noise = torch.randn_like(x0)
    xt = x0 + sigma_t[:, None] * noise

    score_target = -noise / sigma_t[:, None]
    score_pred = model(xt, t)

    loss = ((score_pred - score_target) ** 2).mean()
    loss.backward()
    optimizer.step()

# Sampling via Reverse SDE (Euler-Maruyama integration)
@torch.no_grad()
def sample_sde(model, n_samples=2000, steps=500):
    # Start from t = 1 (noisiest state)
    x = torch.randn(n_samples, 2, device=device) * sde.sigma(torch.tensor(1.0, device=device))
    dt = -1.0 / steps

    for step in tqdm(range(steps), desc="Sampling"):
        # Set current time: decreasing from 1 to 0
        t_val = 1.0 - step / steps
        time_batch = torch.full((n_samples,), t_val, device=device)
        g = sde.g(time_batch, x).unsqueeze(-1)  # unsqueeze to shape [n_samples, 1]
        drift = -g**2 * model(x, time_batch)
        diffusion = g
        # x = x + drift * dt + diffusion * torch.sqrt(-dt) * torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(-dt, device=x.device)) * torch.randn_like(x)

    return x.cpu().numpy()

# Generate samples
samples = sample_sde(model)

# Plot generated samples
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.title("2D Samples from VE-SDE Diffusion (U-Net)")
plt.grid(True)
plt.show()
plt.savefig("fig/sde.png")
