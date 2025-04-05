import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tqdm import tqdm

# === Settings ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512
epochs = 5000
lr = 1e-3
sigma_min = 0.01
sigma_max = 50.0

# === Moons Dataset ===
def sample_data(n_samples=512):
    data, _ = make_moons(n_samples=n_samples, noise=0.1)
    return torch.tensor(data, dtype=torch.float32, device=device)

# === Score Model (MLP) ===
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        t_proj = torch.log(sigma_min * (sigma_max / sigma_min)**t).unsqueeze(-1)
        t_proj = t_proj.expand_as(x[:, :1])
        input = torch.cat([x, t_proj], dim=-1)
        return self.net(input)

# === VE-SDE Class ===
class VESDE(torchsde.SDEIto):
    def __init__(self, sigma_min, sigma_max):
        super().__init__(noise_type='diagonal')
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min)**t

    def f(self, t, y):
        return torch.zeros_like(y)

    def g(self, t, y):
        sigma_t = self.sigma(t)
        sigma_min_tensor = torch.tensor(self.sigma_min, device=y.device)
        sigma_max_tensor = torch.tensor(self.sigma_max, device=y.device)
        constant = torch.sqrt(torch.tensor(2.0, device=y.device) *
                              (torch.log(sigma_max_tensor) - torch.log(sigma_min_tensor)))
        return sigma_t * constant

# === Instantiate Model and Optimizer ===
model = ScoreNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
sde = VESDE(sigma_min, sigma_max)

# === Training Loop ===
for epoch in tqdm(range(epochs), desc="Training"):
    optimizer.zero_grad()
    x0 = sample_data(batch_size)
    t = torch.rand(batch_size, device=device)
    sigma_t = sde.sigma(t)
    noise = torch.randn_like(x0)
    xt = x0 + sigma_t[:, None] * noise
    target_score = -noise / sigma_t[:, None]
    predicted_score = model(xt, t)
    loss = ((predicted_score - target_score) ** 2).mean()
    loss.backward()
    optimizer.step()

# === Sampling from the model ===
@torch.no_grad()
def sample_vesde(model, n_samples=1000, steps=1000):
    x = torch.randn(n_samples, 2, device=device) * sigma_max
    dt = -1.0 / steps
    for step in tqdm(range(steps), desc="Sampling"):
        t_val = 1.0 - step / steps
        t = torch.full((n_samples,), t_val, device=device)
        g = sde.g(t, x).unsqueeze(-1)
        drift = -g**2 * model(x, t)
        diffusion = g
        x += drift * dt + diffusion * torch.sqrt(torch.tensor(-dt, device=x.device)) * torch.randn_like(x)
    return x.cpu().numpy()

# === Generate samples and plot ===
real_data = sample_data(1000).cpu().numpy()
samples = sample_vesde(model, n_samples=1000)
plt.figure(figsize=(7, 7))
plt.scatter(real_data[:, 0], real_data[:, 1], s=10, alpha=0.5, label='Real Data', color='blue')
plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5, label='Generated Samples', color='orange')
plt.legend()
plt.title("Real vs VE-SDE Generated Samples (Moons Dataset)")
plt.grid(True)
plt.axis("equal")
plt.savefig("fig/moon.png")

