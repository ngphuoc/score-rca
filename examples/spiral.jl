using Random
using Plots
using Flux
using Statistics
include("lib/utils.jl")

# Function to generate a 2D spiral dataset with optional noise
function generate_spiral(n_points, noise_std=0.1)
    t = range(0, 4 * π, length=n_points)
    x = t .* cos.(t)
    y = t .* sin.(t)
    spiral = hcat(x, y)
    noise = noise_std * randn(size(spiral))
    return spiral .+ noise
end

# Generate and visualize the dataset
n_points = 1000
spiral_data = generate_spiral(n_points)
scatter(spiral_data[:, 1], spiral_data[:, 2], markersize=2, title="Noisy Spiral Dataset", aspect_ratio=:equal)

# Define the LinearSDE struct
struct LinearSDE
    beta_min::Float32
    beta_max::Float32
    T::Float32
end

function LinearSDE(; beta_min=0.1, beta_max=20.0, T=1.0)
    return LinearSDE(beta_min, beta_max, T)
end

function noise_schedule(sde::LinearSDE, t)
    return sde.beta_min .+ t .* (sde.beta_max - sde.beta_min)
end

function forward_process(sde::LinearSDE, x0, t)
    alpha_t = exp.(-0.5f0 * noise_schedule(sde, t) .* t)
    noise = randn(size(x0))
    xt = @. alpha_t * x0 + sqrt(1 - alpha_t^2) * noise
    return xt, noise
end

# Define the TimeEmbedding unet
struct TimeEmbedding
    fc::Chain
end

function TimeEmbedding(time_dim)
    fc = Chain(
        Dense(1, time_dim, relu),
        Dense(time_dim, time_dim)
    )
    return TimeEmbedding(fc)
end

function (te::TimeEmbedding)(t)
    return te.fc(t)
end

# Define the UNet unet
struct UNet
    time_embed::TimeEmbedding
    encoder::Chain
    bottleneck::Chain
    decoder::Chain
end

function UNet(input_dim, base_channels, time_dim)
    time_embed = TimeEmbedding(time_dim)
    
    encoder = Chain(
        Conv((3,), input_dim => base_channels, pad=1),
        relu,
        Conv((3,), base_channels => base_channels * 2, stride=2, pad=1),
        relu
    )
    
    bottleneck = Chain(
        Conv((3,), base_channels * 2 => base_channels * 2, pad=1),
        relu
    )
    
    decoder = Chain(
        ConvTranspose((4,), base_channels * 2 => base_channels, stride=2, pad=1),
        relu,
        Conv((3,), base_channels => input_dim, pad=1)
    )
    
    return UNet(time_embed, encoder, bottleneck, decoder)
end

function (unet::UNet)(x, t)
    t_emb = unet.time_embed(t)
    t_emb = reshape(t_emb, :, 1)  # Unsqueeze the time embedding
    x = x .+ t_emb
    x = unet.encoder(x)
    x = unet.bottleneck(x)
    x = unet.decoder(x)
    return x
end

# Hyperparameters
input_dim = 2
base_channels = 64
time_dim = 16
epochs = 500
batch_size = 64
lr = 1e-3

# unet, optimizer, and loss
unet = UNet(input_dim, base_channels, time_dim)
opt = ADAM(lr)
loss(x, y) = mean((unet(x, y) .- y).^2)

# Dataset preparation
spiral_tensor = Flux.DataLoader((spiral_data',), batchsize=batch_size, shuffle=true)

# Training
sde = LinearSDE()
for epoch = 1:epochs
    # unet.train()
    epoch_loss = 0.0
    for batch = spiral_tensor
        x0 = batch[1]
        # x0 = permutedims(x0, (2, 1))  # Shape (batch_size, input_dim, 1)
        t = rand(Float32, 1, size(x0, 2)) * sde.T
        xt, noise = forward_process(sde, x0, t)
        gs = Flux.gradient(unet) do unet
            predicted_noise = unet(xt, t)
            loss(predicted_noise, noise)
        end
        Flux.Optimise.update!(opt, params(unet), gs)

        epoch_loss += loss(unet(xt, t), noise)
    end
    println("Epoch $epoch/$epochs, Loss: $(epoch_loss / length(spiral_tensor))")
end

# Generate noisy data and denoise it
# unet.eval()
x0_sample = spiral_tensor.data[1][:, 1:100]  # Take a few samples
t_sample = fill(sde.T, 1, size(x0_sample, 2))  # Maximum noise level
xt, _ = forward_process(sde, x0_sample, t_sample)

denoised = unet(xt, t_sample)

# Plot noisy vs. denoised
xt_np = xt'
denoised_np = denoised'

scatter(xt_np[:, 1], xt_np[:, 2], markersize=2, title="Noisy Data", aspect_ratio=:equal)
scatter(denoised_np[:, 1], denoised_np[:, 2], markersize=2, title="Denoised Data", aspect_ratio=:equal)
