using Functors, Flux, BSON, Dates, JLD2, Random, Statistics, JSON, DataFrames, Distributions, CSV, FileIO, Optimisers, Printf, ProgressMeter
using Flux: chunk, params, DataLoader, crossentropy
using Parameters: @with_kw, @unpack
using Logging: with_logger
using ProgressMeter: Progress, next!
import NNlib: batched_mul
using CUDA

include("lib/utils.jl")
include("lib/diffusion.jl")
include("lib/nnlib.jl")

args = @env begin
    activation=Flux.swish
    batchsize = 128
    d_hid = 16
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 1000
    fourier_scale=30.0f0
    has_node_outliers = true  # node outlier setting
    hidden_dim = 128  # hiddensize factor
    hidden_dims = [64, 128]
    input_dim = 2
    save_path = "data/main-2d.bson"
    load_path = ""
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-3  # learning rate
    min_depth = 2  # minimum depth of ancestors for the target node
    n_anomaly_nodes = 2
    n_anomaly_samples = 100  # n faulty observations
    n_batch = 10_000
    n_layers = 3
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_root_nodes = 1  # n_root_nodes
    n_samples = 3000  # n observations
    n_timesteps = 100
    noise_scale = 1.0
    output_dim = 2
    perturbed_scale = 1f0
    seed = 1  #  random seed
    to_device = Flux.gpu
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
end

include("models/embed.jl")
include("models/ConditionalChain.jl")
include("models/blocks.jl")
include("models/attention.jl")
include("models/batched_mul_4d.jl")
include("models/UNetFixed.jl")
include("models/UNetConditioned.jl")

function make_spiral(rng::AbstractRNG, n_samples::Int=1000)
    t_min = 1.5π
    t_max = 4.5π
    t = rand(rng, n_samples) * (t_max - t_min) .+ t_min
    x = t .* cos.(t)
    y = t .* sin.(t)
    [x y]'
end

make_spiral(n_samples::Int=1000) = make_spiral(Random.GLOBAL_RNG, n_samples)

function normalize_zero_to_one(x)
    x_min, x_max = extrema(x)
    x_norm = (x .- x_min) ./ (x_max - x_min)
    x_norm
end

function normalize_neg_one_to_one(x)
    2 * normalize_zero_to_one(x) .- 1
end

function spirals_2d()
    X = normalize_neg_one_to_one(make_spiral(n_batch))
    X_val = normalize_neg_one_to_one(make_spiral(floor(Int, 0.1 * n_batch)))
    loader = Flux.DataLoader((X,) |> gpu; batchsize=32, shuffle=true);
    val_loader = Flux.DataLoader((X_val,) |> gpu; batchsize=32, shuffle=false);
    (x,) = @> loader first gpu
    d = size(x, 1)
end

struct Diffusion2d{T}
    σ_max::Float32
    score_net::T
end
@functor Diffusion2d
# @showfields Diffusion2d
Optimisers.trainable(diffusion_model::Diffusion2d) = (; diffusion_model.score_net)

function Diffusion2d(; args)
    @assert args.input_dim == 2
    D, H, fourier_scale = (2, args.hidden_dim, args.fourier_scale)
    score_net = ConditionalChain(
                             Parallel(.+, Dense(2, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Dense(H, 2),
                            )
    return Diffusion2d(args.σ_max, score_net)
end

function (diffusion_model::Diffusion2d)(x::AbstractMatrix{T}, t) where {T}
    h = diffusion_model.score_net(x, t)
    σ_t = expand_dims(marginal_prob_std(t; diffusion_model.σ_max), 1)
    h ./ σ_t
end

function sm_loss(diffusion_model, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max=25f0)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    x̃ = x + z .* σ_t
    score = diffusion_model(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

diffusion_model = @> Diffusion2d(; args) gpu
args
X = @> make_spiral() f32
n = size(X, 2)
loader = Flux.DataLoader((X,) |> gpu; batchsize=128, shuffle=true);
(x,) = @> loader first gpu
D = size(x, 1)
sm_loss(diffusion_model, x)  # check clip_var
x

opt = Flux.setup(Optimisers.Adam(args.lr_unet), diffusion_model);
loss, (grad,) = Flux.withgradient(diffusion_model, ) do diffusion_model
    sm_loss(diffusion_model, x)
end
Flux.update!(opt, diffusion_model, grad);

opt = Flux.setup(Optimisers.Adam(args.lr_unet), diffusion_model);
progress = Progress(args.epochs, desc="Fitting diffusion_model");
for epoch = 1:args.epochs
    total_loss = 0.0
    for (x,) = loader
        @≥ x gpu
        global loss, (grad,) = Flux.withgradient(diffusion_model, ) do diffusion_model
            sm_loss(diffusion_model, x)
        end
        grad
        Flux.update!(opt, diffusion_model, grad)
        total_loss += loss
    end
    # @show total_loss/n
    next!(progress; showvalues=[(:loss, total_loss/length(loader))])
end
# @≥ X, diffusion_model cpu.();
# BSON.@save "data/main-2d.bson" args X diffusion_model

using PyPlot, CUDA  # assume CUDA.jl for gpu support

# Define plot limits
xlims = (-15, 15)
ylims = (-15, 15)

# ---------------------------
# 1. Plot Original Data
# ---------------------------
# Get a CPU copy of the data
X_val = X |> cpu  # X is assumed to be 2×N
# Extract rows as x and y data (eachrow returns an iterator; here we use indexing)
x_data = X_val[1, :]
y_data = X_val[2, :]

# Create a 2×2 grid figure
fig, axs = subplots(2, 2, figsize=(12, 10))

# First subplot: Original Data
ax1 = axs[1, 1]
ax1.set_title("Data (x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xlim(xlims)
ax1.set_ylim(ylims)
ax1.scatter(x_data, y_data, s=10, alpha=0.5)

# ---------------------------
# 2. Plot Perturbed Data
# ---------------------------
# Move X_val to GPU (if needed)
x_gpu = X_val |> gpu
# Generate random time t for each sample
t = rand(Float32, size(x_gpu, 2)) .* (1f0 - 1f-5) .+ 1f-5  # vector of size N
# Compute σ_t using your marginal probability function (assumed defined)
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1) |> gpu
# Add noise (using same shape as x_gpu)
z = randn(Float32, size(x_gpu)) |> gpu
# Perturb data: x̃ = x + σ_t .* noise
x_tilde = x_gpu .+ σ_t .* z
# Bring perturbed data back to CPU
x_tilde_cpu = cpu(x_tilde)
x_data2 = x_tilde_cpu[1, :]
y_data2 = x_tilde_cpu[2, :]

# Second subplot: Perturbed Data
ax2 = axs[1, 2]
ax2.set_title("Perturbed score matching (x, y)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(xlims)
ax2.set_ylim(ylims)
ax2.scatter(x_data2, y_data2, s=10, alpha=0.5)

# ---------------------------
# 3. Plot Gradients
# ---------------------------
# Create a grid of points over the plot limits

using PythonCall

function meshgrid(x, y)
    X = repeat(reshape(collect(x), 1, length(x)), length(y), 1)
    Y = repeat(reshape(collect(y), length(y), 1), 1, length(x))
    return X, Y
end

x_grid = range(xlims[1], stop=xlims[2], length=20)
y_grid = range(ylims[1], stop=ylims[2], length=20)
X_grid, Y_grid = meshgrid(x_grid, y_grid)
# Flatten the grid into 2×(20*20) array (each column is a point)
xy = vcat(vec(X_grid)', vec(Y_grid)')
# Convert to Float32 and move to GPU
xy_gpu = gpu(Float32.(xy))
# Create a constant time vector (e.g. 0.001 for all grid points)
t_grid = fill(0.001f0, size(xy_gpu, 2)) |> gpu
# (Optional) Compute σ_t on the grid if needed:
σ_t_grid = expand_dims(marginal_prob_std(t_grid; args.σ_max), 1)
# Evaluate your diffusion model at these grid points (multiplied by a factor, e.g. 0.2)
J = 0.2f0 .* diffusion_model(xy_gpu, t_grid) |> cpu
# Extract grid point coordinates and gradient components
x_grid = xy[1, :]
y_grid = xy[2, :]
u = J[1, :]  # gradient in x-direction
v = J[2, :]  # gradient in y-direction

# Third subplot: Gradients (using quiver)
ax3 = axs[2, 1]
ax3.set_title("Gradients")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_xlim(xlims)
ax3.set_ylim(ylims)
ax3.quiver(x_grid, y_grid, u, v)

# ---------------------------
# 4. Plot Stream Plot
# ---------------------------
# For a stream plot we need to reshape the gradients back to a grid shape.
U = reshape(u, size(X_grid))
V = reshape(v, size(Y_grid))
ax4 = axs[2, 2]
ax4.set_title("Stream Plot")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_xlim(xlims)
ax4.set_ylim(ylims)
ax4.streamplot(X_grid, Y_grid, U, V)

# Adjust layout and display/save the figure
tight_layout()
savefig("fig/combined-spiral-2d.png")

