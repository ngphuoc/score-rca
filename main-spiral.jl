using Functors, Flux, BSON, Dates, JLD2, Random, Statistics, JSON, DataFrames, Distributions, CSV, FileIO, Optimisers, Printf, ProgressMeter
using Flux: chunk, params, DataLoader, crossentropy
using Parameters: @with_kw, @unpack
using Logging: with_logger
using ProgressMeter: Progress, next!
import NNlib: batched_mul

include("lib/utils.jl")
include("lib/diffusion.jl")
include("lib/nnlib.jl")
include("lib/plot.jl")

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
loader = Flux.DataLoader((X,) |> gpu; batchsize=32, shuffle=true);
(x,) = @> loader first gpu
D = size(x, 1)
sm_loss(diffusion_model, x)  # check clip_var

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


@info "Plots"
using LaTeXStrings
using CairoMakie

# Define the plot limits
xlim = ylim = (-15, 15)

# Plot data
# X_val = X[:, 1:10:end] |> cpu
X_val = X |> cpu
x, y = eachrow(X_val)

# Create the combined figure
combined_fig = Figure()

# Add the first subplot
ax1 = Axis(combined_fig[1, 1], title=L"Data $(x, y)$", xlabel=L"x", ylabel=L"y", limits=(xlim, ylim))
scatter!(ax1, x, y)
combined_fig

# Plot perturbed data
x = @> X_val gpu;
t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
z = randn!(similar(x));  # Normal
x̃ = x .+ σ_t .* z
x, y = eachrow(cpu(x̃))

# Add the second subplot
ax2 = Axis(combined_fig[1, 2], title=L"Perturbed score matching $(x,y)$", xlabel=L"x", ylabel=L"y", limits=(xlim, ylim))
scatter!(ax2, x, y)

combined_fig

# Plot gradients

x = @> Iterators.product(range(xlim..., length=20), range(ylim..., length=20)) collect vec;
x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
t = fill!(similar(x, size(x)[end]), 0.001) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
# t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
J = @> 0.2diffusion_model(x, t)
@≥ J, x cpu.()

x, y = eachrow(x);
u, v = eachrow(J);

# Add the third subplot
ax3 = Axis(combined_fig[2, 1], title="Gradients", xlabel=L"x", ylabel=L"y", limits=(xlim, ylim))
@which quiver!(ax3, x, y, u, v)

# Add the stream plot

@≥ x, y, u, v reshape.(20, 20)
ax4 = Axis(combined_fig[2, 2], title="Stream Plot", xlabel="x", ylabel="y")
splot!(ax4, u, v)

splot(u, v)

# Display the combined plot
combined_fig

# Save the combined figure
save("fig/combined-spiral-2d.png", combined_fig)

