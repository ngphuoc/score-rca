using Flux
using Flux: crossentropy
using Flux.Data: DataLoader
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Plots
using Plots
using Flux
using Dates
using BSON, JSON
using Printf
import Base.show, Base.eltype
using Flux
import Flux._big_show, Flux._show_children
using ProgressMeter
using BSON
using Random
import NNlib: batched_mul
using Optimisers
using Optimisers: Optimisers, trainable

include("./lib/utils.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/graph.jl")
include("./lib/diffusion.jl")
include("./dlsm-losses.jl")
include("datasets.jl")
include("utilities.jl")
include("models/embed.jl")
include("models/ConditionalChain.jl")
include("models/blocks.jl")
include("models/attention.jl")
include("models/batched_mul_4d.jl")
include("models/UNetFixed.jl")
include("models/UNetConditioned.jl")

args = @env begin
    # Denoising
    input_dim = 2
    output_dim = 2
    # hidden_dims = [50, 10]
    hidden_dim = 32  # hiddensize factor
    n_layers = 3
    embed_dim = 50  # hiddensize factor
    activation=swish
    perturbed_scale = 1f0
    fourier_scale=30.0f0
    # scale=25.0f0  # RandomFourierFeatures scale
    # σ_max=25f0
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-4  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = Flux.gpu
    batchsize = 32
    epochs = 10
    save_path = ""
    load_path = "data/exp2d-joint.bson"
    # RCA
    min_depth = 2  # minimum depth of ancestors for the target node
    n_root_nodes = 1  # n_root_nodes
    n_timesteps = 40
    n_samples = 1000  # n observations
    n_anomaly_nodes = 2
    n_anomaly_samples = 100  # n faulty observations
    has_node_outliers = true  # node outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
    #-- settings
    n_batch = 10_000
    d_hid = 16
end


@info "Data"

X = normalize_neg_one_to_one(make_spiral(n_batch))
X_val = normalize_neg_one_to_one(make_spiral(floor(Int, 0.1 * n_batch)))
loader = Flux.DataLoader((X,) |> gpu; batchsize=32, shuffle=true);
val_loader = Flux.DataLoader((X_val,) |> gpu; batchsize=32, shuffle=false);
(x,) = @> loader first gpu
d = size(x, 1)


@info "layers"

struct Diffusion2d{T}
    σ_max::Float32
    layers::T
end

@functor Diffusion2d
# @showfields Diffusion2d
Optimisers.trainable(net::Diffusion2d) = (; net.layers)

function Diffusion2d(; args)
    @assert args.input_dim == 2
    X, H, E, fourier_scale = (2, args.hidden_dim, args.embed_dim, args.fourier_scale)
    return Diffusion2d(σ_max, (
                               f1 = Dense(2, H),
                               f2 = Dense(H, H),
                               f3 = Dense(H, H),
                               f4 = Dense(H, 2),
                               e1 = Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H)),
                               e2 = Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H)),
                               e3 = Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H)),
                              )
                      )
end

function (net::Diffusion2d)(x::AbstractMatrix{T}, t::AbstractVector{T}) where T
    @unpack f1, f2, f3, f4, e1, e2, e3 = net.layers
    h = x
    h = f1(h) + e1(t)
    h = f2(h) + e2(t)
    h = f3(h) + e3(t)
    h = f4(h)
    σ_t = expand_dims(marginal_prob_std(t; net.σ_max), 1)
    h ./ σ_t
end

function sm_loss(net, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max=25f0)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    # (batch) of perturbed 𝘹(𝘵)'s to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    x̃ = x + z .* σ_t
    score = net(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

net = @> Diffusion2d(; args) gpu

(x,) = loader |> first
batchsize = size(x)[end]
@≥ x gpu
sm_loss(net, x)

opt = Flux.setup(Optimisers.Adam(args.lr_unet), net);
loss, (grad,) = Flux.withgradient(net, ) do net
    sm_loss(net, x)
end
Flux.update!(opt, net, grad);

opt = Flux.setup(Optimisers.Adam(args.lr_unet), net);
progress = Progress(args.epochs, desc="Fitting net");
for epoch = 1:args.epochs
    total_loss = 0.0
    for (x,) = loader
        batchsize = size(x)[end]
        @≥ x gpu
        loss, (grad,) = Flux.withgradient(net, ) do net
            sm_loss(net, x)
        end
        Flux.update!(opt, net, grad)
        total_loss += loss
    end
    next!(progress; showvalues=[(:loss, total_loss/length(loader))])
end

