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
using Printf
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
include("models/UNet.jl")
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
    epochs = 300
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


@info "Model"

struct Diffusion2d{T}
    σ_max::Float32
    model::T
end

function RoundTimesteps(max_timesteps::Int)
    function round_timesteps(t::AbstractArray{T,N}) where {T<:Real,N}
        round.(Int, max_timesteps .* t)
    end
end

@functor Diffusion2d
# @showfields Diffusion2d
Optimisers.trainable(unet::Diffusion2d) = (; unet.model)

function Diffusion2d(; args)
    @assert args.input_dim == 2
    X, H, E, n_timesteps = (2, args.hidden_dim, args.embed_dim, args.n_timesteps)
    model = ConditionalChain(
                             Parallel(.+, Dense(2, H), Chain(RoundTimesteps(n_timesteps), SinusoidalPositionEmbedding(n_timesteps, H), Dense(H, H))),
                             swish,
                             Parallel(.+, Dense(H, H), Chain(RoundTimesteps(n_timesteps), SinusoidalPositionEmbedding(n_timesteps, H), Dense(H, H))),
                             swish,
                             Parallel(.+, Dense(H, H), Chain(RoundTimesteps(n_timesteps), SinusoidalPositionEmbedding(n_timesteps, H), Dense(H, H))),
                             swish,
                             Dense(H, 2),
                            )
    return Diffusion2d(σ_max, model)
end

function (unet::Diffusion2d)(x::AbstractMatrix{T}, t) where {T}
    h = unet.model(x, t)
    σ_t = expand_dims(marginal_prob_std(t; unet.σ_max), 1)
    h ./ σ_t
end

unet = @> Diffusion2d(; args) gpu
score_matching_loss(unet, x)  # check clip_var

opt = Flux.setup(Optimisers.Adam(args.lr_unet), unet);
loss, (grad,) = Flux.withgradient(unet, ) do unet
    score_matching_loss(unet, x)
end
Flux.update!(opt, unet, grad);

opt = Flux.setup(Optimisers.Adam(args.lr_unet), unet);
progress = Progress(args.epochs, desc="Fitting unet");
for epoch = 1:args.epochs
    total_loss = 0.0
    for (x,) = loader
        @≥ x gpu
        loss, (grad,) = Flux.withgradient(unet, ) do unet
            score_matching_loss(unet, x)
        end
        grad
        Flux.update!(opt, unet, grad)
        total_loss += loss
    end
    next!(progress; showvalues=[(:loss, total_loss/length(loader))])
end

