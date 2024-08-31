using MLDatasets
using Flux
using Flux: @functor, chunk, params
using Flux.Data: DataLoader
using Parameters: @with_kw
using BSON
using CUDA
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using Statistics
using Optimisers: Optimisers, trainable
using Dates
using JLD2

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

