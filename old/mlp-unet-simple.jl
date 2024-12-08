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
using Optimisers: Optimisers
using Dates
using JLD2

struct MlpUnet{T}
    σ_max::Float32
    layers::NamedTuple
end

@functor MlpUnet
@showfields MlpUnet
Optimisers.trainable(unet::MlpUnet) = (; unet.layers)

function (unet::MlpUnet)(x::AbstractArray{T,N}, t) where {T,N}
    unet.mlp(x)
end

function MlpUnet(; input_dim::Int64, hidden_dims=[50, 10], embed_dim=50, fourier_scale=30.0f0, activation=swish, σ_max=25f0)
    Chain(
          Dense(2, 50, relu),
          SkipConnection(Dense(50, 50, relu), +),
          Dense(50, 2),
         )

    I, A, B, E = (input_dim, hidden_dims..., embed_dim)
    return MlpUnet(σ_max,
                   (embed = Chain(
                                  RandomFourierFeatures(embed_dim, fourier_scale),
                                  Dense(embed_dim, embed_dim, activation),
                                 ),
                    e1  = Dense(I, A),
                    e2  = Dense(A, B),
                    d2  = Dense(B, A),
                    d1  = Dense(2A, I),
                    _e1 = Dense(E, A),
                    _e2 = Dense(E, B),
                    _d2 = Dense(E, A),
                    g1  = InstanceNorm(A, activation),
                    g2  = InstanceNorm(B, activation),
                    _g2 = InstanceNorm(A, activation),
                   ))
end

function (unet::MlpUnet)(x::AbstractArray{T,N}, t) where {T,N}
    @unpack embed, e1, e2, d2, d1, _e1, _e2, _d2, g1, g2, _g2 = unet.layers
    emb = embed(t);
    h1 = @> e1(x)  .+ _e1(emb) g1;
    h2 = @> e2(h1) .+ _e2(emb) g2;
    h = @> d2(h2) _g2;
    h = @> d1(cat(h, h1; dims=1))
    σ_t = expand_dims(marginal_prob_std(t; unet.σ_max), N-1)
    h ./ σ_t
end

