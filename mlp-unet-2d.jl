using Flux
using Flux: @functor, chunk, params
using Flux: DataLoader
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

struct MlpUnet
    σ_max::Float32
    layers::NamedTuple
end

@functor MlpUnet
@showfields MlpUnet
Optimisers.trainable(unet::MlpUnet) = (; unet.layers)

function MlpUnet(; args)
    I, A, B, E = (args.input_dim, args.hidden_dims..., args.embed_dim)
    return MlpUnet(σ_max,
                   (embed = Chain(
                                  RandomFourierFeatures(embed_dim, fourier_scale),
                                  Dense(embed_dim, embed_dim, activation),
                                 ),
                    e1  = Dense(I, A, activation),
                    e2  = Dense(A, B, activation),
                    d2  = Dense(B, A, activation),
                    d1  = Dense(2A, I),
                    _e1 = Dense(E, A, activation),
                    _e2 = Dense(E, B, activation),
                    _d2 = Dense(E, A, activation),
                   ))
end

function (unet::MlpUnet)(x::AbstractArray{T,N}, t) where {T,N}
    @unpack embed, e1, e2, d2, d1, _e1, _e2, _d2 = unet.layers
    emb = embed(t)
    h1 = @> e1(x)  .+ _e1(emb)
    h2 = @> e2(h1) .+ _e2(emb);
    h = @> d2(h2) .+ _d2(emb)
    h = @> d1(vcat(h, h1))
    σ_t = expand_dims(marginal_prob_std(t; unet.σ_max), N-1)
    h ./ σ_t
end

