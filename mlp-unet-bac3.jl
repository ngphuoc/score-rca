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
                    e1  = Dense(I, A),
                    e2  = Dense(A, B),
                    d2  = Dense(B, A),
                    d1  = Dense(2A, I),
                    _e1 = Dense(E, A),
                    _e2 = Dense(E, B),
                    # _e3 = Dense(E, C),
                    # _e4 = Dense(E, D),
                    # _d4 = Dense(E, C),
                    # _d3 = Dense(E, B),
                    _d2 = Dense(E, A),
                    g1  = InstanceNorm(A, activation),
                    g2  = InstanceNorm(B, activation),
                    # g3  = InstanceNorm(C, activation),
                    # g4  = InstanceNorm(D, activation),
                    # _g4 = InstanceNorm(C, activation),
                    # _g3 = InstanceNorm(B, activation),
                    _g2 = InstanceNorm(A, activation),
                   ))
end

function (unet::MlpUnet)(x::AbstractArray{T,N}, t) where {T,N}
    @unpack embed, e1, e2, d2, d1, _e1, _e2, _d2, g1, g2, _g2 = unet.layers
    #-- Embedding
    emb = embed(t);
    #-- Encoder
    # @show size(x) size(emb) size(e1(x)) size(_e1(emb))
    h1 = @> e1(x)  .+ _e1(emb) g1;
    h2 = @> e2(h1) .+ _e2(emb) g2;
    # h3 = @> e3(h2) .+ _e3(emb) g3;
    # h4 = @> e4(h3) .+ _e4(emb) g4;
    #-- Decoder
    # h = @> d4(h4)                 .+ _d4(emb) _g4;
    # h = @> d3(cat(h, h3; dims=1)) .+ _d3(emb) _g3;
    # h = @> d2(cat(h, h2; dims=1)) .+ _d2(emb) _g2;
    h = @> d2(h2) _g2;
    h = @> d1(cat(h, h1; dims=1))
    #-- Scaling Factor
    σ_t = expand_dims(marginal_prob_std(t; unet.σ_max), N-1)
    h ./ σ_t
end

# struct MlpUnet{T}
#     σ_max::Float32
#     layers::T
# end

# @functor MlpUnet
# @showfields MlpUnet
# Optimisers.trainable(unet::MlpUnet) = (; unet.layers)

# function MlpUnet(; args)
#     X, A, B, E = (args.input_dim, args.hidden_dims..., args.embed_dim)
#     @unpack activation, fourier_scale = args
#     return MlpUnet(σ_max, (
#                            e1 = Dense(X, A, activation),
#                            d1 = Dense(A, X)
#                           ))
# end

# function (unet::MlpUnet)(x::AbstractMatrix{T}, t::AbstractVector{T}) where {T}
#     @unpack e1, d1 = unet.layers
#     h1 = @> e1(x)
#     h = @> d1(h1)
#     σ_t = @> marginal_prob_std(t; unet.σ_max) unsqueeze(1)
#     h ./ σ_t
# end

# struct MlpUnet
#     σ_max::Float32
#     rff
#     embed
#     e1
#     e2
#     d2
#     d1
#     _e1
#     _e2
#     _d2
#     g1
#     g2
#     _g2
# end

# @functor MlpUnet
# @showfields MlpUnet
# Optimisers.trainable(unet::MlpUnet) = (; unet.embed, unet.e1, unet.e2, unet.d2, unet.d1, unet._e1, unet._e2, unet._d2, unet.g1, unet.g2, unet._g2)

# function MlpUnet(; args)
#     X, A, B, E = (args.input_dim, args.hidden_dims..., args.embed_dim)
#     @unpack activation, fourier_scale = args
#     rff = RandomFourierFeatures(E, fourier_scale)
#     embed = Dense(E, E, activation)
#     e1    = Dense(X, A)
#     e2    = Dense(A, B)
#     d2    = Dense(B, A)
#     d1    = Dense(2A, X)
#     _e1   = Dense(E, A)
#     _e2   = Dense(E, B)
#     _d2   = Dense(E, A)
#     g1    = InstanceNorm(A, activation)
#     g2    = InstanceNorm(B, activation)
#     _g2   = InstanceNorm(A, activation)
#     return MlpUnet(σ_max, rff, embed, e1, e2, d2, d1, _e1, _e2, _d2, g1, g2, _g2)
# end

# function (unet::MlpUnet)(x::AbstractMatrix{T}, t::AbstractVector{T}) where {T}
#     @unpack σ_max, rff, embed, e1, e2, d2, d1, _e1, _e2, _d2, g1, g2, _g2 = unet
#     emb = embed(rff(t));
#     h1 = @> e1(x)  .+ _e1(emb) g1;
#     h2 = @> e2(h1) .+ _e2(emb) g2;
#     h = @> d2(h2) _g2;
#     h = @> d1(cat(h, h1; dims=1))
#     σ_t = @> marginal_prob_std(t; unet.σ_max) unsqueeze(1)
#     h ./ σ_t
# end


