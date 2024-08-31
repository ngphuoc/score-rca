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

function MlpUnet(; input_dim::Int64, hidden_dims=[50, 10], embed_dim=50, fourier_scale=30.0f0, activation=swish, σ_max=25f0)
    # I, A, B, C, D, E = (input_dim, hidden_dims..., embed_dim)
    I, A, B, E = (input_dim, hidden_dims..., embed_dim)
    return MlpUnet(σ_max,
                   (embed = Chain(
                                  RandomFourierFeatures(embed_dim, fourier_scale),
                                  Dense(embed_dim, embed_dim, activation),
                                 ),
                    e1  = Dense(I, A),
                    e2  = Dense(A, B),
                    # e3  = Dense(B, C),
                    # e4  = Dense(C, D),
                    # d4  = Dense(D, C),
                    # d3  = Dense(2C, B),
                    # d2  = Dense(2B, A),
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
    @unpack embed, e1, e2, e3, e4, d4, d3, d2, d1, _e1, _e2, _e3, _e4, _d4, _d3, _d2, g1, g2, g3, g4, _g4, _g3, _g2 = unet.layers
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

