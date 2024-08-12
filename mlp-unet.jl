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
    layers::NamedTuple
end

@functor MlpUnet

function MlpUnet(input_dim, hidden_dims=[2input_dim, input_dim, input_dim÷2+1, input_dim÷4+1], embed_dim=2input_dim, scale=30.0f0)
    I, A, B, C, D, E = (input_dim, hidden_dims..., embed_dim)
    return MlpUnet((
        gaussfourierproj = RandomFourierFeatures(embed_dim, scale),
        linear = Dense(embed_dim, embed_dim, swish),
        #-- Encoding
        e1 = Dense(I, A),
        e2 = Dense(A, B),
        e3 = Dense(B, C),
        e4 = Dense(C, D),
        #-- Decoding
        d4 = Dense(D, C),
        d3 = Dense(C, B),
        d2 = Dense(B, A),
        d1 = Dense(A, I),
        #-- Condition
        c1 = Dense(E, A),
        c2 = Dense(E, B),
        c3 = Dense(E, C),
        c4 = Dense(E, D),
        c5 = Dense(E, C),
        c6 = Dense(E, B),
        c7 = Dense(E, A),
        #-- GroupNorm
        g1 = GroupNorm(I, I, swish),
        g2 = GroupNorm(A, A, swish),
        g3 = GroupNorm(B, B, swish),
        g4 = GroupNorm(C, C, swish),
        g5 = GroupNorm(D, D, swish),
        g6 = GroupNorm(C, C, swish),
        g7 = GroupNorm(B, B, swish),
    ))
end

function (unet::MlpUnet)(x, t)
    #-- Embedding
    embed = unet.layers.linear(unet.layers.gaussfourierproj(t))
    #-- Encoder
    h1 = @> unet.layers.e1(x) .+ unet.layers.c1(embed) unet.layers.g1
    h2 = @> unet.layers.e2(h1) .+ unet.layers.c2(embed) unet.layers.g2
    h3 = @> unet.layers.e3(h2) .+ unet.layers.c3(embed) unet.layers.g3
    h4 = @> unet.layers.e4(h3) .+ unet.layers.c4(embed) unet.layers.g4
    #-- Decoder
    h = @> unet.layers.d4(h4) .+ unet.layers.c5(embed) unet.layers.g5
    h = @> unet.layers.d3(cat(h, h3; dims=1)) .+ unet.layers.c6(embed) unet.layers.g6
    h = @> unet.layers.d2(cat(h, h2; dims=1)) .+ unet.layers.c7(embed) unet.layers.g7
    h = @> unet.layers.d1(cat(h, h1; dims=1))
    #-- Scaling Factor
    h ./ marginal_prob_std(t)
end

