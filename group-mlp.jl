using MLDatasets
using Flux
using Flux: @functor, chunk, params
using Flux.Data: DataLoader
using Parameters: @with_kw, @unpack
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
include("lib/nn.jl")
include("lib/nnlib.jl")

#-- MLP Regression

struct GroupMlpRegression{T}
    mask::AbstractMatrix{Bool}
    mlp::T
end

@functor GroupMlpRegression
Optimisers.trainable(mlp::GroupMlpRegression) = (; mlp.mlp)  # no trainable parameters

@showfields GroupMlpRegression

function GroupMlpRegression(adjmat::AbstractMatrix{T}; hidden_dims=[5, 3], activation=relu) where {T}
    mask = @> adjmat Matrix{Bool}
    F = X = input_dim = size(adjmat, 1)
    H = hidden_dims
    return GroupMlpRegression(mask, Chain(
                                          GroupDense(X, H[1], F, activation), # InstanceNorm(F, relu),
                                          [GroupDense(H[i], H[i+1], F, activation) for i=1:length(H)-1]...,
                                          GroupDense(H[end], 1, F),
                                         ))
end

function (mlp::GroupMlpRegression)(x::AbstractArray{T, 3}) where T
    mask = mlp.mask
    x0 = x .* mask
    x̂ = mlp.mlp(x0)
end

#-- MLP UNet

struct GroupMlpUnet
    mask::AbstractMatrix{Bool}
    layers::NamedTuple
end

@functor GroupMlpUnet
@showfields GroupMlpUnet
Optimisers.trainable(unet::GroupMlpUnet) = (; unet.layers)  # no trainable parameters

function scalar_zero(::AbstractArray{T,N}) where {T,N}
    0f0
end

function GroupMlpUnet(adjmat::AbstractMatrix{T}, scale=30.0f0; hidden_dims=[7, 5], activation=relu) where {T}
    X = input_dim = size(adjmat, 1)
    E = F = X  # the first node j has only marginal prob, mask for conditional is zeros
    H = [X, hidden_dims...]
    paj_mask = adjmat
    j_mask = I(F)
    @≥ j_mask, paj_mask Matrix{Bool}.()
    mask = max.(j_mask, paj_mask)
    return GroupMlpUnet(mask, (
                               embed = Chain(
                                             vec,
                                             RandomFourierFeatures(2E, scale),
                                             Reshape(2E, F, :),
                                             GroupDense(2E, E, F, swish),
                                            ),
                               #-- encoding
                               enc = [GroupDense(H[i-1], H[i], F) for i=2:length(H)],
                               enc_cond = [GroupDense(E, H[i], F) for i=2:length(H)],
                               enc_norm = [InstanceNorm(F, swish) for i=2:length(H)],
                               #-- decoding
                               dec = [GroupDense(2H[i], H[i-1], F) for i=length(H):-1:2],
                               dec_cond = [[GroupDense(E, H[i-1], F) for i=length(H):-1:3]..., scalar_zero],
                               dec_norm = [[InstanceNorm(F, swish) for i=length(H):-1:3]..., identity],
                              ))
end

"""
One t_j for each x_j, x and t have the same size
Parallel by groups
"""
function (unet::GroupMlpUnet)(x::AbstractArray{T, 3}, t::AbstractArray{T, 3}) where T
    @unpack enc, enc_cond, enc_norm, dec, dec_cond, dec_norm = unet.layers
    mask = unet.mask
    embed = unet.layers.embed(t)
    h = x .* mask
    c = embed .* mask
    hs = typeof(x)[]
    for i=1:length(enc)
        h = @> enc[i](h) .+ enc_cond[i](c) enc_norm[i]
        push!(hs, h)
    end
    hs = reverse(hs)
    for i=1:length(dec)
        h = vcat(h, hs[i])
        h = @> dec[i](h) .+ dec_cond[i](c) dec_norm[i]
    end
    σ_t = marginal_prob_std(t)
    mask .* h ./ σ_t
end

