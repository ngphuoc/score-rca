using MLDatasets
using Flux
using Flux: @functor, chunk, params, InstanceNorm
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

struct GroupMlpRegression{M,T}
    μx::AbstractVector{T}  # for normalisation of observations
    σx::AbstractVector{T}
    paj_mask::AbstractMatrix{Bool}
    mlp::M
end

@functor GroupMlpRegression
Optimisers.trainable(mlp::GroupMlpRegression) = (; mlp.mlp)  # no trainable parameters

@showfields GroupMlpRegression

function GroupMlpRegression(paj_mask::Matrix{Bool}; x, μx=mean(x, dims=2)', σx=std(x, dims=2)', hidden_dims=[100, ], activation=Flux.σ)
    F = I = input_dim = size(paj_mask, 1)
    H = hidden_dims
    return GroupMlpRegression(μx, σx,
                              paj_mask,
                              Chain(
                                    GroupDense(I, H[1], F),
                                    [GroupDense(H[i], H[i+1], F, activation) for i=1:length(H)-1]...,
                                    GroupDense(H[end], 1, F),
                                   ))
end

function (mlp::GroupMlpRegression)(x::AbstractArray{T, 3}) where T
    paj_mask = mlp.paj_mask
    x0 = @. (x - μx) / σx * paj_mask
    x̂ = mlp.mlp(x0)
    x̂ * σx + μx
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

function GroupMlpUnet(paj_mask::Matrix{Bool}, scale=30.0f0; hidden_dims=[7, 5], activation=relu)
    X = input_dim = size(paj_mask, 1)
    E = F = X  # the first node j has only marginal prob, mask for conditional is zeros
    A, B, C, D, E = 2X, X, X÷2+1, X÷4+1, X
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
                               e1 = GroupDense(X, A, F),
                               e2 = GroupDense(A, B, F),
                               e3 = GroupDense(B, C, F),
                               e4 = GroupDense(C, D, F),
                               d4 = GroupDense(D, C, F),
                               d3 = GroupDense(2C, B, F),
                               d2 = GroupDense(2B, A, F),
                               d1 = GroupDense(2A, X, F),
                               c1 = GroupDense(E, A, F),
                               c2 = GroupDense(E, B, F),
                               c3 = GroupDense(E, C, F),
                               c4 = GroupDense(E, D, F),
                               c5 = GroupDense(E, C, F),
                               c6 = GroupDense(E, B, F),
                               c7 = GroupDense(E, A, F),
                               g1 = InstanceNorm(F, swish),
                               g2 = InstanceNorm(F, swish),
                               g3 = InstanceNorm(F, swish),
                               g4 = InstanceNorm(F, swish),
                               g5 = InstanceNorm(F, swish),
                               g6 = InstanceNorm(F, swish),
                               g7 = InstanceNorm(F, swish),
                              ))
end

"""
One t_j for each x_j, x and t have the same size
Parallel by groups
"""
function (unet::GroupMlpUnet)(x::AbstractArray{T, 3}, t::AbstractArray{T, 3}) where T
    @unpack e1, e2, e3, e4, d4, d3, d2, d1, c1, c2, c3, c4, c5, c6, c7, g1, g2, g3, g4, g5, g6, g7 = unet.layers
    mask = unet.mask
    embed = unet.layers.embed(t)
    x0 = x .* mask
    t0 = embed .* mask
    #-- Encoder
    h1 = @> e1(x0) .+ c1(t0) g1
    h2 = @> e2(h1) .+ c2(t0) g2
    h3 = @> e3(h2) .+ c3(t0) g3
    h4 = @> e4(h3) .+ c4(t0) g4
    #-- Decoder
    h = @> d4(h4) .+ c5(t0) g5
    h = @> d3(vcat(h, h3)) .+ c6(t0) g6
    h = @> d2(vcat(h, h2)) .+ c7(t0) g7
    h = @> d1(vcat(h, h1))
    #-- Scaling Factor
    σ_t = marginal_prob_std(t)
    mask .* h ./ σ_t
end

