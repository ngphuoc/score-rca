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

#-- ORACLE Regression

struct GroupOracleRegression{M}
    paj_mask::AbstractMatrix{Bool}
    mlp::M
end

@functor GroupOracleRegression
Optimisers.trainable(oracle::GroupOracleRegression) = (; oracle.mlp)  # no trainable parameters

@showfields GroupOracleRegression

function GroupOracleRegression(dag)
    paj_mask = @> dag.dag adjacency_matrix Matrix{Bool}
    m = dag.cpds[2].mlp
    W1 = m[1].weight
    W2 = m[2].weight
    H = size(W1, 1)
    activation = m[1].σ
    X = F = size(paj_mask, 1)
    mlp = Chain(
          GroupDense(X, H, F, activation),
          GroupDense(H, 1, F),
         )
    # to check input output masks, or just set zeros
    mlp[1].weight[:, 1, 2] .= W1  # get :X1 input
    mlp[1].weight[:, 2, 2] .= 0f0
    mlp[2].weight[:, :, 2] .= W2  # get :X2 output
    # @≥ mlp gpu
    # x = @> rand(Float32, 2, 2, 1) gpu
    # @> mlp(x) flatend(2)
    GroupOracleRegression(paj_mask, mlp)
end

function (regressor::GroupOracleRegression)(x::AbstractArray{T, 3}) where T
    @unpack paj_mask, mlp = regressor
    x̂ = @> mlp(paj_mask .* x)
end

#-- MLP Regression

struct GroupMlpRegression{M}
    paj_mask::AbstractMatrix{Bool}
    mlp::M
end

@functor GroupMlpRegression
Optimisers.trainable(regressor::GroupMlpRegression) = (; regressor.mlp)  # no trainable parameters

@showfields GroupMlpRegression

function GroupMlpRegression(dag; hidden_dims=[100, ], activation=swish)
    paj_mask = @> dag.dag adjacency_matrix Matrix{Bool}
    F = I = input_dim = size(paj_mask, 1)
    H = hidden_dims
    return GroupMlpRegression(paj_mask,
                              Chain(
                                    GroupDense(I, H[1], F), InstanceNorm(F, activation),
                                    [Chain(
                                           GroupDense(H[i], H[i+1], F), InstanceNorm(F, activation),
                                          ) for i=1:length(H)-1]...,
                                    GroupDense(H[end], 1, F),
                                   ))
end

function (regressor::GroupMlpRegression)(x::AbstractArray{T, 3}) where T
    @unpack paj_mask, mlp = regressor
    x̂ = mlp(paj_mask .* x)
end

#-- Group MLP UNet

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

function GroupMlpUnet(dag; scale=30.0f0, hidden_dims=[50, 20, 10], activation=swish)
    paj_mask = @> dag.dag adjacency_matrix Matrix{Bool}
    I = input_dim = size(paj_mask, 1)
    F = I  # the first node j has only marginal prob, mask for conditional is zeros
    A, B, C, E = (hidden_dims..., hidden_dims[1])
    j_mask = LinearAlgebra.I(F)
    @≥ j_mask, paj_mask Matrix{Bool}.()
    mask = max.(j_mask, paj_mask)
    return GroupMlpUnet(mask, (
                               embed = Chain(
                                             vec,
                                             RandomFourierFeatures(2E, scale),
                                             Reshape(2E, F, :),
                                             GroupDense(2E, E, F, activation),
                                            ),
                               e1  = GroupDense(I, A, F),
                               e2  = GroupDense(A, B, F),
                               e3  = GroupDense(B, C, F),
                               d3  = GroupDense(C, B, F),
                               d2  = GroupDense(2B, A, F),
                               d1  = GroupDense(2A, I, F),
                               _e1 = GroupDense(E, A, F),
                               _e2 = GroupDense(E, B, F),
                               _e3 = GroupDense(E, C, F),
                               _d3 = GroupDense(E, B, F),
                               _d2 = GroupDense(E, A, F),
                               g1  = InstanceNorm(F, activation),
                               g2  = InstanceNorm(F, activation),
                               g3  = InstanceNorm(F, activation),
                               _g3 = InstanceNorm(F, activation),
                               _g2 = InstanceNorm(F, activation),
                              ))
end


"""
One t_j for each x_j, x and t have the same size
Parallel by groups
"""
function (unet::GroupMlpUnet)(x::AbstractArray{T, 3}, t::AbstractArray{T, 3}) where T
    @unpack e1, e2, e3, d3, d2, d1, _e1, _e2, _e3, _d3, _d2, g1, g2, g3, _g3, _g2 = unet.layers
    mask = unet.mask
    embed = unet.layers.embed(t)
    x0 = x .* mask
    t0 = embed
    #-- Encoder
    h1 = @> e1(x0) .+ _e1(t0) g1
    h2 = @> e2(h1) .+ _e2(t0) g2
    h3 = @> e3(h2) .+ _e3(t0) g3
    #-- Decoder
    h = @> d3(h3) .+ _d3(t0) _g3
    h = @> d2(vcat(h, h2)) .+ _d2(t0) _g2
    h = @> d1(vcat(h, h1))
    #-- Scaling Factor
    σ_t = marginal_prob_std(t)
    mask .* h ./ σ_t
end

