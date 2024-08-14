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

struct GroupMlpUnet
    mask::AbstractMatrix{Bool}
    layers::NamedTuple
end

@functor GroupMlpUnet
@showfields GroupMlpUnet

function GroupMlpUnet(adjmat::AbstractMatrix{T}, scale=30.0f0) where {T}
    X = input_dim = size(adjmat, 1)
    F = X  # the first node j has only marginal prob, mask for conditional is zeros
    A, B, C, D, E = 2X, X, X÷2+1, X÷4+1, X
    paj_mask = adjmat
    j_mask = I(F)
    @≥ j_mask, paj_mask Matrix{Bool}.()
    mask = max.(j_mask, paj_mask)
    return GroupMlpUnet(mask, (
                               embed = Chain(
                                             vec,
                                             RandomFourierFeatures(2E, scale),
                                             Reshape(1, 2E*F, :),
                                             Conv((1,), 2E*F=>E*F, groups=F, swish),
                                            ),
                               #-- Encoding
                               e1 = Conv((1,), X*F=>A*F, groups=F),
                               e2 = Conv((1,), A*F=>B*F, groups=F),
                               e3 = Conv((1,), B*F=>C*F, groups=F),
                               e4 = Conv((1,), C*F=>D*F, groups=F),
                               #-- Decoding
                               d4 = Conv((1,), D*F=>C*F, groups=F),
                               d3 = Conv((2,), C*F=>B*F, groups=F),  # vcat with shortcut connection
                               d2 = Conv((2,), B*F=>A*F, groups=F),
                               d1 = Conv((2,), A*F=>X*F, groups=F),
                               #-- Condition
                               c1 = Conv((1,), E*F=>A*F, groups=F),
                               c2 = Conv((1,), E*F=>B*F, groups=F),
                               c3 = Conv((1,), E*F=>C*F, groups=F),
                               c4 = Conv((1,), E*F=>D*F, groups=F),
                               c5 = Conv((1,), E*F=>C*F, groups=F),
                               c6 = Conv((1,), E*F=>B*F, groups=F),
                               c7 = Conv((1,), E*F=>A*F, groups=F),
                               #-- GroupNorm
                               g1 = GroupNorm(A*F, F, swish),
                               g2 = GroupNorm(B*F, F, swish),
                               g3 = GroupNorm(C*F, F, swish),
                               g4 = GroupNorm(D*F, F, swish),
                               g5 = GroupNorm(C*F, F, swish),
                               g6 = GroupNorm(B*F, F, swish),
                               g7 = GroupNorm(A*F, F, swish),
                              ))
end

"""
One t_j for each x_j, x and t have the same size
Parallel by groups
"""
function (unet::GroupMlpUnet)(x::AbstractArray{T, 3}, t::AbstractArray{T, 3}) where T
    @unpack e1, e2, e3, e4, d4, d3, d2, d1, c1, c2, c3, c4, c5, c6, c7, g1, g2, g3, g4, g5, g6, g7 = unet.layers
    mask = @> unet.mask vec unsqueeze(1)
    #-- Embedding
    embed = unet.layers.embed(t)
    #-- Encoder
    X = size(x, 1)
    @≥ x reshape(1, X*X, :)
    x0 = x .* mask
    t0 = embed .* mask
    # mask |> printmask
    @assert size(x0)[1:2] == size(t0)[1:2] == (1, X*X)
    h1 = @> e1(x0) .+ c1(t0) g1
    h2 = @> e2(h1) .+ c2(t0) g2
    h3 = @> e3(h2) .+ c3(t0) g3
    h4 = @> e4(h3) .+ c4(t0) g4
    #-- Decoder
    h = @> d4(h4) .+ c5(t0) g5
    h = @> d3(cat(h, h3; dims=1)) .+ c6(t0) g6
    h = @> d2(cat(h, h2; dims=1)) .+ c7(t0) g7
    h = @> d1(cat(h, h1; dims=1))
    #-- Scaling Factor
    h = @> mask .* h reshape(X, X, :)
    σ_t = marginal_prob_std(t)
    h ./ σ_t
end

