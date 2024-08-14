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

struct GroupMlpRegression{T}
    mask::AbstractMatrix{Bool}
    mlp::T
end

@functor GroupMlpRegression
@showfields GroupMlpRegression

function GroupMlpRegression(adjmat::AbstractMatrix{T}) where {T}
    mask = @> adjmat Matrix{Bool}
    X = input_dim = size(adjmat, 1)
    F = X  # the first node j has only marginal prob, mask for conditional is zeros
    A, B, C, D, E = 2X, X, X÷2+1, X÷4+1, X
    return GroupMlpRegression(mask, Chain(
                               Conv((1,), X*F=>A*F, groups=F), # GroupNorm(A*F, F, swish),
                               Conv((1,), A*F=>B*F, groups=F), # GroupNorm(B*F, F, swish),
                               Conv((1,), B*F=>C*F, groups=F), # GroupNorm(C*F, F, swish),
                               Conv((1,), C*F=>D*F, groups=F), # GroupNorm(D*F, F, swish),
                               Conv((1,), D*F=>1*F, groups=F),
                              ))
end

"""
"""
function (mlp::GroupMlpRegression)(x::AbstractArray{T, 3}) where T
    # mlp.mask |> printmask
    mask = @> mlp.mask vec unsqueeze(1)
    F = size(x, 1)
    @≥ x reshape(1, F*F, :)
    x0 = x .* mask
    x̂ = mlp.mlp(x0)
end

